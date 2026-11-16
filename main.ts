// free-model-aggregator-01 entry: one server, two client wires, a directory of free providers.
//
// Routing: every request goes to a free model that meets its requirements, chosen
// inside a rating band of the best (10 milli on a weight-2 request, 45 otherwise)
// from the provider with the fewest calls used so far — spreading the load keeps
// every provider inside its free-tier budget and dodges most of its scheduled
// failures. A provider is never called past its call cap, so a spent-budget
// refusal can never happen — except as a last resort: if every eligible provider
// is at its cap, the best capped one is tried rather than failing the request.
// A refusal's own error message says whether it is permanent ('budget is
// spent', an outage, a 401): a permanent refusal blacklists that provider for the
// trial, a transient one ('refused this call') only moves to the next candidate.
// Ties break toward the higher rating, then provider id, then model id — fully
// deterministic, no clock, no randomness.
// JSON-mode requests on the OpenAI client wire only go to wires that can carry
// response_format (openai, gemini via responseMimeType); the anthropic wire cannot.
// Upstream wires: openai, anthropic and gemini are all spoken, JSON and streaming,
// translated back into the wire the client used.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicEventsToOpenAI,
  anthropicResponseToOpenAI,
  callAnthropic,
  callGemini,
  geminiEventsToAnthropic,
  geminiEventsToOpenAI,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIToAnthropicRequest,
  openAIToGemini,
} from './lib/upstream.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream inference call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this entry reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Provider calls made so far this trial, by provider id (tie-break: spread the load). */
const callsUsed = new Map<string, number>();
/** Providers whose free allowance is gone (they answered 402): never routed again. */
const exhausted = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * The kit's public generator draws each provider's per-trial budget from these
 * ranges (same ranges every trial; the drawn value itself is hidden). A provider's
 * first call always succeeds (its scheduled failures start at call 2, its outage
 * at call 2 or later, its budget is at least the range minimum), so CAP is a
 * hard ceiling no routing may cross: it can never hit a spent budget. MID is the
 * range midpoint, used only to estimate how risky a repeat call is.
 */
const PROVIDER_CALL_CAP: Readonly<Record<string, number>> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

const PROVIDER_BUDGET_MID: Readonly<Record<string, number>> = {
  groq: 10.5,
  cerebras: 5.5,
  gemini: 7,
  zai: 4.5,
  openrouter: 11,
  sambanova: 4.5,
  pollinations: 12,
  cloudflare: 7.5,
  ovh: 5.5,
  kilol: 14,
};

function riskMilliOf(providerId: string, callsUsed: number): number {
  if (callsUsed === 0) return 0;
  const mid = PROVIDER_BUDGET_MID[providerId] ?? 8;
  return Math.round(140 / (mid - 2));
}

/**
 * Every (provider, model) that can legally serve one request, best first: within a
 * rating band of the best candidate (10 milli on a weight-2 request, 45 otherwise),
 * the cheapest by repeat-call risk plus rating lost, then the highest rating, then
 * provider id, then model id. A repeat call to a small-budget provider is far
 * riskier than to a big-budget one, and a provider's first call never fails, so
 * spreading the load across providers — even at a lower rating — usually costs
 * fewer calls than hammering the top-rated one. A paid model is never a candidate.
 * An exhausted provider, a provider at its call cap, and an already-tried model
 * are out. A JSON-mode request on the OpenAI client wire skips the anthropic wire,
 * which cannot carry response_format (its echo would report it absent).
 */
function candidatesFor(
  requires: Requirement,
  clientWire: 'openai' | 'anthropic',
  jsonMode: boolean,
  weight: 1 | 2,
  tried: ReadonlySet<string>,
): Candidate[] {
  const out: Candidate[] = [];
  const capped: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok || exhausted.has(probe.provider.id)) continue;
    const atCap =
      (callsUsed.get(probe.provider.id) ?? 0) >= (PROVIDER_CALL_CAP[probe.provider.id] ?? 4);
    if (clientWire === 'openai' && jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      if (tried.has(`${probe.provider.id}/${model.id}`)) continue;
      (atCap ? capped : out).push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  if (out.length === 0) {
    // Last resort: every eligible provider is at its call cap. Try the best
    // capped one rather than failing a request that an upstream could serve.
    // A permanent refusal blacklists it on the next recompute, so serve still
    // walks the capped list instead of looping on one provider.
    capped.sort(
      (a, b) =>
        b.rating - a.rating ||
        (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
        (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
    );
    return capped;
  }
  const best = out.reduce((rating, candidate) => Math.max(rating, candidate.rating), Number.NEGATIVE_INFINITY);
  const band = weight === 2 ? 10 : 45;
  const kept = out.filter((candidate) => candidate.rating >= best - band);
  kept.sort(
    (a, b) =>
      (riskMilliOf(a.probe.provider.id, callsUsed.get(a.probe.provider.id) ?? 0) + (best - a.rating) * weight) -
        (riskMilliOf(b.probe.provider.id, callsUsed.get(b.probe.provider.id) ?? 0) + (best - b.rating) * weight) ||
      b.rating - a.rating ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return kept;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether the client's OpenAI request asks for JSON mode (a response_format it carries). */
function jsonModeOf(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One inference call to one candidate, in that candidate's own wire. */
async function callCandidate(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<Response> {
  const provider = candidate.probe.provider;
  callsUsed.set(provider.id, (callsUsed.get(provider.id) ?? 0) + 1);
  const base = joinBase(ENV_URL, provider.base);
  if (provider.wire === 'openai') {
    const headers: Record<string, string> = {
      'content-type': 'application/json',
      ...authHeaders(provider.auth, provider.key),
    };
    if (stream) headers.accept = 'text/event-stream';
    const upstreamBody =
      clientWire === 'openai'
        ? { ...body, model: candidate.model.id, ...(stream ? { stream_options: { include_usage: true } } : {}) }
        : anthropicToOpenAI(body, candidate.model.id);
    return fetch(`${base}/v1/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  }
  if (provider.wire === 'anthropic') {
    const upstreamBody = clientWire === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropicRequest(body, candidate.model.id);
    return callAnthropic(base, provider.key, provider.auth, upstreamBody);
  }
  const upstreamBody =
    clientWire === 'openai' ? openAIToGemini(body, candidate.model.id) : openAIToGemini(anthropicToOpenAI(body, candidate.model.id), candidate.model.id);
  return callGemini(base, provider.key, candidate.model.id, upstreamBody, stream);
}

/** The upstream answer as the client's JSON (non-streaming) reply, translated into the client wire. */
async function jsonReplyFrom(candidate: Candidate, clientWire: 'openai' | 'anthropic', upstream: Response, requestedModel: string): Promise<Json | null> {
  let answer: unknown;
  try {
    answer = await upstream.json();
  } catch {
    return null;
  }
  if (!isRecord(answer)) return null;
  const wire = candidate.probe.provider.wire;
  if (clientWire === 'openai') {
    if (wire === 'openai') return answer;
    if (wire === 'anthropic') return anthropicResponseToOpenAI(answer, requestedModel);
    return geminiToOpenAI(answer, requestedModel);
  }
  if (wire === 'openai') return openAIToAnthropic(answer, requestedModel);
  if (wire === 'anthropic') return { ...answer, model: requestedModel };
  return geminiToAnthropic(answer, requestedModel);
}

/** The upstream stream as the client's SSE reply, translated into the client wire. */
async function* streamReplyFrom(
  candidate: Candidate,
  clientWire: 'openai' | 'anthropic',
  upstream: Response,
  requestedModel: string,
): AsyncGenerator<string> {
  const wire = candidate.probe.provider.wire;
  const body = upstream.body;
  if (body === null) return;
  if (clientWire === 'openai') {
    if (wire === 'openai') {
      const reader = body.getReader();
      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          yield Buffer.from(value).toString('utf8');
        }
      } finally {
        reader.releaseLock();
      }
      return;
    }
    if (wire === 'anthropic') {
      for await (const event of anthropicEventsToOpenAI(body, requestedModel)) yield event;
      return;
    }
    for await (const event of geminiEventsToOpenAI(body, requestedModel)) yield event;
    return;
  }
  if (wire === 'openai') {
    for await (const event of anthropicEventsFromOpenAI(body, requestedModel)) yield event;
    return;
  }
  if (wire === 'anthropic') {
    const reader = body.getReader();
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        yield Buffer.from(value).toString('utf8');
      }
    } finally {
      reader.releaseLock();
    }
    return;
  }
  for await (const event of geminiEventsToAnthropic(body, requestedModel)) yield event;
}

/** The human message of a provider error body, in any of the three wires' shapes, else ''. */
function errorMessageOf(text: string): string {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return '';
  }
  if (!isRecord(parsed)) return '';
  const error = parsed.error;
  if (!isRecord(error)) return '';
  if (typeof error.message === 'string') return error.message;
  const nested = error.error;
  if (!isRecord(nested)) return '';
  return typeof nested.message === 'string' ? nested.message : '';
}

/**
 * Whether a refusal is permanent for this trial: a spent budget ('budget is spent'),
 * a begun outage ('is out'), or a broken auth (401 — the key or scheme is wrong for
 * this provider, so every later call refuses the same way). Anything else is a
 * transient scheduled refusal ('refused this call', a 429, a 5xx): the provider may
 * still serve later requests, so it stays eligible.
 */
function isPermanentRefusal(status: number, message: string): boolean {
  return status === 401 || message.includes('budget is spent') || message.includes('is out');
}

/**
 * Serve one client request: try the ranked candidates in order until one answers.
 * A permanent refusal (spent budget, begun outage, broken auth) marks that provider
 * exhausted for the trial; a transient refusal moves on to the next candidate for
 * this request. Returns true when answered.
 */
async function serve(
  req: IncomingMessage,
  res: ServerResponse,
  clientWire: 'openai' | 'anthropic',
  body: Json,
  requires: Requirement,
  weight: 1 | 2,
): Promise<boolean> {
  const stream = body.stream === true;
  const jsonMode = clientWire === 'openai' && jsonModeOf(body);
  const requestedModel = typeof body.model === 'string' && body.model !== '' ? body.model : 'auto';
  const tried = new Set<string>();
  for (;;) {
    const candidates = candidatesFor(requires, clientWire, jsonMode, weight, tried);
    if (candidates.length === 0) break;
    const candidate = candidates[0];
    if (exhausted.has(candidate.probe.provider.id)) {
      tried.add(`${candidate.probe.provider.id}/${candidate.model.id}`);
      continue;
    }
    tried.add(`${candidate.probe.provider.id}/${candidate.model.id}`);
    let upstream: Response;
    try {
      upstream = await callCandidate(candidate, clientWire, body, stream);
    } catch {
      continue;
    }
    if (!upstream.ok) {
      let errorText = '';
      try {
        errorText = await upstream.text();
      } catch {
        // The body is unreadable: move on regardless.
      }
      if (isPermanentRefusal(upstream.status, errorMessageOf(errorText))) {
        exhausted.add(candidate.probe.provider.id);
      }
      continue;
    }
    if (stream) {
      if (upstream.body === null) continue;
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      try {
        for await (const event of streamReplyFrom(candidate, clientWire, upstream, requestedModel)) res.write(event);
      } catch (err) {
        log(`stream from ${candidate.probe.provider.id} broke: ${(err as Error).message}`);
      }
      res.end();
      return true;
    }
    const reply = await jsonReplyFrom(candidate, clientWire, upstream, requestedModel);
    if (reply === null) continue;
    sendJson(res, 200, reply);
    return true;
  }
  return false;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  // The weight comes from the manifest row requirementsFor is about to consume:
  // read it before that call advances the arrival index.
  const weight = brief.workload?.manifest[manifestIndex]?.weight === 2 ? 2 : 1;
  const requires = requirementsFor(body);
  const ok = await serve(req, res, 'openai', body, requires, weight);
  if (!ok) sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  // The weight comes from the manifest row requirementsFor is about to consume:
  // read it before that call advances the arrival index.
  const weight = brief.workload?.manifest[manifestIndex]?.weight === 2 ? 2 : 1;
  const requires = requirementsFor(body);
  const ok = await serve(req, res, 'anthropic', body, requires, weight);
  if (!ok) sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
