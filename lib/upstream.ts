import { isRecord, type Json } from './providers.ts';
import { sseEvent } from './wire.ts';

type AnyRecord = Record<string, unknown>;

function asRecord(value: unknown): AnyRecord | undefined {
  return isRecord(value) ? value : undefined;
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function asString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

function parseObject(value: unknown): AnyRecord {
  if (isRecord(value)) return value;
  if (typeof value === 'string') {
    try {
      const parsed: unknown = JSON.parse(value);
      return isRecord(parsed) ? parsed : {};
    } catch {
      return {};
    }
  }
  return {};
}

function jsonString(value: unknown): string {
  try {
    const result = JSON.stringify(value);
    return result === undefined ? '{}' : result;
  } catch {
    return '{}';
  }
}

function textValue(value: unknown): string {
  if (typeof value === 'string') return value;
  if (value === null || value === undefined) return '';
  return jsonString(value);
}

function numeric(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function imageDataUrl(value: unknown): { mimeType: string; data: string } | undefined {
  if (typeof value !== 'string') return undefined;
  const match = /^data:([^;,]+);base64,([\s\S]*)$/i.exec(value);
  if (!match) return undefined;
  return { mimeType: match[1], data: match[2] };
}

function anthropicImageDataUrl(value: unknown): AnyRecord | undefined {
  const image = imageDataUrl(value);
  if (!image) return undefined;
  return {
    type: 'image',
    source: {
      type: 'base64',
      media_type: image.mimeType,
      data: image.data,
    },
  };
}

function contentText(value: unknown): string {
  if (typeof value === 'string') return value;
  return asArray(value)
    .map((part) => {
      const record = asRecord(part);
      return record && record.type === 'text' ? asString(record.text) : '';
    })
    .join('');
}

function defined(target: AnyRecord, key: string, value: unknown): void {
  if (value !== undefined) target[key] = value;
}

export function openAIToGemini(body: Json, model: string): Json {
  const contents: AnyRecord[] = [];
  const systemTexts: string[] = [];

  for (const item of asArray(body.messages)) {
    const message = asRecord(item);
    if (!message) continue;

    const role = asString(message.role);
    if (role === 'system' || role === 'developer') {
      const text = contentText(message.content);
      if (text) systemTexts.push(text);
      continue;
    }

    if (role === 'tool') {
      contents.push({
        role: 'function',
        parts: [{
          functionResponse: {
            name: '',
            response: { content: textValue(message.content) },
          },
        }],
      });
      continue;
    }

    if (role !== 'user' && role !== 'assistant') continue;
    const parts: AnyRecord[] = [];
    const rawContent = message.content;

    if (typeof rawContent === 'string') {
      if (rawContent) parts.push({ text: rawContent });
    } else {
      for (const rawPart of asArray(rawContent)) {
        const part = asRecord(rawPart);
        if (!part) continue;
        if (part.type === 'text' && typeof part.text === 'string') {
          parts.push({ text: part.text });
        } else if (part.type === 'image_url') {
          const imageUrl = asRecord(part.image_url);
          const image = imageDataUrl(imageUrl?.url);
          if (image) parts.push({ inlineData: image });
        }
      }
    }

    if (role === 'assistant') {
      for (const rawCall of asArray(message.tool_calls)) {
        const call = asRecord(rawCall);
        const fn = asRecord(call?.function);
        if (!fn) continue;
        parts.push({
          functionCall: {
            name: asString(fn.name),
            args: parseObject(fn.arguments),
          },
        });
      }
    }

    contents.push({
      role: role === 'assistant' ? 'model' : 'user',
      parts,
    });
  }

  const result: AnyRecord = { contents };
  if (systemTexts.length) result.systemInstruction = { parts: [{ text: systemTexts.join('\n') }] };

  const declarations: AnyRecord[] = [];
  for (const rawTool of asArray(body.tools)) {
    const tool = asRecord(rawTool);
    const fn = asRecord(tool?.function);
    if (!fn) continue;
    const declaration: AnyRecord = { name: asString(fn.name) };
    defined(declaration, 'description', fn.description);
    defined(declaration, 'parameters', fn.parameters);
    declarations.push(declaration);
  }
  if (declarations.length) result.tools = [{ functionDeclarations: declarations }];

  const generationConfig: AnyRecord = {};
  defined(generationConfig, 'temperature', body.temperature);
  defined(generationConfig, 'topP', body.top_p);
  defined(generationConfig, 'maxOutputTokens', body.max_tokens);
  if (typeof body.stop === 'string') generationConfig.stopSequences = [body.stop];
  else if (Array.isArray(body.stop)) generationConfig.stopSequences = body.stop;

  const responseFormat = asRecord(body.response_format);
  if (responseFormat?.type === 'json_object' || responseFormat?.type === 'json_schema') {
    generationConfig.responseMimeType = 'application/json';
  }
  if (Object.keys(generationConfig).length) result.generationConfig = generationConfig;

  void model;
  return result as Json;
}

export function geminiFinishToOpenAI(finish: unknown): string {
  switch (finish) {
    case 'MAX_TOKENS':
      return 'length';
    case 'SAFETY':
      return 'content_filter';
    case 'STOP':
    default:
      return 'stop';
  }
}

export function anthropicStopToOpenAI(stop: unknown): string {
  switch (stop) {
    case 'max_tokens':
      return 'length';
    case 'tool_use':
      return 'tool_calls';
    case 'refusal':
      return 'content_filter';
    case 'end_turn':
    default:
      return 'stop';
  }
}

function geminiUsage(body: AnyRecord): AnyRecord | undefined {
  const usage = asRecord(body.usageMetadata);
  if (!usage) return undefined;
  const prompt = numeric(usage.promptTokenCount);
  const completion = numeric(usage.candidatesTokenCount);
  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: numeric(usage.totalTokenCount, prompt + completion),
  };
}

function geminiParts(body: AnyRecord): unknown[] {
  const candidate = asRecord(asArray(body.candidates)[0]);
  const content = asRecord(candidate?.content);
  return asArray(content?.parts);
}

export function geminiToOpenAI(body: Json, model: string): Json {
  const id = typeof body.id === 'string' ? body.id : 'gemini';
  const candidate = asRecord(asArray(body.candidates)[0]);
  let content = '';
  const toolCalls: AnyRecord[] = [];

  for (const rawPart of geminiParts(body)) {
    const part = asRecord(rawPart);
    if (!part) continue;
    if (typeof part.text === 'string') content += part.text;
    const call = asRecord(part.functionCall);
    if (call) {
      toolCalls.push({
        id: `call_${toolCalls.length}`,
        type: 'function',
        function: {
          name: asString(call.name),
          arguments: jsonString(parseObject(call.args)),
        },
      });
    }
  }

  const message: AnyRecord = { role: 'assistant', content };
  if (toolCalls.length) message.tool_calls = toolCalls;

  const result: AnyRecord = {
    id: `chatcmpl_${id}`,
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{
      index: 0,
      message,
      finish_reason: geminiFinishToOpenAI(candidate?.finishReason),
    }],
  };
  const usage = geminiUsage(body);
  if (usage) result.usage = usage;
  return result as Json;
}

export function geminiToAnthropic(body: Json, model: string): Json {
  const id = typeof body.id === 'string' ? body.id : 'gemini';
  const candidate = asRecord(asArray(body.candidates)[0]);
  const content: AnyRecord[] = [];

  for (const rawPart of geminiParts(body)) {
    const part = asRecord(rawPart);
    if (!part) continue;
    if (typeof part.text === 'string') content.push({ type: 'text', text: part.text });
    const call = asRecord(part.functionCall);
    if (call) {
      content.push({
        type: 'tool_use',
        id: `toolu_${content.filter((block) => block.type === 'tool_use').length}`,
        name: asString(call.name),
        input: parseObject(call.args),
      });
    }
  }

  const usageMetadata = asRecord(body.usageMetadata);
  return {
    id: `msg_${id}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: candidate?.finishReason === 'MAX_TOKENS'
      ? 'max_tokens'
      : candidate?.finishReason === 'SAFETY'
        ? 'refusal'
        : 'end_turn',
    stop_sequence: null,
    usage: {
      input_tokens: numeric(usageMetadata?.promptTokenCount),
      output_tokens: numeric(usageMetadata?.candidatesTokenCount),
    },
  } as Json;
}

export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const systems: string[] = [];
  const messages: AnyRecord[] = [];

  for (const rawMessage of asArray(body.messages)) {
    const message = asRecord(rawMessage);
    if (!message) continue;
    const role = asString(message.role);

    if (role === 'system' || role === 'developer') {
      const text = contentText(message.content);
      if (text) systems.push(text);
      continue;
    }

    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [{
          type: 'tool_result',
          tool_use_id: asString(message.tool_call_id),
          content: textValue(message.content),
        }],
      });
      continue;
    }

    if (role !== 'user' && role !== 'assistant') continue;
    const content: AnyRecord[] = [];
    const rawContent = message.content;

    if (typeof rawContent === 'string') {
      if (rawContent) content.push({ type: 'text', text: rawContent });
    } else {
      for (const rawPart of asArray(rawContent)) {
        const part = asRecord(rawPart);
        if (!part) continue;
        if (part.type === 'text' && typeof part.text === 'string') {
          content.push({ type: 'text', text: part.text });
        } else if (part.type === 'image_url') {
          const imageUrl = asRecord(part.image_url);
          const image = anthropicImageDataUrl(imageUrl?.url);
          if (image) content.push(image);
        }
      }
    }

    if (role === 'assistant') {
      for (const rawCall of asArray(message.tool_calls)) {
        const call = asRecord(rawCall);
        const fn = asRecord(call?.function);
        if (!fn) continue;
        content.push({
          type: 'tool_use',
          id: asString(call?.id),
          name: asString(fn.name),
          input: parseObject(fn.arguments),
        });
      }
    }

    messages.push({ role, content });
  }

  const result: AnyRecord = { model, messages };
  if (systems.length) result.system = systems.join('\n');

  const tools: AnyRecord[] = [];
  for (const rawTool of asArray(body.tools)) {
    const tool = asRecord(rawTool);
    const fn = asRecord(tool?.function);
    if (!fn) continue;
    tools.push({
      name: asString(fn.name),
      description: typeof fn.description === 'string' ? fn.description : '',
      input_schema: isRecord(fn.parameters)
        ? fn.parameters
        : { type: 'object', properties: {} },
    });
  }
  if (tools.length) result.tools = tools;

  if (body.tool_choice === 'auto') {
    result.tool_choice = 'auto';
  } else if (body.tool_choice === 'required') {
    result.tool_choice = { type: 'any' };
  } else {
    const choice = asRecord(body.tool_choice);
    const fn = asRecord(choice?.function);
    if (choice?.type === 'function' && fn && typeof fn.name === 'string') {
      result.tool_choice = { type: 'tool', name: fn.name };
    }
  }

  defined(result, 'max_tokens', body.max_tokens);
  defined(result, 'temperature', body.temperature);
  defined(result, 'top_p', body.top_p);
  if (typeof body.stop === 'string') result.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop)) result.stop_sequences = body.stop;
  if (body.stream === true) result.stream = true;

  return result as Json;
}

export function anthropicResponseToOpenAI(body: Json, model: string): Json {
  const id = typeof body.id === 'string' ? body.id : 'anthropic';
  const blocks = asArray(body.content);
  const texts: string[] = [];
  const toolCalls: AnyRecord[] = [];

  for (const rawBlock of blocks) {
    const block = asRecord(rawBlock);
    if (!block) continue;
    if (block.type === 'text' && typeof block.text === 'string') {
      texts.push(block.text);
    } else if (block.type === 'tool_use') {
      toolCalls.push({
        id: asString(block.id),
        type: 'function',
        function: {
          name: asString(block.name),
          arguments: jsonString(parseObject(block.input)),
        },
      });
    }
  }

  const joinedText = texts.join('\n');
  const message: AnyRecord = {
    role: 'assistant',
    content: toolCalls.length && !joinedText ? null : joinedText,
  };
  if (toolCalls.length) message.tool_calls = toolCalls;

  const usage = asRecord(body.usage);
  const prompt = numeric(usage?.input_tokens);
  const completion = numeric(usage?.output_tokens);
  return {
    id: `chatcmpl_${id}`,
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{
      index: 0,
      message,
      finish_reason: anthropicStopToOpenAI(body.stop_reason),
    }],
    usage: {
      prompt_tokens: prompt,
      completion_tokens: completion,
      total_tokens: prompt + completion,
    },
  } as Json;
}

type SseFrame = { event: string; data: string };

async function* readSseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let event = '';
  let dataLines: string[] = [];

  const consumeLine = (line: string): SseFrame | undefined => {
    if (line === '') {
      if (dataLines.length === 0) {
        event = '';
        return undefined;
      }
      const frame = { event, data: dataLines.join('\n') };
      event = '';
      dataLines = [];
      return frame;
    }
    if (line.startsWith(':')) return undefined;
    const colon = line.indexOf(':');
    const field = colon < 0 ? line : line.slice(0, colon);
    let value = colon < 0 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'event') event = value;
    else if (field === 'data') dataLines.push(value);
    return undefined;
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });

      let newline = buffer.indexOf('\n');
      while (newline >= 0) {
        let line = buffer.slice(0, newline);
        if (line.endsWith('\r')) line = line.slice(0, -1);
        buffer = buffer.slice(newline + 1);
        const frame = consumeLine(line);
        if (frame) yield frame;
        newline = buffer.indexOf('\n');
      }

      if (done) break;
    }

    if (buffer.length) {
      const line = buffer.endsWith('\r') ? buffer.slice(0, -1) : buffer;
      const frame = consumeLine(line);
      if (frame) yield frame;
    }
    const finalFrame = consumeLine('');
    if (finalFrame) yield finalFrame;
  } finally {
    reader.releaseLock();
  }
}

function openAIChunk(delta: AnyRecord, finishReason: unknown = null, extra?: AnyRecord): string {
  const chunk: AnyRecord = {
    id: 'chatcmpl_gemini_stream',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model: '',
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  };
  if (extra) Object.assign(chunk, extra);
  return jsonString(chunk);
}

export async function* geminiEventsToOpenAI(
  upstream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> {
  let toolIndex = 0;
  let finished = false;

  for await (const frame of readSseFrames(upstream)) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(frame.data);
    } catch {
      continue;
    }
    const body = asRecord(parsed);
    if (!body) continue;
    const candidate = asRecord(asArray(body.candidates)[0]);

    for (const rawPart of geminiParts(body)) {
      const part = asRecord(rawPart);
      if (!part) continue;
      if (typeof part.text === 'string') {
        const chunk = JSON.parse(openAIChunk({ content: part.text })) as AnyRecord;
        chunk.model = model;
        yield `data: ${jsonString(chunk)}\n\n`;
      }
      const call = asRecord(part.functionCall);
      if (call) {
        const index = toolIndex++;
        const chunk = JSON.parse(openAIChunk({
          tool_calls: [{
            index,
            id: `call_${index}`,
            type: 'function',
            function: {
              name: asString(call.name),
              arguments: jsonString(parseObject(call.args)),
            },
          }],
        })) as AnyRecord;
        chunk.model = model;
        yield `data: ${jsonString(chunk)}\n\n`;
      }
    }

    if (candidate?.finishReason !== undefined) {
      const extra: AnyRecord = {};
      const usage = geminiUsage(body);
      if (usage) extra.usage = usage;
      const chunk = JSON.parse(openAIChunk({}, geminiFinishToOpenAI(candidate.finishReason), extra)) as AnyRecord;
      chunk.model = model;
      yield `data: ${jsonString(chunk)}\n\n`;
      finished = true;
      break;
    }
  }

  if (!finished) {
    const chunk = JSON.parse(openAIChunk({}, 'stop')) as AnyRecord;
    chunk.model = model;
    yield `data: ${jsonString(chunk)}\n\n`;
  }
  yield 'data: [DONE]\n\n';
}

function anthropicFinishChunk(
  model: string,
  finishReason: string,
  inputTokens: number,
  outputTokens: number,
): string {
  return `data: ${jsonString({
    id: 'chatcmpl_anthropic_stream',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model,
    choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
    usage: {
      prompt_tokens: inputTokens,
      completion_tokens: outputTokens,
      total_tokens: inputTokens + outputTokens,
    },
  })}\n\n`;
}

export async function* anthropicEventsToOpenAI(
  upstream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> {
  let inputTokens = 0;
  let outputTokens = 0;
  let finishEmitted = false;
  let messageStopped = false;
  const blocks = new Map<number, { type: string; id: string; name: string }>();

  for await (const frame of readSseFrames(upstream)) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(frame.data);
    } catch {
      continue;
    }
    const body = asRecord(parsed);
    if (!body) continue;

    if (frame.event === 'message_start') {
      const message = asRecord(body.message);
      const usage = asRecord(message?.usage);
      inputTokens = numeric(usage?.input_tokens, inputTokens);
    } else if (frame.event === 'content_block_start') {
      const index = numeric(body.index, -1);
      const block = asRecord(body.content_block);
      if (index < 0 || !block) continue;
      const type = asString(block.type);
      const id = asString(block.id);
      const name = asString(block.name);
      blocks.set(index, { type, id, name });
      if (type === 'tool_use') {
        yield `data: ${jsonString({
          id: 'chatcmpl_anthropic_stream',
          object: 'chat.completion.chunk',
          created: 1767225600,
          model,
          choices: [{
            index: 0,
            delta: {
              tool_calls: [{
                index,
                id,
                type: 'function',
                function: { name, arguments: '' },
              }],
            },
            finish_reason: null,
          }],
        })}\n\n`;
      }
    } else if (frame.event === 'content_block_delta') {
      const index = numeric(body.index, -1);
      const delta = asRecord(body.delta);
      if (!delta || index < 0) continue;
      if (delta.type === 'text_delta' && typeof delta.text === 'string') {
        yield `data: ${jsonString({
          id: 'chatcmpl_anthropic_stream',
          object: 'chat.completion.chunk',
          created: 1767225600,
          model,
          choices: [{ index: 0, delta: { content: delta.text }, finish_reason: null }],
        })}\n\n`;
      } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string') {
        yield `data: ${jsonString({
          id: 'chatcmpl_anthropic_stream',
          object: 'chat.completion.chunk',
          created: 1767225600,
          model,
          choices: [{
            index: 0,
            delta: {
              tool_calls: [{
                index,
                function: { arguments: delta.partial_json },
              }],
            },
            finish_reason: null,
          }],
        })}\n\n`;
      }
    } else if (frame.event === 'message_delta') {
      const delta = asRecord(body.delta);
      const usage = asRecord(body.usage);
      outputTokens = numeric(usage?.output_tokens, outputTokens);
      if (delta && delta.stop_reason !== undefined) {
        yield anthropicFinishChunk(
          model,
          anthropicStopToOpenAI(delta.stop_reason),
          inputTokens,
          outputTokens,
        );
        finishEmitted = true;
      }
    } else if (frame.event === 'message_stop') {
      messageStopped = true;
      break;
    }
  }

  if (!messageStopped && !finishEmitted) {
    yield anthropicFinishChunk(model, 'stop', inputTokens, outputTokens);
  }
  yield 'data: [DONE]\n\n';
  void blocks;
}

function geminiFinishToAnthropic(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY') return 'refusal';
  return 'end_turn';
}

export async function* geminiEventsToAnthropic(
  upstream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> {
  yield sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_gemini_stream',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: 0, output_tokens: 0 },
    },
  });

  let blockIndex = 0;
  let textBlockIndex: number | undefined;
  let finished = false;
  let outputTokens = 0;

  const closeTextBlock = function* (): Generator<string> {
    if (textBlockIndex !== undefined) {
      yield sseEvent('content_block_stop', {
        type: 'content_block_stop',
        index: textBlockIndex,
      });
      textBlockIndex = undefined;
    }
  };

  for await (const frame of readSseFrames(upstream)) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(frame.data);
    } catch {
      continue;
    }
    const body = asRecord(parsed);
    if (!body) continue;
    const candidate = asRecord(asArray(body.candidates)[0]);
    const usage = asRecord(body.usageMetadata);
    outputTokens = numeric(usage?.candidatesTokenCount, outputTokens);

    for (const rawPart of geminiParts(body)) {
      const part = asRecord(rawPart);
      if (!part) continue;
      if (typeof part.text === 'string') {
        if (textBlockIndex === undefined) {
          textBlockIndex = blockIndex++;
          yield sseEvent('content_block_start', {
            type: 'content_block_start',
            index: textBlockIndex,
            content_block: { type: 'text', text: '' },
          });
        }
        yield sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index: textBlockIndex,
          delta: { type: 'text_delta', text: part.text },
        });
      }

      const call = asRecord(part.functionCall);
      if (call) {
        const index = blockIndex++;
        yield sseEvent('content_block_start', {
          type: 'content_block_start',
          index,
          content_block: {
            type: 'tool_use',
            id: `toolu_${index}`,
            name: asString(call.name),
            input: {},
          },
        });
        yield sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: {
            type: 'input_json_delta',
            partial_json: jsonString(parseObject(call.args)),
          },
        });
        yield sseEvent('content_block_stop', {
          type: 'content_block_stop',
          index,
        });
      }
    }

    if (candidate?.finishReason !== undefined) {
      yield* closeTextBlock();
      yield sseEvent('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: geminiFinishToAnthropic(candidate.finishReason), stop_sequence: null },
        usage: { output_tokens: outputTokens },
      });
      yield sseEvent('message_stop', { type: 'message_stop' });
      finished = true;
      break;
    }
  }

  if (!finished) {
    yield* closeTextBlock();
    yield sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: { output_tokens: outputTokens },
    });
    yield sseEvent('message_stop', { type: 'message_stop' });
  }
}

export async function callGemini(
  base: string,
  key: string,
  model: string,
  body: Json,
  stream: boolean,
): Promise<Response> {
  const action = stream ? ':streamGenerateContent?alt=sse' : ':generateContent';
  const url = `${base.replace(/\/+$/, '')}/v1beta/models/${encodeURIComponent(model)}${action}`;
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  if (key) headers['x-goog-api-key'] = key;

  return fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(30000),
  });
}

export async function callAnthropic(
  base: string,
  key: string,
  body: Json,
): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'anthropic-version': '2023-06-01',
  };
  if (key) headers['x-api-key'] = key;

  return fetch(`${base.replace(/\/+$/, '')}/v1/messages`, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(30000),
  });
}
