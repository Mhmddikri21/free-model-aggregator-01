# free-model-aggregator-01

Entry for the make.fun bounty **"The best free AI model aggregator"** ($500, top 20 paid).

One server, two client wires (OpenAI Chat Completions + Anthropic Messages, JSON and
streaming), routing each request to the best-rated **free** model that meets its
requirements across ten simulated free-tier providers (OpenAI, Anthropic and Gemini
wires), with adaptive failover: a 402 marks that provider's allowance gone for the
trial, a 429/5xx moves to the next candidate. Fully deterministic — no clock, no
unseeded randomness — so a replay leaves the same state export.

## Layout

- `main.ts` — the server: brief/catalog probe, rating-ranked routing, failover,
  register + `/work` poll. Run as `node main.ts` (Node 22, stdlib only);
  `node main.ts --list` prints the catalog document and exits.
- `lib/providers.ts` — brief parsing, catalog probe, capability checks.
- `lib/wire.ts` — OpenAI <-> Anthropic translation and SSE writers.
- `lib/upstream.ts` — Gemini + Anthropic upstream translators (JSON + streaming).
- `run-trial.sh` — local practice runner: starts the kit env for a practice trial,
  runs the bot, scores the exported state (needs the kit at
  `~/workspace/makefun/free-model-aggregator-01-kit`).

## Practice scores (local, published env + scorer, not dev/final scores)

| trial | score | quality | served | wire | failover | catalog |
|---|---|---|---|---|---|---|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 |

20/20 requests valid on all three trials. A repeat run of trial 0 produced a
byte-identical `state.json` (deterministic replay).

## License

MIT — see `LICENSE`.
