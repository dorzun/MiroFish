# MiroFish — dorzun fork

Upstream: [666ghj/MiroFish](https://github.com/666ghj/MiroFish) — a multi-agent
public-opinion simulation engine (OASIS-based): seed document → knowledge
graph → agent personas → parallel Twitter/Reddit simulation → prediction
report.

This fork makes it run **fully self-hosted in English, with the host agent
(Muse Spark) as the only "LLM"** — no OpenAI key, no Zep Cloud account.

## What changed vs upstream

1. **Full English localization.** Every backend prompt, API error, log line,
   generator and the whole Vue frontend are in English; `en` is the default
   locale everywhere (reports generate in English). Backend suite: **130 tests
   passing** on the translated code.
2. **Local graph layer instead of Zep Cloud** (`backend/app/services/local_graph/`).
   A duck-typed replacement for the `zep_cloud` client surface backed by
   SQLite: ontology introspection from the generated pydantic models, LLM
   extraction constrained to the ontology (8 parallel workers, one
   byte-identical retry so queued endpoints can cache), name+label dedupe,
   token-overlap search. Selected by `GRAPH_BACKEND=local` (the default);
   `GRAPH_BACKEND=zep` restores upstream behavior. No `ZEP_API_KEY` needed.
3. **Muse LLM Bridge** (`llm-bridge/`). An OpenAI-compatible endpoint whose
   completions are answered by the host agent itself through a durable queue:
   requests are classified by kind (ontology, extraction, persona, sim_config,
   agent_action, report…), exact repeats are served from an on-disk cache
   (SHA-256 of the canonical payload), and tool-call responses are supported.
   Point any OpenAI client at `http://localhost:8791/v1`, model `muse-spark`.
4. **MCP server** (`mcp-server/`). 27 tools wrapping the whole pipeline
   (projects, ontology, graph build, simulations, reports) plus the bridge
   queue (`bridge_pending`, `bridge_respond`, `bridge_stats`), so an MCP host
   can drive simulations end to end.

## Quickstart

```bash
# backend deps (once)
cd backend && uv sync && cd ..
cd llm-bridge && uv sync && cd ..
cd frontend && npm install && cd ..
cp .env.example .env   # set LLM_BASE_URL=http://localhost:8791/v1, LLM_MODEL_NAME=muse-spark
./scripts/start_local.sh
```

Then, as the host agent: poll `GET :8791/bridge/pending?full=1` and answer
each request via `POST :8791/bridge/respond` (or use the MCP tools).

`scripts/start_local.sh` unsets proxy variables for the local processes —
httpx fails to build proxy mounts from `NO_PROXY` values containing
bracketed IPv6 literals (`Invalid port: ':1]'`), and these services only
talk to localhost anyway. tiktoken encodings are cached under
`TIKTOKEN_CACHE_DIR` (default `~/.cache/tiktoken`); pre-download
`cl100k_base`/`o200k_base` once if the machine has no direct egress.

## Verified end-to-end run (2026-10-20)

Seed: a short announcement article ("DORZUN announces $100M seed round…"),
requirement: *How will defense investors, Pentagon stakeholders, defense
media, and skeptics react publicly over the two weeks after this
announcement?* Every LLM step was answered by the host agent through the
bridge:

- Ontology: 10 entity types, 8 edge types (generated + accepted live)
- Graph: **6 nodes / 5 edges** built by the local graph layer, DORZUN deduped
  across chunks
- Personas: 6 profiles; simulation config (time, event, per-agent activity)
- Simulation: **24 rounds × 2 platforms (Twitter + Reddit), 76 agent
  actions**, all decided via bridge tool calls
- Report: 5 sections, ~17K characters, each section grounded in 3 retrieval
  calls against the local graph — see
  [`docs/example-report-dorzun-seed.md`](docs/example-report-dorzun-seed.md)

## Known limitations

- Local graph search is lexical token overlap, not semantic/hybrid; facts
  are never invalidated or superseded, and entity resolution is exact
  normalized-name + label dedupe only.
- The Twitter recommendation path tries to download `Twitter/twhin-bert-base`
  from HuggingFace; without egress it retries and degrades to empty feeds
  (agents still act; Reddit feeds work fully).
- `interview_agents` (report tool) needs the OASIS environment running;
  stop the simulation only after report generation if you want interviews.
- Extraction quality is single-pass and bounded by the answering model.
