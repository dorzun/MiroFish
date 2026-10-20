# Muse LLM Bridge

An OpenAI-compatible endpoint whose "model" is the MCP host agent itself.

Point MiroFish at the bridge and every LLM call in the pipeline (ontology
generation, graph extraction, agent personas, simulation actions, report
writing) is queued here and answered by the agent driving the MCP tools.
No API key, no external model, no local weights: the host agent *is* the
engine.

## Run

```sh
cd llm-bridge
uv run python bridge.py        # listens on :8791
```

## Wire MiroFish to it (project-root `.env`)

```
GRAPH_BACKEND=local
LLM_API_KEY=<redacted>
LLM_BASE_URL=http://localhost:8791/v1
LLM_MODEL_NAME=muse-spark
```

## Flow

1. MiroFish POSTs `/v1/chat/completions` and waits (up to
   `BRIDGE_TIMEOUT_SECONDS`, default 1800).
2. The host agent fetches the queue via the MiroFish MCP tools
   `pending_llm_requests` (full message payloads included), generates the
   answers, and submits them with `answer_llm_requests`.
3. The bridge returns each answer as a standard chat completion.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Bridge health + queue stats |
| `GET /v1/models` | Advertises the bridge model |
| `POST /v1/chat/completions` | OpenAI-compatible, non-streaming |
| `GET /bridge/pending?full=1` | Queued requests with full messages |
| `GET /bridge/request/<id>` | One queued request |
| `POST /bridge/respond` | Answer one: `{request_id, content, tool_calls?}` |
| `POST /bridge/respond_batch` | Answer many: `{answers: [{request_id, content}]}` |
| `GET /bridge/stats` | Counters |

Scale note: this mode trades speed for quality and control. Each simulation
round can queue dozens of agent calls; the host agent answers them in
batches. It is ideal for focused, high-value scenarios. For mass runs with
thousands of agents, swap `LLM_BASE_URL` back to a regular model API.

## Speed pack (quality-preserving)

1. **Exact-repeat response cache** (`response_cache.json`, disable with
   `BRIDGE_CACHE_ENABLED=0`): client retries and duplicate prompts are
   answered instantly from disk instead of queueing again. Cache hits are
   counted in `/bridge/stats`.
2. **Kind classification**: every queued request is tagged (`ontology`,
   `graph_extraction`, `persona`, `sim_config`, `sim_action`, `interview`,
   `report`, `other`) in `/bridge/pending`, so the coordinator can dispatch
   by category.
3. **Fan-out answering protocol**: the coordinator fetches the queue once,
   splits it into disjoint chunks by `request_id`, and hands each chunk to a
   parallel answerer subagent (same model, same quality). Each answerer
   generates its completions and POSTs them straight to
   `/bridge/respond_batch`. Wall-clock time per round drops roughly by the
   number of answerers.
4. **Answer discipline**: for `sim_action` and other machine-parsed calls,
   answer with the minimal valid JSON only - no prose, no markdown fences.
   Shorter answers are faster answers, and the parsers prefer them.
5. **Pipeline concurrency knobs** (set on the MiroFish side):
   `prepare_simulation(parallel_profile_count=20)`,
   `LOCAL_GRAPH_EXTRACTION_WORKERS=8` for graph extraction, platforms run in
   parallel natively, and `enable_graph_memory_update` stays off during runs
   unless the report needs live graph memory.
