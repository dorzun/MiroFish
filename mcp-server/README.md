# MiroFish MCP Server

An MCP (Model Context Protocol) server that wraps the MiroFish backend REST
API, so any MCP client can drive the full prediction pipeline as tools:

`seed text -> ontology -> knowledge graph -> agent profiles -> OASIS
Twitter/Reddit simulation -> prediction report -> agent interviews`

## Prerequisites

1. The MiroFish backend running (from the repo root: `npm run backend`,
   frontend on :3000, backend API on :5001), with a project-root `.env`
   containing `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL_NAME` and
   `ZEP_API_KEY`.
2. Python >= 3.11 and [uv](https://docs.astral.sh/uv/).

## Run

```sh
cd mcp-server
uv run python server.py            # stdio transport
```

Environment:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MIROFISH_API_URL` | `http://localhost:5001` | MiroFish backend base URL |
| `MIROFISH_LOCALE` | `en` | `Accept-Language` sent to the backend |

## MCP client config (example)

```json
{
  "mcpServers": {
    "mirofish": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/MiroFish/mcp-server", "python", "server.py"],
      "env": { "MIROFISH_API_URL": "http://localhost:5001" }
    }
  }
}
```

## Tools

| Tool | Purpose |
| --- | --- |
| `mirofish_health` | Backend health check |
| `list_projects` / `get_project` / `list_tasks` / `get_task` | Inspect projects and async tasks |
| `create_project` / `create_project_from_file` | Step 1: seed -> project + ontology |
| `build_graph` / `get_graph_data` | Step 2: Zep knowledge graph |
| `create_simulation` | Step 3: simulation shell |
| `prepare_simulation` / `prepare_status` | Step 4: agent profiles + config |
| `start_simulation` / `simulation_run_status` / `stop_simulation` | Step 5: run control |
| `simulation_posts` / `simulation_timeline` | What the agents are doing |
| `interview_agent` | Talk to one simulated agent |
| `generate_report` / `report_status` / `get_report_by_simulation` / `chat_with_report` | Step 6: prediction report |
| `run_full_prediction` | One call: whole pipeline, returns the report |

Cost note: every agent action in a simulation is an LLM call, and report
generation adds more. Start with a short seed and `max_rounds` <= 5.
