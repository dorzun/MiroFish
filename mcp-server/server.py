#!/usr/bin/env python3
"""MiroFish MCP server.

Wraps the MiroFish Flask REST API (backend on http://localhost:5001 by default)
as MCP tools, so an MCP client (Muse, Claude, Cursor, ...) can drive the
full prediction pipeline:

  seed text -> ontology -> knowledge graph (Zep) -> agent profiles ->
  OASIS Twitter/Reddit simulation -> prediction report -> agent interviews

Config (environment):
  MIROFISH_API_URL   Base URL of the MiroFish backend (default http://localhost:5001)
  MIROFISH_LOCALE    Accept-Language sent to the backend (default "en")

The backend itself needs LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_NAME and
ZEP_API_KEY in the project-root .env (see .env.example).
"""

import os
import time

import httpx
from mcp.server.fastmcp import FastMCP

API_URL = os.environ.get("MIROFISH_API_URL", "http://localhost:5001").rstrip("/")
BRIDGE_URL = os.environ.get("MIROFISH_BRIDGE_URL", "http://localhost:8791").rstrip("/")
LOCALE = os.environ.get("MIROFISH_LOCALE", "en")

mcp = FastMCP("mirofish")


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=API_URL,
        headers={"Accept-Language": LOCALE},
        timeout=httpx.Timeout(120.0, connect=10.0),
    )


def _unwrap(resp: httpx.Response):
    """Return the `data` field of a MiroFish envelope, or raise a readable error."""
    try:
        payload = resp.json()
    except ValueError:
        resp.raise_for_status()
        return {"raw": resp.text}
    if isinstance(payload, dict) and payload.get("success") is False:
        raise RuntimeError(f"MiroFish API error ({resp.status_code}): {payload.get('error')}")
    if resp.status_code >= 400:
        raise RuntimeError(f"MiroFish API error ({resp.status_code}): {payload}")
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def _get(path: str, **params):
    with _client() as c:
        return _unwrap(c.get(path, params={k: v for k, v in params.items() if v is not None}))


def _post(path: str, json_body=None):
    with _client() as c:
        return _unwrap(c.post(path, json=json_body or {}))


# ---------------------------------------------------------------- health / meta

@mcp.tool()
def mirofish_health() -> dict:
    """Check that the MiroFish backend is up. Returns its health payload."""
    with _client() as c:
        r = c.get("/health")
        r.raise_for_status()
        return r.json()


@mcp.tool()
def list_projects() -> dict:
    """List all MiroFish projects (seed uploads + ontology runs)."""
    return _get("/api/graph/project/list")


@mcp.tool()
def get_project(project_id: str) -> dict:
    """Get one project by id, including its graph_id once the graph is built."""
    return _get(f"/api/graph/project/{project_id}")


@mcp.tool()
def list_tasks() -> dict:
    """List backend async tasks (graph builds, report generations, ...)."""
    return _get("/api/graph/tasks")


@mcp.tool()
def get_task(task_id: str) -> dict:
    """Get the status of one async backend task."""
    return _get(f"/api/graph/task/{task_id}")


# ---------------------------------------------------------------- pipeline steps

@mcp.tool()
def create_project(seed_text: str, simulation_requirement: str, project_name: str = "Prediction") -> dict:
    """Step 1: create a project from raw seed text and generate its ontology.

    seed_text: the seed material (news, policy draft, report, briefing) as plain text.
    simulation_requirement: what to predict, in natural language, e.g.
        "How will defense investors and the public react if DORZUN announces a
         $100M seed round next week?"
    Returns project_id and the generated ontology (entity/edge types).
    """
    with _client() as c:
        files = {"files": ("seed.md", seed_text.encode("utf-8"), "text/markdown")}
        data = {
            "simulation_requirement": simulation_requirement,
            "project_name": project_name,
        }
        return _unwrap(c.post("/api/graph/ontology/generate", files=files, data=data))


@mcp.tool()
def create_project_from_file(file_path: str, simulation_requirement: str, project_name: str = "Prediction") -> dict:
    """Step 1 (file variant): create a project from a local PDF/MD/TXT seed file."""
    with open(file_path, "rb") as f:
        content = f.read()
    filename = os.path.basename(file_path)
    with _client() as c:
        files = {"files": (filename, content)}
        data = {
            "simulation_requirement": simulation_requirement,
            "project_name": project_name,
        }
        return _unwrap(c.post("/api/graph/ontology/generate", files=files, data=data))


@mcp.tool()
def build_graph(project_id: str) -> dict:
    """Step 2: build the Zep knowledge graph for a project (async; poll get_task)."""
    return _post("/api/graph/build", {"project_id": project_id})


@mcp.tool()
def get_graph_data(graph_id: str) -> dict:
    """Fetch nodes/edges of a built knowledge graph."""
    return _get(f"/api/graph/data/{graph_id}")


@mcp.tool()
def create_simulation(project_id: str, enable_twitter: bool = True, enable_reddit: bool = True) -> dict:
    """Step 3: create a simulation for a project whose graph is built."""
    return _post(
        "/api/simulation/create",
        {"project_id": project_id, "enable_twitter": enable_twitter, "enable_reddit": enable_reddit},
    )


@mcp.tool()
def prepare_simulation(simulation_id: str, force_regenerate: bool = False,
                       parallel_profile_count: int = 20) -> dict:
    """Step 4: generate agent profiles + simulation config (async; poll prepare_status).
    parallel_profile_count controls how many persona generations run concurrently
    (upstream default is 5; 20 is a safe speed-up with the bridge, since queued
    requests are answered in batches)."""
    return _post(
        "/api/simulation/prepare",
        {"simulation_id": simulation_id, "force_regenerate": force_regenerate,
         "parallel_profile_count": parallel_profile_count},
    )


@mcp.tool()
def prepare_status(simulation_id: str) -> dict:
    """Poll the prepare step for a simulation."""
    return _post("/api/simulation/prepare/status", {"simulation_id": simulation_id})


@mcp.tool()
def start_simulation(simulation_id: str, max_rounds: int = 10, platform: str = "parallel") -> dict:
    """Step 5: run the simulation. platform: 'twitter', 'reddit' or 'parallel'.
    Keep max_rounds small (<=10) for a first test: every agent action is an LLM call."""
    return _post(
        "/api/simulation/start",
        {"simulation_id": simulation_id, "max_rounds": max_rounds, "platform": platform},
    )


@mcp.tool()
def simulation_run_status(simulation_id: str) -> dict:
    """Current run status of a simulation (running/completed/stopped, round, counts)."""
    return _get(f"/api/simulation/{simulation_id}/run-status")


@mcp.tool()
def simulation_posts(simulation_id: str, limit: int = 50) -> dict:
    """Posts produced inside the simulation so far."""
    return _get(f"/api/simulation/{simulation_id}/posts", limit=limit)


@mcp.tool()
def simulation_timeline(simulation_id: str) -> dict:
    """Round-by-round timeline of a simulation."""
    return _get(f"/api/simulation/{simulation_id}/timeline")


@mcp.tool()
def stop_simulation(simulation_id: str) -> dict:
    """Stop a running simulation."""
    return _post("/api/simulation/stop", {"simulation_id": simulation_id})


@mcp.tool()
def interview_agent(simulation_id: str, agent_id: int, prompt: str, platform: str | None = None) -> dict:
    """Interview one simulated agent (simulation must be in its post-run waiting state)."""
    body = {"simulation_id": simulation_id, "agent_id": agent_id, "prompt": prompt}
    if platform:
        body["platform"] = platform
    return _post("/api/simulation/interview", body)


@mcp.tool()
def generate_report(simulation_id: str, force_regenerate: bool = False) -> dict:
    """Step 6: generate the prediction report for a finished simulation (async)."""
    return _post("/api/report/generate", {"simulation_id": simulation_id, "force_regenerate": force_regenerate})


@mcp.tool()
def report_status(simulation_id: str) -> dict:
    """Poll report generation progress for a simulation."""
    return _post("/api/report/generate/status", {"simulation_id": simulation_id})


@mcp.tool()
def get_report_by_simulation(simulation_id: str) -> dict:
    """Fetch the finished prediction report for a simulation."""
    return _get(f"/api/report/by-simulation/{simulation_id}")


@mcp.tool()
def chat_with_report(report_id: str, message: str) -> dict:
    """Ask the ReportAgent a follow-up question about a finished report."""
    return _post("/api/report/chat", {"report_id": report_id, "message": message})


# ---------------------------------------------------------------- one-shot run

def _poll(fn, done_states: set[str], timeout_s: int, interval_s: int = 5, label: str = "step"):
    """Poll fn() until its status field lands in done_states or looks failed."""
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        last = fn()
        status = str((last or {}).get("status", "")).lower()
        if status in done_states:
            return last
        if any(bad in status for bad in ("fail", "error", "cancel")):
            raise RuntimeError(f"{label} failed: {last}")
        time.sleep(interval_s)
    raise TimeoutError(f"{label} did not finish within {timeout_s}s; last state: {last}")


@mcp.tool()
def run_full_prediction(
    seed_text: str,
    simulation_requirement: str,
    project_name: str = "Prediction",
    max_rounds: int = 5,
    platform: str = "parallel",
    timeout_minutes: int = 60,
) -> dict:
    """Run the WHOLE pipeline end to end and return the prediction report.

    Creates the project, builds the graph, prepares agents, runs the
    simulation, generates the report, polling each async step. This can take
    a long time and burn many LLM calls on big seeds - start with a short
    seed and max_rounds <= 5.

    Returns project/simulation/report ids plus the report content.
    """
    budget = timeout_minutes * 60
    step_budget = max(300, budget // 4)

    project = create_project(seed_text, simulation_requirement, project_name)
    project_id = project["project_id"]

    build = build_graph(project_id)
    task_id = (build or {}).get("task_id")
    if task_id:
        _poll(lambda: get_task(task_id), {"completed", "done", "success", "finished"},
              step_budget, label="graph build")

    project_after = get_project(project_id)
    if not (project_after or {}).get("graph_id"):
        raise RuntimeError(f"Graph build finished but project has no graph_id: {project_after}")

    sim = create_simulation(project_id)
    simulation_id = sim["simulation_id"]

    prep = prepare_simulation(simulation_id)
    if (prep or {}).get("status") != "ready":
        _poll(lambda: prepare_status(simulation_id), {"ready", "completed", "done"},
              step_budget, label="simulation prepare")

    start_simulation(simulation_id, max_rounds=max_rounds, platform=platform)
    _poll(lambda: simulation_run_status(simulation_id),
          {"completed", "stopped", "finished", "done"},
          step_budget, label="simulation run")

    generate_report(simulation_id)
    _poll(lambda: report_status(simulation_id), {"completed", "done", "success", "finished"},
          step_budget, label="report generation")

    report = get_report_by_simulation(simulation_id)
    return {
        "project_id": project_id,
        "simulation_id": simulation_id,
        "report": report,
    }


# ------------------------------------------------- Muse LLM bridge (agent = model)
#
# When the MiroFish backend is pointed at the llm-bridge (LLM_BASE_URL=
# http://localhost:8791/v1), every LLM call in the pipeline lands in the
# bridge queue. The MCP host agent (you) IS the model: fetch the queued
# requests with full message payloads, generate the completions yourself,
# and submit them back. Answer in the exact format each request demands
# (strict JSON when response_format asks for it, an agent's social action
# JSON during simulation rounds, report prose during report generation).

def _bridge_client() -> httpx.Client:
    return httpx.Client(base_url=BRIDGE_URL, timeout=httpx.Timeout(30.0, connect=5.0))


@mcp.tool()
def bridge_health() -> dict:
    """Health and queue stats of the Muse LLM bridge (the agent-as-model endpoint)."""
    with _bridge_client() as c:
        r = c.get("/health")
        r.raise_for_status()
        return r.json()


@mcp.tool()
def pending_llm_requests(limit: int = 200) -> dict:
    """Fetch LLM requests queued by MiroFish at the bridge, with FULL message
    payloads. These are calls the pipeline is blocked on right now: ontology
    generation, graph extraction, persona writing, per-agent simulation
    actions, report drafting. Generate each completion yourself, then submit
    all answers with answer_llm_requests."""
    with _bridge_client() as c:
        r = c.get("/bridge/pending", params={"full": 1, "limit": limit})
        r.raise_for_status()
        return r.json()


@mcp.tool()
def answer_llm_request(request_id: str, content: str) -> dict:
    """Submit your generated completion for one queued bridge request."""
    with _bridge_client() as c:
        r = c.post("/bridge/respond", json={"request_id": request_id, "content": content})
        r.raise_for_status()
        return r.json()


@mcp.tool()
def answer_llm_requests(answers: list[dict]) -> dict:
    """Submit completions for many queued bridge requests at once.
    answers: [{"request_id": "...", "content": "..."}]"""
    with _bridge_client() as c:
        r = c.post("/bridge/respond_batch", json={"answers": answers})
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    mcp.run()
