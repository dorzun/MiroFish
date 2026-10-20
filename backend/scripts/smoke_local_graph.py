#!/usr/bin/env python3
"""Smoke test for the self-hosted local graph backend (no Zep account).

Runs the real MiroFish pipeline pieces against a throwaway SQLite database
with a MOCK LLM (canned extraction JSON), so it needs no network and no API
keys:

  create graph -> set ontology (2 entity types, 1 edge type)
  -> batch-add 2 episodes (GraphBuilderService.add_text_batches)
  -> poll the batch to a terminal state (GraphBuilderService._wait_for_batch)
  -> read entities back (ZepEntityReader.filter_defined_entities)
  -> graph.search for a fact
  -> graph.add one simulation-style episode and confirm it gets processed

Usage (from backend/):
    uv run python scripts/smoke_local_graph.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

# Configure BEFORE importing app modules: Config reads env at import time.
_TMPDIR = tempfile.mkdtemp(prefix="mirofish-local-graph-smoke-")
os.environ["GRAPH_BACKEND"] = "local"
os.environ["MIROFISH_LOCAL_GRAPH_DB"] = os.path.join(_TMPDIR, "smoke.db")
os.environ.setdefault("LLM_API_KEY", "smoke-test-key")  # never used; LLM is mocked
os.environ.pop("ZEP_API_KEY", None)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import Config  # noqa: E402
from app.services.graph_builder import GraphBuilderService  # noqa: E402
from app.services.local_graph import get_local_graph_client  # noqa: E402
from app.services.zep_entity_reader import ZepEntityReader  # noqa: E402

EPISODE_1 = (
    "Jane Smith is the CEO of Acme Robotics. Jane Smith works at Acme Robotics, "
    "where she leads the counter-drone program."
)
EPISODE_2 = (
    "Acme Robotics signed a supply contract with Northwind Defense. "
    "Jane Smith announced the contract on Tuesday."
)


class MockLLM:
    """Returns canned extraction JSON keyed by markers in the episode text."""

    def chat_json(self, messages, **_kwargs):
        text = messages[-1]["content"]
        entities = [
            {
                "name": "Jane Smith",
                "type": "Person",
                "summary": "CEO of Acme Robotics.",
                "attributes": {"role": "CEO"},
            },
            {
                "name": "Acme Robotics",
                "type": "Organization",
                "summary": "A robotics company building counter-drone systems.",
                "attributes": {},
            },
        ]
        relationships = [
            {
                "source": "Jane Smith",
                "target": "Acme Robotics",
                "type": "WORKS_AT",
                "fact": "Jane Smith works at Acme Robotics.",
                "attributes": {},
            }
        ]
        if "Northwind Defense" in text:
            entities.append(
                {
                    "name": "Northwind Defense",
                    "type": "Organization",
                    "summary": "A defense contractor.",
                    "attributes": {},
                }
            )
            relationships.append(
                {
                    "source": "Acme Robotics",
                    "target": "Northwind Defense",
                    "type": "WORKS_AT",
                    "fact": "Acme Robotics signed a supply contract with Northwind Defense.",
                    "attributes": {},
                }
            )
        return {"entities": entities, "relationships": relationships}


ONTOLOGY = {
    "entity_types": [
        {
            "name": "Person",
            "description": "A human individual.",
            "attributes": [{"name": "role", "description": "The person's job role."}],
        },
        {
            "name": "Organization",
            "description": "A company or institution.",
            "attributes": [],
        },
    ],
    "edge_types": [
        {
            "name": "WORKS_AT",
            "description": "Employment or contractual relationship.",
            "attributes": [],
            "source_targets": [
                {"source": "Person", "target": "Organization"},
                {"source": "Organization", "target": "Organization"},
            ],
        }
    ],
}


def main() -> int:
    assert Config.GRAPH_BACKEND == "local", Config.GRAPH_BACKEND
    # Simulate a deployment with no Zep account: a repo-root .env may carry a
    # placeholder ZEP_API_KEY (loaded by app.config), so clear it in-process.
    # Local mode must never consult it.
    Config.ZEP_API_KEY = None
    assert not Config.ZEP_API_KEY, "smoke test must run without ZEP_API_KEY"
    config_errors = Config.validate()
    assert not any("ZEP" in error for error in config_errors), config_errors
    print(f"[ok] config: GRAPH_BACKEND=local, DB={Config.LOCAL_GRAPH_DB_PATH}")

    # Inject the mock LLM into the singleton local client used by all services.
    client = get_local_graph_client()
    client.extractor._llm_client = MockLLM()
    print("[ok] local graph client created, mock LLM injected")

    builder = GraphBuilderService()  # must not require ZEP_API_KEY in local mode
    graph_id = builder.create_graph("Smoke Test Graph")
    print(f"[ok] graph created: {graph_id}")

    builder.set_ontology(graph_id, ONTOLOGY)
    stored = client.store.get_ontology(graph_id)
    print(
        "[ok] ontology stored: "
        f"entities={[e['name'] for e in stored['entity_types']]}, "
        f"edges={[e['name'] for e in stored['edge_types']]}"
    )

    submission = builder.add_text_batches(graph_id, [EPISODE_1, EPISODE_2])
    print(
        f"[..] batch submitted: batch_id={submission.batch_id}, "
        f"items={submission.item_count}, episodes={len(submission.episode_uuids)}"
    )
    episode_uuids = builder._wait_for_batch(submission, timeout=120)
    print(f"[ok] batch terminal: episodes confirmed={episode_uuids}")

    for episode_uuid in episode_uuids:
        episode = client.graph.episode.get(uuid_=episode_uuid)
        assert episode.processed, f"episode {episode_uuid} not processed"
    print("[ok] all episodes report processed=True")

    reader = ZepEntityReader()
    filtered = reader.filter_defined_entities(graph_id, enrich_with_edges=True)
    print(
        f"[ok] entity reader: total_nodes={filtered.total_count}, "
        f"filtered={filtered.filtered_count}, types={sorted(filtered.entity_types)}"
    )
    for entity in filtered.entities:
        facts = [edge["fact"] for edge in entity.related_edges]
        print(f"     - {entity.name} [{entity.get_entity_type()}]: {entity.summary}")
        for fact in facts:
            print(f"         fact: {fact}")

    results = client.graph.search(
        graph_id=graph_id, query="Jane Smith Acme Robotics", limit=10, scope="edges"
    )
    print(f"[ok] search returned {len(results.edges)} edge(s):")
    for edge in results.edges:
        print(f"     * [{edge.name}] {edge.fact}")

    # Fire-and-forget ingestion path used by the simulation memory updater.
    episode = client.graph.add(
        graph_id=graph_id,
        type="text",
        data=EPISODE_2,
        source_description="MiroFish simulation activity batch",
        metadata={"source": "mirofish_simulation"},
    )
    deadline = time.time() + 60
    while time.time() < deadline:
        if client.graph.episode.get(uuid_=episode.uuid_).processed:
            break
        time.sleep(0.2)
    else:
        raise AssertionError("graph.add episode never became processed")
    print(f"[ok] graph.add episode processed: {episode.uuid_}")

    info = builder._get_graph_info(graph_id)
    print(
        f"[ok] graph info: nodes={info.node_count}, edges={info.edge_count}, "
        f"entity_types={sorted(info.entity_types)}"
    )

    names = {entity.name for entity in filtered.entities}
    assert {"Jane Smith", "Acme Robotics", "Northwind Defense"} <= names, names
    assert results.edges, "search found no edges"
    print("\nSMOKE PASSED: local graph pipeline works end to end without Zep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
