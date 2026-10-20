"""LLM-based entity/relationship extraction for the local graph backend.

Zep Cloud runs Graphiti server-side; locally we ask the configured
OpenAI-compatible LLM (``app.utils.llm_client.LLMClient``) for strict JSON
constrained to the ontology stored for the graph, then upsert the results
into SQLite with (graph, normalized name, label) deduplication.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Callable, Dict, List, Optional

from ...utils.logger import get_logger
from .store import LocalGraphStore

logger = get_logger("mirofish.local_graph.extraction")

_MAX_EPISODE_CHARS_FOR_LLM = 8000

_SYSTEM_PROMPT = """You are an information extraction engine. Extract entities and \
relationships from the provided text and return ONLY a JSON object with this shape:
{
  "entities": [
    {"name": "<entity name>", "type": "<one of the allowed entity types>",
     "summary": "<one short sentence describing the entity>",
     "attributes": {"<attribute name>": "<value as string>"}}
  ],
  "relationships": [
    {"source": "<source entity name>", "target": "<target entity name>",
     "type": "<one of the allowed edge types>",
     "fact": "<the sentence or statement expressing this relationship>",
     "attributes": {"<attribute name>": "<value as string>"}}
  ]
}
Rules:
- Use ONLY the allowed entity types and edge types listed in the ontology.
  If the ontology lists no types, use "Entity" for entities and "RELATED_TO" for relationships.
- Every relationship source/target must also appear in the entities list.
- Entity names must be the canonical name of the entity as written in the text.
- Do not invent facts that are not supported by the text.
- Return an empty list for entities/relationships when nothing matches."""


def build_extraction_prompt(text: str, ontology: Dict[str, Any]) -> List[Dict[str, str]]:
    entity_types = ontology.get("entity_types", [])
    edge_types = ontology.get("edge_types", [])
    ontology_lines = ["Allowed entity types:"]
    if entity_types:
        for entity in entity_types:
            attrs = ", ".join(
                attr.get("name", "") for attr in entity.get("attributes", [])
            )
            ontology_lines.append(
                f"- {entity.get('name')}: {entity.get('description', '')}"
                + (f" (attributes: {attrs})" if attrs else "")
            )
    else:
        ontology_lines.append("- Entity")
    ontology_lines.append("Allowed edge types:")
    if edge_types:
        for edge in edge_types:
            pairs = edge.get("source_targets", [])
            pair_text = "; ".join(
                f"{pair.get('source', 'Entity')} -> {pair.get('target', 'Entity')}"
                for pair in pairs
            )
            ontology_lines.append(
                f"- {edge.get('name')}: {edge.get('description', '')}"
                + (f" (allowed: {pair_text})" if pair_text else "")
            )
    else:
        ontology_lines.append("- RELATED_TO")
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "\n".join(ontology_lines)
                + "\n\nText to extract from:\n"
                + text[:_MAX_EPISODE_CHARS_FOR_LLM]
            ),
        },
    ]


class LocalGraphExtractor:
    """Extracts graph structure from episode text into the local store."""

    def __init__(
        self,
        store: LocalGraphStore,
        llm_client_factory: Optional[Callable[[], Any]] = None,
    ):
        self.store = store
        self._llm_client_factory = llm_client_factory
        self._llm_client: Any = None
        self._llm_lock = threading.Lock()

    def _llm(self) -> Any:
        # Extraction runs on a worker pool; guard the lazy client creation.
        if self._llm_client is None:
            with self._llm_lock:
                if self._llm_client is None:
                    if self._llm_client_factory is not None:
                        self._llm_client = self._llm_client_factory()
                    else:
                        from ...utils.llm_client import LLMClient

                        self._llm_client = LLMClient()
        return self._llm_client

    def extract_episode(self, graph_id: str, episode_uuid: str, text: str) -> None:
        """Run extraction for one episode and mark it processed.

        Raises on LLM/parse failure so batch items can be marked failed;
        callers handling fire-and-forget ``graph.add`` ingestion catch and
        still mark the episode processed (mirroring Zep, where ``graph.add``
        success is independent of downstream extraction quality).

        Retry policy: on failure the SAME payload is retried exactly once,
        with no prompt/parameter variation. The production LLM endpoint (the
        Muse LLM Bridge) caches identical calls, so a repeat is cheap and
        deterministic; ``LLMClient.chat_json`` regeneration is disabled via
        ``max_attempts=1`` for the same reason.
        """
        last_error: Optional[Exception] = None
        for _attempt in range(2):
            try:
                ontology = self.store.get_ontology(graph_id)
                messages = build_extraction_prompt(text, ontology)
                result = self._llm().chat_json(
                    messages,
                    temperature=0.1,
                    max_tokens=4096,
                    max_attempts=1,
                )
                self._apply_result(graph_id, episode_uuid, result, ontology)
                self.store.mark_episode_processed(episode_uuid)
                return
            except Exception as error:
                last_error = error
                logger.warning(
                    "Local extraction attempt failed for episode %s: %s",
                    episode_uuid,
                    error,
                )
        assert last_error is not None
        raise last_error

    def _apply_result(
        self,
        graph_id: str,
        episode_uuid: str,
        result: Dict[str, Any],
        ontology: Dict[str, Any],
    ) -> None:
        allowed_entity_types = {
            entity.get("name") for entity in ontology.get("entity_types", [])
        }
        allowed_edge_types = {
            edge.get("name"): edge for edge in ontology.get("edge_types", [])
        }

        node_uuid_by_key: Dict[str, str] = {}
        extracted_entities: List[Dict[str, Any]] = []
        for raw_entity in result.get("entities") or []:
            if not isinstance(raw_entity, dict):
                continue
            name = str(raw_entity.get("name") or "").strip()
            if not name:
                continue
            entity_type = str(raw_entity.get("type") or "Entity").strip() or "Entity"
            if allowed_entity_types and entity_type not in allowed_entity_types:
                entity_type = "Entity"
            attributes = raw_entity.get("attributes")
            if not isinstance(attributes, dict):
                attributes = {}
            node = self.store.upsert_node(
                graph_id,
                name,
                entity_type,
                summary=str(raw_entity.get("summary") or ""),
                attributes={str(k): v for k, v in attributes.items()},
            )
            node_uuid_by_key[_entity_key(name, entity_type)] = node.uuid_
            node_uuid_by_key.setdefault(_entity_key(name, ""), node.uuid_)
            extracted_entities.append(
                {"name": name, "type": entity_type, "uuid": node.uuid_}
            )

        type_by_name = {
            _entity_key(e["name"], ""): e["type"] for e in extracted_entities
        }

        for raw_rel in result.get("relationships") or []:
            if not isinstance(raw_rel, dict):
                continue
            source_name = str(raw_rel.get("source") or "").strip()
            target_name = str(raw_rel.get("target") or "").strip()
            if not source_name or not target_name:
                continue
            edge_type = str(raw_rel.get("type") or "RELATED_TO").strip() or "RELATED_TO"
            if allowed_edge_types and edge_type not in allowed_edge_types:
                continue
            source_uuid = self._resolve_node(
                graph_id, source_name, node_uuid_by_key, type_by_name
            )
            target_uuid = self._resolve_node(
                graph_id, target_name, node_uuid_by_key, type_by_name
            )
            if not source_uuid or not target_uuid or source_uuid == target_uuid:
                continue
            edge_def = allowed_edge_types.get(edge_type) or {}
            if not self._pair_allowed(
                edge_def,
                type_by_name.get(_entity_key(source_name, ""), "Entity"),
                type_by_name.get(_entity_key(target_name, ""), "Entity"),
            ):
                logger.debug(
                    "Dropping relationship %s -[%s]-> %s: pair not in ontology",
                    source_name,
                    edge_type,
                    target_name,
                )
                continue
            attributes = raw_rel.get("attributes")
            if not isinstance(attributes, dict):
                attributes = {}
            fact = str(raw_rel.get("fact") or "").strip()
            if not fact:
                fact = f"{source_name} {edge_type} {target_name}"
            self.store.upsert_edge(
                graph_id,
                edge_type,
                fact,
                source_uuid,
                target_uuid,
                fact_type=edge_type,
                attributes={str(k): v for k, v in attributes.items()},
                episode_uuid=episode_uuid,
            )

    def _resolve_node(
        self,
        graph_id: str,
        name: str,
        node_uuid_by_key: Dict[str, str],
        type_by_name: Dict[str, str],
    ) -> Optional[str]:
        """Find or create the node for a relationship endpoint."""
        key = _entity_key(name, "")
        if key in node_uuid_by_key:
            return node_uuid_by_key[key]
        existing = self.store.find_node(graph_id, name)
        if existing is not None:
            node_uuid_by_key[key] = existing.uuid_
            return existing.uuid_
        entity_type = type_by_name.get(key, "Entity")
        node = self.store.upsert_node(graph_id, name, entity_type)
        node_uuid_by_key[key] = node.uuid_
        return node.uuid_

    @staticmethod
    def _pair_allowed(edge_def: Dict[str, Any], source_type: str, target_type: str) -> bool:
        pairs = edge_def.get("source_targets") or []
        if not pairs:
            return True
        for pair in pairs:
            allowed_source = pair.get("source", "Entity")
            allowed_target = pair.get("target", "Entity")
            if allowed_source in ("Entity", source_type) and allowed_target in (
                "Entity",
                target_type,
            ):
                return True
        return False


def _entity_key(name: str, entity_type: str) -> str:
    normalized = " ".join(str(name).split()).casefold()
    return f"{normalized}::{entity_type}"


_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens used by the local search ranker."""
    return _TOKEN_PATTERN.findall(str(text).casefold())
