"""Duck-typed local replacement for the zep_cloud ``Zep`` client.

Exposes exactly the attribute paths MiroFish calls:

- ``client.graph.create / get / delete / set_ontology / add / search``
- ``client.graph.node.get / get_edges / get_by_graph_id`` (+ ``with_raw_response``)
- ``client.graph.edge.get_by_graph_id`` (+ ``with_raw_response``)
- ``client.graph.episode.get / get_by_graph_id``
- ``client.batch.create / add / process / get / list / list_items``

Ingestion is asynchronous like Zep's: ``graph.add`` returns an episode
immediately and extraction runs on a background thread; ``batch.process``
transitions the batch through processing -> succeeded/partial/failed while
the GraphBuilder polls ``batch.get`` and ``batch.list_items``.
"""

from __future__ import annotations

import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from ...config import Config
from ...utils.logger import get_logger
from .extraction import LocalGraphExtractor, tokenize
from .models import (
    BatchItemListPage,
    BatchListPage,
    BatchSummary,
    EpisodeListPage,
    LocalEpisode,
    RawGraphPage,
    SearchResults,
)
from .store import LocalGraphStore

logger = get_logger("mirofish.local_graph.client")

try:  # Consumers catch zep_cloud's NotFoundError, so raise the real type.
    from zep_cloud import NotFoundError as _ZepNotFoundError
except Exception:  # pragma: no cover - zep_cloud is a hard dependency today
    _ZepNotFoundError = None


def _not_found(message: str) -> Exception:
    if _ZepNotFoundError is not None:
        try:
            return _ZepNotFoundError(status_code=404, body={"message": message})
        except TypeError:
            try:
                return _ZepNotFoundError()
            except Exception:
                pass
    error = LookupError(message)
    return error


def _ontology_from_models(entities: Any, edges: Any) -> Dict[str, Any]:
    """Introspect zep_cloud ontology model classes into a plain registry dict.

    ``graph_builder.set_ontology`` builds pydantic classes dynamically
    (EntityModel/EdgeModel subclasses with Field descriptions) plus
    EntityEdgeSourceTarget pairs; we capture names, descriptions, attribute
    names/descriptions and allowed source/target pairs from them.
    """
    entity_types: List[Dict[str, Any]] = []
    for name, cls in (entities or {}).items():
        attributes = []
        for field_name, field in (getattr(cls, "model_fields", None) or {}).items():
            attributes.append(
                {
                    "name": field_name,
                    "description": getattr(field, "description", "") or "",
                }
            )
        entity_types.append(
            {
                "name": name,
                "description": (getattr(cls, "__doc__", "") or "").strip(),
                "attributes": attributes,
            }
        )

    edge_types: List[Dict[str, Any]] = []
    for name, value in (edges or {}).items():
        cls = value
        source_targets: List[Dict[str, str]] = []
        if isinstance(value, (tuple, list)):
            cls = value[0] if value else None
            raw_pairs = value[1] if len(value) > 1 else []
            for pair in raw_pairs or []:
                source_targets.append(
                    {
                        "source": getattr(pair, "source", "Entity") or "Entity",
                        "target": getattr(pair, "target", "Entity") or "Entity",
                    }
                )
        attributes = []
        if cls is not None:
            for field_name, field in (getattr(cls, "model_fields", None) or {}).items():
                attributes.append(
                    {
                        "name": field_name,
                        "description": getattr(field, "description", "") or "",
                    }
                )
        edge_types.append(
            {
                "name": name,
                "description": (getattr(cls, "__doc__", "") or "").strip()
                if cls is not None
                else "",
                "attributes": attributes,
                "source_targets": source_targets,
            }
        )
    return {"entity_types": entity_types, "edge_types": edge_types}


def _page_slice(items: List[Any], limit: Any, cursor: Any) -> RawGraphPage:
    """Emulate Zep's cursor paging: opaque cursor = next offset as a string."""
    try:
        offset = int(cursor) if cursor is not None else 0
    except (TypeError, ValueError):
        offset = 0
    try:
        page_size = int(limit) if limit else 100
    except (TypeError, ValueError):
        page_size = 100
    page = items[offset : offset + page_size]
    headers: Dict[str, str] = {}
    if offset + page_size < len(items):
        headers["zep-next-cursor"] = str(offset + page_size)
    return RawGraphPage(data=page, headers=headers)


class _RawResponseApi:
    """``with_raw_response`` wrapper returning data + headers objects."""

    def __init__(self, fetch):
        self._fetch = fetch

    def get_by_graph_id(self, graph_id: str, **kwargs: Any) -> RawGraphPage:
        return self._fetch(graph_id, **kwargs)


class _NodeApi:
    def __init__(self, client: "LocalGraphClient"):
        self._client = client
        self.with_raw_response = _RawResponseApi(self._get_page)

    def _get_page(self, graph_id: str, **kwargs: Any) -> RawGraphPage:
        store = self._client.store
        if store.get_graph(graph_id) is None:
            raise _not_found(f"graph {graph_id} not found")
        return _page_slice(
            store.list_nodes(graph_id), kwargs.get("limit"), kwargs.get("cursor")
        )

    def get_by_graph_id(self, graph_id: str, **kwargs: Any) -> RawGraphPage:
        return self._get_page(graph_id, **kwargs)

    def get(self, uuid_: Optional[str] = None, **kwargs: Any):
        node_uuid = uuid_ or kwargs.get("uuid")
        node = self._client.store.get_node(node_uuid) if node_uuid else None
        if node is None:
            raise _not_found(f"node {node_uuid} not found")
        return node

    def get_edges(self, node_uuid: Optional[str] = None, **kwargs: Any):
        resolved = node_uuid or kwargs.get("node_uuid")
        return self._client.store.edges_for_node(resolved)


class _EdgeApi:
    def __init__(self, client: "LocalGraphClient"):
        self._client = client
        self.with_raw_response = _RawResponseApi(self._get_page)

    def _get_page(self, graph_id: str, **kwargs: Any) -> RawGraphPage:
        store = self._client.store
        if store.get_graph(graph_id) is None:
            raise _not_found(f"graph {graph_id} not found")
        return _page_slice(
            store.list_edges(graph_id), kwargs.get("limit"), kwargs.get("cursor")
        )

    def get_by_graph_id(self, graph_id: str, **kwargs: Any) -> RawGraphPage:
        return self._get_page(graph_id, **kwargs)


class _EpisodeApi:
    def __init__(self, client: "LocalGraphClient"):
        self._client = client

    def get(self, uuid_: Optional[str] = None, **kwargs: Any) -> LocalEpisode:
        episode_uuid = uuid_ or kwargs.get("uuid")
        episode = (
            self._client.store.get_episode(episode_uuid) if episode_uuid else None
        )
        if episode is None:
            raise _not_found(f"episode {episode_uuid} not found")
        return episode

    def get_by_graph_id(self, graph_id: str = "", **kwargs: Any) -> EpisodeListPage:
        lastn = kwargs.get("lastn", 50)
        return EpisodeListPage(
            episodes=self._client.store.list_episodes(graph_id, lastn=lastn)
        )


class _GraphApi:
    def __init__(self, client: "LocalGraphClient"):
        self._client = client
        self.node = _NodeApi(client)
        self.edge = _EdgeApi(client)
        self.episode = _EpisodeApi(client)

    # ------------------------------------------------------------------ CRUD
    def create(self, graph_id: Optional[str] = None, **kwargs: Any):
        resolved_id = graph_id or kwargs.get("graph_id")
        if not resolved_id:
            raise ValueError("graph_id is required")
        return self._client.store.create_graph(
            resolved_id,
            kwargs.get("name", "") or "",
            kwargs.get("description", "") or "",
        )

    def get(self, graph_id: Optional[str] = None, **kwargs: Any):
        resolved_id = graph_id or kwargs.get("graph_id")
        graph = self._client.store.get_graph(resolved_id) if resolved_id else None
        if graph is None:
            raise _not_found(f"graph {resolved_id} not found")
        return graph

    def delete(self, graph_id: Optional[str] = None, **kwargs: Any) -> None:
        resolved_id = graph_id or kwargs.get("graph_id")
        if resolved_id:
            self._client.store.delete_graph(resolved_id)

    def set_ontology(self, **kwargs: Any) -> None:
        graph_ids = kwargs.get("graph_ids") or []
        ontology = _ontology_from_models(kwargs.get("entities"), kwargs.get("edges"))
        for graph_id in graph_ids:
            if self._client.store.get_graph(graph_id) is None:
                raise _not_found(f"graph {graph_id} not found")
            self._client.store.set_ontology(graph_id, ontology)

    # -------------------------------------------------------------- ingestion
    def add(self, **kwargs: Any) -> LocalEpisode:
        graph_id = kwargs.get("graph_id")
        if not graph_id:
            raise ValueError("graph_id is required")
        store = self._client.store
        if store.get_graph(graph_id) is None:
            raise _not_found(f"graph {graph_id} not found")
        created_at = kwargs.get("created_at")
        episode = store.add_episode(
            graph_id,
            str(kwargs.get("data") or ""),
            type=str(kwargs.get("type") or "text"),
            source_description=str(kwargs.get("source_description") or ""),
            metadata=kwargs.get("metadata") or {},
            created_at=str(created_at) if created_at else None,
        )
        self._client.spawn_extraction(graph_id, episode.uuid_)
        return episode

    # ----------------------------------------------------------------- search
    def search(self, **kwargs: Any) -> SearchResults:
        graph_id = kwargs.get("graph_id")
        query = str(kwargs.get("query") or "")
        try:
            limit = int(kwargs.get("limit") or 10)
        except (TypeError, ValueError):
            limit = 10
        scope = str(kwargs.get("scope") or "edges").lower()
        store = self._client.store
        if not graph_id or store.get_graph(graph_id) is None:
            raise _not_found(f"graph {graph_id} not found")

        query_tokens = Counter(tokenize(query))
        if not query_tokens:
            return SearchResults()

        node_by_uuid = {n.uuid_: n for n in store.list_nodes(graph_id)}

        def token_overlap(text: str) -> int:
            candidate = Counter(tokenize(text))
            return sum((query_tokens & candidate).values())

        edges: List[Any] = []
        nodes: List[Any] = []
        episodes: List[Any] = []

        if scope in ("edges", "auto", ""):
            scored_edges = []
            for edge in store.list_edges(graph_id):
                score = 3 * token_overlap(edge.fact) + 2 * token_overlap(edge.name)
                source = node_by_uuid.get(edge.source_node_uuid)
                target = node_by_uuid.get(edge.target_node_uuid)
                if source:
                    score += 2 * token_overlap(source.name)
                if target:
                    score += 2 * token_overlap(target.name)
                if score > 0:
                    scored_edges.append((score, edge))
            scored_edges.sort(key=lambda pair: (-pair[0], pair[1].created_at or ""))
            edges = [edge for _score, edge in scored_edges[:limit]]

        if scope in ("nodes", "auto", ""):
            scored_nodes = []
            for node in node_by_uuid.values():
                score = (
                    3 * token_overlap(node.name)
                    + 2 * token_overlap(node.summary)
                    + token_overlap(" ".join(node.labels))
                )
                if score > 0:
                    scored_nodes.append((score, node))
            scored_nodes.sort(key=lambda pair: (-pair[0], pair[1].created_at or ""))
            nodes = [node for _score, node in scored_nodes[:limit]]

        if scope in ("episodes", "auto", ""):
            scored_episodes = []
            for episode in store.list_episodes(graph_id, lastn=1000):
                score = token_overlap(episode.content)
                if score > 0:
                    scored_episodes.append((score, episode))
            scored_episodes.sort(key=lambda pair: -pair[0])
            episodes = [ep for _score, ep in scored_episodes[:limit]]

        context = "\n".join(edge.fact for edge in edges if edge.fact)
        return SearchResults(
            edges=edges, nodes=nodes, episodes=episodes, context=context
        )


class _BatchApi:
    def __init__(self, client: "LocalGraphClient"):
        self._client = client

    def create(self, **kwargs: Any) -> BatchSummary:
        return self._client.store.create_batch(kwargs.get("metadata") or {})

    def add(self, **kwargs: Any) -> List[Any]:
        batch_id = kwargs.get("batch_id")
        items = kwargs.get("items") or []
        store = self._client.store
        if store.get_batch(batch_id) is None:
            raise _not_found(f"batch {batch_id} not found")
        details = []
        for item in items:
            graph_id = getattr(item, "graph_id", "") or ""
            data = getattr(item, "data", "") or ""
            episode = store.add_episode(
                graph_id,
                str(data),
                type=str(getattr(item, "data_type", "text") or "text"),
                source_description=str(getattr(item, "source_description", "") or ""),
                metadata=getattr(item, "metadata", None) or {},
            )
            detail = store.add_batch_item(
                batch_id,
                graph_id=graph_id,
                data=str(data),
                data_type=str(getattr(item, "data_type", "text") or "text"),
                source_description=str(getattr(item, "source_description", "") or ""),
                metadata=getattr(item, "metadata", None) or {},
                episode_uuid=episode.uuid_,
            )
            details.append(detail)
        return details

    def process(self, **kwargs: Any) -> BatchSummary:
        batch_id = kwargs.get("batch_id")
        store = self._client.store
        summary = store.get_batch(batch_id)
        if summary is None:
            raise _not_found(f"batch {batch_id} not found")
        store.set_batch_status(batch_id, "processing")
        thread = threading.Thread(
            target=self._process_batch,
            args=(batch_id,),
            name=f"local-graph-batch-{batch_id}",
            daemon=True,
        )
        thread.start()
        return store.get_batch(batch_id) or summary

    def _process_batch(self, batch_id: str) -> None:
        store = self._client.store
        items = store.list_batch_items(batch_id)

        def process_one(item) -> bool:
            payload = store.get_batch_item_payload(batch_id, item.sequence_index)
            if payload is None:
                return False
            try:
                self._client.extractor.extract_episode(
                    payload["graph_id"],
                    payload["episode_uuid"],
                    payload["data"],
                )
                store.update_batch_item(
                    batch_id, item.sequence_index, status="succeeded"
                )
                return True
            except Exception as error:  # extraction quality must not hang polls
                logger.error(
                    "Local extraction failed for batch %s item %s: %s",
                    batch_id,
                    item.sequence_index,
                    error,
                )
                store.update_batch_item(
                    batch_id,
                    item.sequence_index,
                    status="failed",
                    error=str(error),
                )
                if payload["episode_uuid"]:
                    store.mark_episode_processed(payload["episode_uuid"])
                return False

        # Fire the extraction calls through a worker pool so they burst into
        # the LLM endpoint's queue together and can be answered in batches.
        # Ordering guarantees consumers rely on are preserved: item results
        # are recorded per sequence_index, batch status is set only after
        # every item finishes, and all store writes (including dedupe
        # upserts) are serialized by the store's write lock.
        workers = max(1, int(Config.LOCAL_GRAPH_EXTRACTION_WORKERS or 8))
        if workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix=f"local-graph-batch-{batch_id}",
            ) as pool:
                results = list(pool.map(process_one, items))
        else:
            results = [process_one(item) for item in items]

        succeeded = sum(1 for ok in results if ok)
        failed = len(results) - succeeded
        if failed == 0:
            status = "succeeded"
        elif succeeded == 0:
            status = "failed"
        else:
            status = "partial"
        store.set_batch_status(batch_id, status)

    def get(self, **kwargs: Any) -> BatchSummary:
        batch_id = kwargs.get("batch_id")
        summary = self._client.store.get_batch(batch_id)
        if summary is None:
            raise _not_found(f"batch {batch_id} not found")
        return summary

    def list(self, **kwargs: Any) -> BatchListPage:
        limit = kwargs.get("limit") or 100
        cursor = kwargs.get("cursor")
        offset = int(cursor) if cursor is not None else 0
        batches = self._client.store.list_batches()
        page = batches[offset : offset + int(limit)]
        next_cursor = offset + int(limit) if offset + int(limit) < len(batches) else None
        return BatchListPage(batches=page, next_cursor=next_cursor)

    def list_items(self, **kwargs: Any) -> BatchItemListPage:
        batch_id = kwargs.get("batch_id")
        limit = kwargs.get("limit") or 100
        cursor = kwargs.get("cursor")
        offset = int(cursor) if cursor is not None else 0
        items = self._client.store.list_batch_items(batch_id)
        page = items[offset : offset + int(limit)]
        next_cursor = offset + int(limit) if offset + int(limit) < len(items) else None
        return BatchItemListPage(items=page, next_cursor=next_cursor)


class LocalGraphClient:
    """Process-local client facade backed by SQLite + LLM extraction."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        llm_client_factory: Any = None,
    ):
        self.store = LocalGraphStore(db_path or Config.LOCAL_GRAPH_DB_PATH)
        self.extractor = LocalGraphExtractor(
            self.store, llm_client_factory=llm_client_factory
        )
        self.graph = _GraphApi(self)
        self.batch = _BatchApi(self)

    def spawn_extraction(self, graph_id: str, episode_uuid: str) -> None:
        """Run extraction for a fire-and-forget ``graph.add`` episode."""

        def _run() -> None:
            episode = self.store.get_episode(episode_uuid)
            if episode is None:
                return
            try:
                self.extractor.extract_episode(graph_id, episode_uuid, episode.content)
            except Exception as error:
                logger.error(
                    "Local extraction failed for episode %s: %s", episode_uuid, error
                )
                self.store.mark_episode_processed(episode_uuid)

        thread = threading.Thread(
            target=_run, name=f"local-graph-episode-{episode_uuid}", daemon=True
        )
        thread.start()


_singleton: Optional[LocalGraphClient] = None
_singleton_lock = threading.Lock()


def get_local_graph_client(db_path: Optional[str] = None) -> LocalGraphClient:
    """Return the process-wide local graph client (one per DB path)."""
    global _singleton
    with _singleton_lock:
        resolved = db_path or Config.LOCAL_GRAPH_DB_PATH
        if _singleton is None or _singleton.store.db_path != resolved:
            _singleton = LocalGraphClient(db_path=resolved)
        return _singleton


def clear_local_graph_client_cache() -> None:
    global _singleton
    with _singleton_lock:
        _singleton = None
