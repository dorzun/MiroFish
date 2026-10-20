"""SQLite storage for the local graph backend.

One database file holds every graph: graphs, their ontology, episodes,
extracted nodes/edges, and the batch-ingestion bookkeeping the GraphBuilder
polls. Writes are serialized with a process-wide lock and short-lived
connections (WAL mode) so the Flask app, background ingestion threads and
simulation updaters can share the file safely within one process.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models import (
    BatchItem,
    BatchProgress,
    BatchSummary,
    LocalEdge,
    LocalEpisode,
    LocalGraph,
    LocalNode,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS graphs (
    graph_id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    ontology TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    uuid TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    type TEXT NOT NULL DEFAULT 'text',
    source_description TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    processed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_episodes_graph ON episodes(graph_id);
CREATE TABLE IF NOT EXISTS nodes (
    uuid TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    normalized_name TEXT NOT NULL DEFAULT '',
    labels TEXT NOT NULL DEFAULT '[]',
    summary TEXT NOT NULL DEFAULT '',
    attributes TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nodes_graph ON nodes(graph_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_nodes_dedupe
    ON nodes(graph_id, normalized_name, labels);
CREATE TABLE IF NOT EXISTS edges (
    uuid TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    fact TEXT NOT NULL DEFAULT '',
    fact_type TEXT NOT NULL DEFAULT '',
    source_node_uuid TEXT NOT NULL,
    target_node_uuid TEXT NOT NULL,
    attributes TEXT NOT NULL DEFAULT '{}',
    episodes TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_edges_graph ON edges(graph_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_dedupe
    ON edges(graph_id, source_node_uuid, target_node_uuid, name, fact);
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'draft',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batch_items (
    batch_id TEXT NOT NULL,
    sequence_index INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    episode_uuid TEXT,
    error TEXT,
    graph_id TEXT NOT NULL DEFAULT '',
    data TEXT NOT NULL DEFAULT '',
    data_type TEXT NOT NULL DEFAULT 'text',
    source_description TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (batch_id, sequence_index)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_uuid() -> str:
    return uuid.uuid4().hex


def normalize_entity_name(name: str) -> str:
    """Canonical form used for entity deduplication."""
    return " ".join(str(name).split()).casefold()


class LocalGraphStore:
    """Thread-safe SQLite-backed store for local graphs."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        directory = os.path.dirname(os.path.abspath(db_path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._write_lock = threading.RLock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ------------------------------------------------------------------ graphs
    def create_graph(self, graph_id: str, name: str, description: str) -> LocalGraph:
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO graphs(graph_id, name, description, created_at)"
                " VALUES (?, ?, ?, ?)",
                (graph_id, name, description, _now()),
            )
        graph = self.get_graph(graph_id)
        assert graph is not None
        return graph

    def get_graph(self, graph_id: str) -> Optional[LocalGraph]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM graphs WHERE graph_id = ?", (graph_id,)
            ).fetchone()
        if row is None:
            return None
        return LocalGraph(
            graph_id=row["graph_id"],
            name=row["name"],
            description=row["description"],
            created_at=row["created_at"],
        )

    def delete_graph(self, graph_id: str) -> None:
        with self._write_lock, self._connect() as conn:
            conn.execute("DELETE FROM edges WHERE graph_id = ?", (graph_id,))
            conn.execute("DELETE FROM nodes WHERE graph_id = ?", (graph_id,))
            conn.execute("DELETE FROM episodes WHERE graph_id = ?", (graph_id,))
            conn.execute("DELETE FROM graphs WHERE graph_id = ?", (graph_id,))

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]) -> None:
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "UPDATE graphs SET ontology = ? WHERE graph_id = ?",
                (json.dumps(ontology, ensure_ascii=False), graph_id),
            )

    def get_ontology(self, graph_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT ontology FROM graphs WHERE graph_id = ?", (graph_id,)
            ).fetchone()
        if row is None:
            return {"entity_types": [], "edge_types": []}
        try:
            parsed = json.loads(row["ontology"] or "{}")
        except json.JSONDecodeError:
            parsed = {}
        parsed.setdefault("entity_types", [])
        parsed.setdefault("edge_types", [])
        return parsed

    # ---------------------------------------------------------------- episodes
    def add_episode(
        self,
        graph_id: str,
        content: str,
        *,
        type: str = "text",
        source_description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
        episode_uuid: Optional[str] = None,
    ) -> LocalEpisode:
        episode_uuid = episode_uuid or _new_uuid()
        created = created_at or _now()
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO episodes"
                "(uuid, graph_id, content, type, source_description, metadata,"
                " created_at, processed) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    episode_uuid,
                    graph_id,
                    content,
                    type,
                    source_description,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    created,
                ),
            )
        episode = self.get_episode(episode_uuid)
        assert episode is not None
        return episode

    def get_episode(self, episode_uuid: str) -> Optional[LocalEpisode]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM episodes WHERE uuid = ?", (episode_uuid,)
            ).fetchone()
        return self._episode_from_row(row) if row else None

    def list_episodes(self, graph_id: str, lastn: int = 50) -> List[LocalEpisode]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM episodes WHERE graph_id = ?"
                " ORDER BY created_at DESC LIMIT ?",
                (graph_id, max(1, int(lastn))),
            ).fetchall()
        return [self._episode_from_row(row) for row in rows]

    def mark_episode_processed(self, episode_uuid: str) -> None:
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "UPDATE episodes SET processed = 1 WHERE uuid = ?", (episode_uuid,)
            )

    @staticmethod
    def _episode_from_row(row: sqlite3.Row) -> LocalEpisode:
        return LocalEpisode(
            uuid_=row["uuid"],
            graph_id=row["graph_id"],
            content=row["content"],
            type=row["type"],
            source_description=row["source_description"],
            metadata=json.loads(row["metadata"] or "{}"),
            created_at=row["created_at"],
            processed=bool(row["processed"]),
        )

    # ------------------------------------------------------------------- nodes
    def upsert_node(
        self,
        graph_id: str,
        name: str,
        label: str,
        summary: str = "",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> LocalNode:
        """Insert or merge an entity, deduped by (graph, normalized name, label)."""
        labels = ["Entity", label] if label and label != "Entity" else ["Entity"]
        labels_key = json.dumps(labels, ensure_ascii=False)
        normalized = normalize_entity_name(name)
        attributes = attributes or {}
        with self._write_lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM nodes WHERE graph_id = ? AND normalized_name = ?"
                " AND labels = ?",
                (graph_id, normalized, labels_key),
            ).fetchone()
            if row is not None:
                merged_attributes = dict(json.loads(row["attributes"] or "{}"))
                for key, value in attributes.items():
                    if value not in (None, ""):
                        merged_attributes[key] = value
                merged_summary = row["summary"] or summary
                conn.execute(
                    "UPDATE nodes SET summary = ?, attributes = ? WHERE uuid = ?",
                    (
                        merged_summary,
                        json.dumps(merged_attributes, ensure_ascii=False),
                        row["uuid"],
                    ),
                )
                node_uuid = row["uuid"]
            else:
                node_uuid = _new_uuid()
                conn.execute(
                    "INSERT INTO nodes(uuid, graph_id, name, normalized_name,"
                    " labels, summary, attributes, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        node_uuid,
                        graph_id,
                        name.strip(),
                        normalized,
                        labels_key,
                        summary,
                        json.dumps(attributes, ensure_ascii=False),
                        _now(),
                    ),
                )
        node = self.get_node(node_uuid)
        assert node is not None
        return node

    def get_node(self, node_uuid: str) -> Optional[LocalNode]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM nodes WHERE uuid = ?", (node_uuid,)
            ).fetchone()
        return self._node_from_row(row) if row else None

    def list_nodes(self, graph_id: str) -> List[LocalNode]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM nodes WHERE graph_id = ? ORDER BY created_at, name",
                (graph_id,),
            ).fetchall()
        return [self._node_from_row(row) for row in rows]

    def find_node(self, graph_id: str, name: str, label: str = "") -> Optional[LocalNode]:
        labels = ["Entity", label] if label else ["Entity"]
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM nodes WHERE graph_id = ? AND normalized_name = ?"
                " AND labels = ?",
                (graph_id, normalize_entity_name(name), json.dumps(labels)),
            ).fetchone()
            if row is None and label:
                row = conn.execute(
                    "SELECT * FROM nodes WHERE graph_id = ? AND normalized_name = ?"
                    " ORDER BY created_at LIMIT 1",
                    (graph_id, normalize_entity_name(name)),
                ).fetchone()
        return self._node_from_row(row) if row else None

    @staticmethod
    def _node_from_row(row: sqlite3.Row) -> LocalNode:
        return LocalNode(
            uuid_=row["uuid"],
            name=row["name"],
            labels=json.loads(row["labels"] or "[]"),
            summary=row["summary"] or "",
            attributes=json.loads(row["attributes"] or "{}"),
            created_at=row["created_at"],
            graph_id=row["graph_id"],
        )

    # ------------------------------------------------------------------- edges
    def upsert_edge(
        self,
        graph_id: str,
        name: str,
        fact: str,
        source_node_uuid: str,
        target_node_uuid: str,
        *,
        fact_type: str = "",
        attributes: Optional[Dict[str, Any]] = None,
        episode_uuid: Optional[str] = None,
    ) -> LocalEdge:
        with self._write_lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM edges WHERE graph_id = ? AND source_node_uuid = ?"
                " AND target_node_uuid = ? AND name = ? AND fact = ?",
                (graph_id, source_node_uuid, target_node_uuid, name, fact),
            ).fetchone()
            if row is not None:
                episodes = json.loads(row["episodes"] or "[]")
                if episode_uuid and episode_uuid not in episodes:
                    episodes.append(episode_uuid)
                    conn.execute(
                        "UPDATE edges SET episodes = ? WHERE uuid = ?",
                        (json.dumps(episodes), row["uuid"]),
                    )
                edge_uuid = row["uuid"]
            else:
                edge_uuid = _new_uuid()
                conn.execute(
                    "INSERT INTO edges(uuid, graph_id, name, fact, fact_type,"
                    " source_node_uuid, target_node_uuid, attributes, episodes,"
                    " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        edge_uuid,
                        graph_id,
                        name,
                        fact,
                        fact_type or name,
                        source_node_uuid,
                        target_node_uuid,
                        json.dumps(attributes or {}, ensure_ascii=False),
                        json.dumps([episode_uuid] if episode_uuid else []),
                        _now(),
                    ),
                )
        edge = self.get_edge(edge_uuid)
        assert edge is not None
        return edge

    def get_edge(self, edge_uuid: str) -> Optional[LocalEdge]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM edges WHERE uuid = ?", (edge_uuid,)
            ).fetchone()
        return self._edge_from_row(row) if row else None

    def list_edges(self, graph_id: str) -> List[LocalEdge]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM edges WHERE graph_id = ? ORDER BY created_at",
                (graph_id,),
            ).fetchall()
        return [self._edge_from_row(row) for row in rows]

    def edges_for_node(self, node_uuid: str) -> List[LocalEdge]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM edges WHERE source_node_uuid = ?"
                " OR target_node_uuid = ? ORDER BY created_at",
                (node_uuid, node_uuid),
            ).fetchall()
        return [self._edge_from_row(row) for row in rows]

    @staticmethod
    def _edge_from_row(row: sqlite3.Row) -> LocalEdge:
        return LocalEdge(
            uuid_=row["uuid"],
            name=row["name"],
            fact=row["fact"],
            fact_type=row["fact_type"] or row["name"],
            source_node_uuid=row["source_node_uuid"],
            target_node_uuid=row["target_node_uuid"],
            attributes=json.loads(row["attributes"] or "{}"),
            episodes=json.loads(row["episodes"] or "[]"),
            created_at=row["created_at"],
            graph_id=row["graph_id"],
        )

    # ----------------------------------------------------------------- batches
    def create_batch(self, metadata: Optional[Dict[str, Any]]) -> BatchSummary:
        batch_id = _new_uuid()
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO batches(batch_id, status, metadata, created_at)"
                " VALUES (?, 'draft', ?, ?)",
                (batch_id, json.dumps(metadata or {}, ensure_ascii=False), _now()),
            )
        summary = self.get_batch(batch_id)
        assert summary is not None
        return summary

    def add_batch_item(
        self,
        batch_id: str,
        *,
        graph_id: str,
        data: str,
        data_type: str = "text",
        source_description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        episode_uuid: Optional[str] = None,
    ) -> BatchItem:
        with self._write_lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(sequence_index), -1) + 1 AS next_idx"
                " FROM batch_items WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            sequence_index = int(row["next_idx"])
            conn.execute(
                "INSERT INTO batch_items(batch_id, sequence_index, status,"
                " episode_uuid, graph_id, data, data_type, source_description,"
                " metadata) VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?)",
                (
                    batch_id,
                    sequence_index,
                    episode_uuid,
                    graph_id,
                    data,
                    data_type,
                    source_description,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
        item = self.get_batch_item(batch_id, sequence_index)
        assert item is not None
        return item

    def get_batch_item(self, batch_id: str, sequence_index: int) -> Optional[BatchItem]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM batch_items WHERE batch_id = ? AND sequence_index = ?",
                (batch_id, sequence_index),
            ).fetchone()
        return self._batch_item_from_row(row) if row else None

    def list_batch_items(self, batch_id: str) -> List[BatchItem]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM batch_items WHERE batch_id = ?"
                " ORDER BY sequence_index",
                (batch_id,),
            ).fetchall()
        return [self._batch_item_from_row(row) for row in rows]

    def get_batch_item_payload(
        self, batch_id: str, sequence_index: int
    ) -> Optional[Dict[str, Any]]:
        """Internal view of a batch item, including the stored episode text."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM batch_items WHERE batch_id = ? AND sequence_index = ?",
                (batch_id, sequence_index),
            ).fetchone()
        if row is None:
            return None
        return {
            "graph_id": row["graph_id"],
            "data": row["data"],
            "data_type": row["data_type"],
            "source_description": row["source_description"],
            "metadata": json.loads(row["metadata"] or "{}"),
            "episode_uuid": row["episode_uuid"],
            "status": row["status"],
        }

    def update_batch_item(
        self,
        batch_id: str,
        sequence_index: int,
        *,
        status: Optional[str] = None,
        episode_uuid: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        assignments = []
        values: List[Any] = []
        if status is not None:
            assignments.append("status = ?")
            values.append(status)
        if episode_uuid is not None:
            assignments.append("episode_uuid = ?")
            values.append(episode_uuid)
        if error is not None:
            assignments.append("error = ?")
            values.append(error)
        if not assignments:
            return
        values.extend([batch_id, sequence_index])
        with self._write_lock, self._connect() as conn:
            conn.execute(
                f"UPDATE batch_items SET {', '.join(assignments)}"
                " WHERE batch_id = ? AND sequence_index = ?",
                values,
            )

    def set_batch_status(self, batch_id: str, status: str) -> None:
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "UPDATE batches SET status = ? WHERE batch_id = ?",
                (status, batch_id),
            )

    def get_batch(self, batch_id: str) -> Optional[BatchSummary]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if row is None:
                return None
            counts = conn.execute(
                "SELECT status, COUNT(*) AS n FROM batch_items"
                " WHERE batch_id = ? GROUP BY status",
                (batch_id,),
            ).fetchall()
        by_status = {count_row["status"]: count_row["n"] for count_row in counts}
        total = sum(by_status.values())
        succeeded = by_status.get("succeeded", 0)
        failed = by_status.get("failed", 0)
        percent = (100.0 * (succeeded + failed) / total) if total else 0.0
        return BatchSummary(
            batch_id=row["batch_id"],
            status=row["status"],
            metadata=json.loads(row["metadata"] or "{}"),
            progress=BatchProgress(
                percent_complete=percent,
                succeeded_items=succeeded,
                failed_items=failed,
                total_items=total,
            ),
            created_at=row["created_at"],
        )

    def list_batches(self) -> List[BatchSummary]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT batch_id FROM batches ORDER BY created_at"
            ).fetchall()
        return [summary for row in rows if (summary := self.get_batch(row["batch_id"]))]

    @staticmethod
    def _batch_item_from_row(row: sqlite3.Row) -> BatchItem:
        error = row["error"]
        return BatchItem(
            batch_id=row["batch_id"],
            sequence_index=row["sequence_index"],
            status=row["status"],
            episode_uuid=row["episode_uuid"],
            source_uuid=row["episode_uuid"],
            error={"message": error} if error else None,
        )
