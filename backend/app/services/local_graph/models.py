"""Duck-typed result objects mirroring the zep_cloud SDK shapes MiroFish reads.

Every attribute accessed anywhere in the codebase or tests on Zep results is
present here: nodes expose ``uuid_``, ``name``, ``labels``, ``summary``,
``attributes`` and ``created_at``; edges additionally expose ``fact``,
``fact_type``, ``source_node_uuid``, ``target_node_uuid``, temporal fields and
``episodes``; episodes expose ``uuid_`` and ``processed``; batch objects expose
the status / progress / item fields the GraphBuilder polls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LocalGraph:
    graph_id: str
    name: str = ""
    description: str = ""
    created_at: Optional[str] = None


@dataclass
class LocalNode:
    uuid_: str
    name: str = ""
    labels: List[str] = field(default_factory=list)
    summary: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    graph_id: str = ""

    @property
    def uuid(self) -> str:  # some consumers fall back to ``.uuid``
        return self.uuid_


@dataclass
class LocalEdge:
    uuid_: str
    name: str = ""
    fact: str = ""
    fact_type: str = ""
    source_node_uuid: str = ""
    target_node_uuid: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    episodes: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    graph_id: str = ""

    @property
    def uuid(self) -> str:
        return self.uuid_

    @property
    def episode_ids(self) -> List[str]:
        return self.episodes


@dataclass
class LocalEpisode:
    uuid_: str
    graph_id: str = ""
    content: str = ""
    type: str = "text"
    source: str = ""
    source_description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    processed: bool = False

    @property
    def uuid(self) -> str:
        return self.uuid_

    @property
    def data(self) -> str:
        return self.content


@dataclass
class RawGraphPage:
    """Shape returned by ``with_raw_response.get_by_graph_id`` calls."""

    data: List[Any] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class EpisodeListPage:
    episodes: List[LocalEpisode] = field(default_factory=list)


@dataclass
class SearchResults:
    edges: List[LocalEdge] = field(default_factory=list)
    nodes: List[LocalNode] = field(default_factory=list)
    episodes: List[LocalEpisode] = field(default_factory=list)
    observations: List[Any] = field(default_factory=list)
    thread_summaries: List[Any] = field(default_factory=list)
    context: str = ""


@dataclass
class BatchProgress:
    percent_complete: float = 0.0
    succeeded_items: int = 0
    failed_items: int = 0
    total_items: int = 0


@dataclass
class BatchItem:
    batch_id: str = ""
    sequence_index: int = 0
    status: str = "pending"
    episode_uuid: Optional[str] = None
    source_uuid: Optional[str] = None
    error: Any = None


@dataclass
class BatchSummary:
    batch_id: str = ""
    status: str = "draft"
    metadata: Dict[str, Any] = field(default_factory=dict)
    progress: BatchProgress = field(default_factory=BatchProgress)
    created_at: Optional[str] = None


@dataclass
class BatchListPage:
    batches: List[BatchSummary] = field(default_factory=list)
    next_cursor: Optional[int] = None


@dataclass
class BatchItemListPage:
    items: List[BatchItem] = field(default_factory=list)
    next_cursor: Optional[int] = None
