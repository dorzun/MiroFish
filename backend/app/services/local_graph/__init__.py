"""Self-hosted local graph backend (Zep Cloud replacement).

SQLite storage + LLM extraction behind a duck-typed client exposing the
zep_cloud attribute paths MiroFish uses. Selected when
``Config.GRAPH_BACKEND == 'local'`` (the default in this fork).
"""

from .client import (
    LocalGraphClient,
    clear_local_graph_client_cache,
    get_local_graph_client,
)

__all__ = [
    "LocalGraphClient",
    "clear_local_graph_client_cache",
    "get_local_graph_client",
]
