"""Pytest bootstrap for the backend suite.

The pre-existing ``test_zep_*`` modules are contract tests for the Zep Cloud
backend: they construct real ``Zep`` clients through ``get_zep_client`` and
assert Cloud-specific behavior (explicit base URL, timeout policy,
``ZEP_API_URL`` rejection). This fork defaults ``GRAPH_BACKEND`` to
``"local"`` at runtime, so pin the test process to the zep backend before any
``app`` module reads the environment into ``Config``. Local-backend behavior
is covered separately (tests may monkeypatch ``Config.GRAPH_BACKEND``).
"""

import os

os.environ.setdefault("GRAPH_BACKEND", "zep")
