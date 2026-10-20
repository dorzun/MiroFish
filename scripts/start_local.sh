#!/usr/bin/env bash
# MiroFish (dorzun fork) - one-command local start.
# Starts: Muse LLM Bridge (:8791), backend (:5001), frontend (:3000).
# The host agent (Muse) answers the bridge queue, acting as the model.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# Local services must not inherit proxy env: httpx crashes building proxy
# mounts from NO_PROXY entries that contain bracketed IPv6 literals
# ("Invalid port: ':1]'"). The bridge and backend only talk to localhost.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy || true
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$HOME/.cache/tiktoken}"

echo "[1/3] Muse LLM Bridge :8791"
(cd llm-bridge && nohup .venv/bin/python bridge.py > /tmp/mirofish-bridge.log 2>&1 &) || true
echo "[2/3] Backend :5001"
(cd backend && nohup .venv/bin/python run.py > /tmp/mirofish-backend.log 2>&1 &) || true
echo "[3/3] Frontend :3000"
(cd frontend && nohup npm run dev > /tmp/mirofish-frontend.log 2>&1 &) || true

sleep 6
curl -s http://127.0.0.1:8791/health >/dev/null && echo "bridge  OK  http://localhost:8791"
curl -s http://127.0.0.1:5001/health >/dev/null && echo "backend OK  http://localhost:5001"
echo "frontend    http://localhost:3000 (vite, may need a few more seconds)"
echo "Logs: /tmp/mirofish-{bridge,backend,frontend}.log"
