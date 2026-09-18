#!/bin/bash
# startup.sh — container entrypoint for the LearnedFP framework.
#
# The framework is collection-only (Node) — there is no model bootstrap
# step. /save and /save-batch write to $PERSIST_DIR; mount a persistent
# volume there in production.
#
# Env vars:
#   PERSIST_DIR — parent of learnedfp/ + audio/ stores (default /data)
#   PORT        — injected by the platform; defaults to 3000

set -euo pipefail

PERSIST_DIR="${PERSIST_DIR:-/data}"
PORT="${PORT:-3000}"

mkdir -p "$PERSIST_DIR/learnedfp/save" "$PERSIST_DIR/audio/save"

exec node server/index.js --port "$PORT" --open=false
