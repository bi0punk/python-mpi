#!/usr/bin/env bash
set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

echo "=== MPI Job Runner ==="
echo ""

echo "[1/2] Starting master server on port 8000..."
python -m master.app &
MASTER_PID=$!
sleep 2

echo "[2/2] Launching MPI workers (4 processes)..."
echo ""
mpirun --allow-run-as-root -np 4 --hostfile machines \
    python -m worker.coordinator

echo ""
echo "=== Master stopped. Bye! ==="
kill $MASTER_PID 2>/dev/null
