#!/usr/bin/env bash
# clear.sh — Remove all Python runtime artifacts from the dab_ai project

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Cleaning Python runtime artifacts in: $SCRIPT_DIR"

# 1. Remove all __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "  [✓] Removed __pycache__ directories"

# 2. Remove compiled bytecode files
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo "  [✓] Removed .pyc / .pyo files"

# 3. Remove mypy / pyrefly / pyright caches
find . -type d -name ".mypy_cache"   -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pyright"      -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pyrefly_cache" -exec rm -rf {} + 2>/dev/null || true
echo "  [✓] Removed type-checker caches"

# 4. Remove pytest / coverage artifacts
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name ".coverage"     -delete 2>/dev/null || true
find . -type d -name "htmlcov"       -exec rm -rf {} + 2>/dev/null || true
echo "  [✓] Removed pytest / coverage artifacts"

# 5. Remove egg-info and dist-info (if any packaging was run)
find . -type d -name "*.egg-info"    -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "*.dist-info"   -exec rm -rf {} + 2>/dev/null || true
echo "  [✓] Removed egg/dist-info directories"

# 6. Clean up running GPU python processes from interrupted runs
if command -v nvidia-smi &> /dev/null; then
    echo "Checking for running GPU python processes..."
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -E '^[0-9]+$' || true)
    if [ -n "$gpu_pids" ]; then
        for pid in $gpu_pids; do
            if ps -p "$pid" -o comm= 2>/dev/null | grep -iq "python"; then
                echo "  Killing GPU python process: PID $pid"
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
        echo "  [✓] GPU python processes cleaned"
    else
        echo "  [✓] No GPU python processes found"
    fi
fi

# 7. Clean up running TensorBoard processes
echo "Checking for running TensorBoard processes..."
pkill -9 -f "tensorboard" 2>/dev/null || true
echo "  [✓] TensorBoard processes cleaned"

# 8. Clean up running uvicorn processes
echo "Checking for running uvicorn processes..."
pkill -9 -f "uvicorn" 2>/dev/null || true
echo "  [✓] Uvicorn processes cleaned"

echo ""
echo "Done. Project is clean."

clear