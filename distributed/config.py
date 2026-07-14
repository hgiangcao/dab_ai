import os
from pathlib import Path

# ===================
# Paths
# ===================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# We store our runs in the logs directory (e.g., logs/run_1)
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Replay pipeline root and subfolders
REPLAY_DIR = os.path.join(ROOT_DIR, "storage", "replay")
REPLAY_INCOMING = os.path.join(REPLAY_DIR, "incoming")
REPLAY_READY    = os.path.join(REPLAY_DIR, "ready")
REPLAY_TRAINING = os.path.join(REPLAY_DIR, "training")
REPLAY_USED     = os.path.join(REPLAY_DIR, "used")
REPLAY_MERGED   = os.path.join(REPLAY_DIR, "merged")

# The global version file at the root of the project
VERSION_FILE = os.path.join(ROOT_DIR, "version.txt")

# Helper function to get the current model directory dynamically
def get_current_model_dir():
    run_name = "run_1"
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, "r") as f:
            for line in f:
                if line.startswith("run:"):
                    run_name = line.split(":")[1].strip()
                    break
    return os.path.join(LOGS_DIR, run_name)

# ===================
# Training (Matching main.py args)
# ===================

BATCH_SIZE = 512

EPOCHS = 100

LEARNING_RATE = 0.0005


# ===================
# Replay Buffer
# ===================

# maxlen_queue from main.py
MAX_REPLAY_SIZE = 200000

MIN_REPLAY_SIZE = 10000


# ===================
# Evaluation
# ===================

# arena_games from main.py
EVAL_GAMES = 100

# update_threshold from main.py
PROMOTION_THRESHOLD = 0.55


# ===================
# Client (Matching server.py endpoints)
# ===================

MODEL_DOWNLOAD_URL = "/latest_model"
BEST_MODEL_DOWNLOAD_URL = "/best_model"

VERSION_API = "/version"
UPLOAD_REPLAY_API = "/upload_replay"