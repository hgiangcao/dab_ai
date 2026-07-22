import os
from pathlib import Path

# ===================
# Paths
# ===================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Replay pipeline root and subfolders
REPLAY_DIR = os.path.join(ROOT_DIR, "storage", "replay")
REPLAY_INCOMING = os.path.join(REPLAY_DIR, "incoming")
REPLAY_READY = os.path.join(REPLAY_DIR, "ready")
REPLAY_TRAINING = os.path.join(REPLAY_DIR, "training")
REPLAY_USED = os.path.join(REPLAY_DIR, "used")
REPLAY_MERGED = os.path.join(REPLAY_DIR, "merged")

# The global version file at the root of the project
VERSION_FILE = os.path.join(ROOT_DIR, "version.txt")


def get_current_model_dir():
    run_name = "run_1"
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, "r") as f:
            for line in f:
                if line.startswith("run:"):
                    run_name = line.split(":", 1)[1].strip()
                    break
    return os.path.join(LOGS_DIR, run_name)


# ===================
# Training defaults
# ===================

BATCH_SIZE = 512
EPOCHS = 10
LEARNING_RATE = 0.0005
MAX_WORKERS = 8

# Replay buffer defaults
MAX_REPLAY_SIZE = 20000
MIN_REPLAY_SIZE = 2000

# Evaluation defaults
EVAL_GAMES = 50
PROMOTION_THRESHOLD = 0.52


# ===================
# MCTS defaults
# ===================

MCTS_NUM_SIMULATIONS = 200
MCTS_C_PUCT = 1.0
MCTS_TEMPERATURE = 1.0
MCTS_DIRICHLET_ALPHA = 0.2
MCTS_DIRICHLET_EPS = 0.25


def get_mcts_config():
    return {
        "n_simulations": MCTS_NUM_SIMULATIONS,
        "c_puct": MCTS_C_PUCT,
        "temperature": MCTS_TEMPERATURE,
        "dirichlet_alpha": MCTS_DIRICHLET_ALPHA,
        "dirichlet_eps": MCTS_DIRICHLET_EPS,
    }


def get_server_config():
    mcts_config = get_mcts_config()
    return {
        "num_mcts": mcts_config["n_simulations"],
        "cpuct": mcts_config["c_puct"],
        "c_puct": mcts_config["c_puct"],
        "temperature": mcts_config["temperature"],
        "dirichlet_alpha": mcts_config["dirichlet_alpha"],
        "dirichlet_eps": mcts_config["dirichlet_eps"],
    }


# ===================
# Client endpoints
# ===================

MODEL_DOWNLOAD_URL = "/latest_model"
BEST_MODEL_DOWNLOAD_URL = "/best_model"
VERSION_API = "/version"
UPLOAD_REPLAY_API = "/upload_replay"

# Curriculum phases configuration
PHASES_CONFIG = [
    [("random", 0.01)],
    [("greedy", 0.02)],
    [("greedy_chain", 0.02)],
    [("simple_bot", 0.04)],
    [("simple_bot_v2", 0.05)],
    [("ucla_bot_v3", 0.06)],
    [("self", 0.6), ("best", 0.1), ("past", 0.15)]
]

