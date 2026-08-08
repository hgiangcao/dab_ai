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

BATCH_SIZE = 1024
EPOCHS = 10
LEARNING_RATE = 0.0005
MAX_WORKERS = 8

# Replay buffer defaults
MAX_REPLAY_SIZE = 20000
MIN_REPLAY_SIZE = 2000

# Evaluation defaults
EVAL_GAMES_VS_BOTS    = 10   # games vs each external bot during screening
EVAL_GAMES_VS_CURRENT = 50   # games vs current/self checkpoint during screening

# Promotion: BOTH conditions must pass
MIN_BOT_WINRATE_FOR_PROMOTION = 0.80   # min(bot_winrates) threshold
MIN_WINRATE_VS_CURRENT        = 0.55   # winrate vs current checkpoint threshold

# Best-model update: candidate qualifies when its overall bot score is at least
# this fraction of the stored best score
BEST_UPDATE_SCORE_RATIO = 0.90

# Legacy (kept for evaluator backward compat, no longer used for promotion)
EVAL_GAMES = 50
PROMOTION_THRESHOLD = 0.55


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

# Curriculum phases configuration — gradual accumulation per spec
PHASES_CONFIG = [
    # Phase 0: establish basic strategy
    [("greedy", 0.80), ("self", 0.20)],

    # Phase 1: introduce stronger bot
    [("greedy", 0.50), ("simple_bot_v2", 0.30), ("self", 0.20)],

    # Phase 2: increase simple_bot_v2
    [("greedy", 0.30), ("simple_bot_v2", 0.40), ("ucla_bot_v3", 0.10), ("self", 0.20)],

    # Phase 3: introduce strong UCLA bot
    [("greedy", 0.20), ("simple_bot_v2", 0.30), ("ucla_bot_v3", 0.30), ("self", 0.20)],

    # Phase 4: UCLA becomes dominant
    [("greedy", 0.10), ("simple_bot_v2", 0.25), ("ucla_bot_v3", 0.40), ("self", 0.25)],

    # Phase 5: strong RL
    [("greedy", 0.10), ("simple_bot_v2", 0.20), ("ucla_bot_v3", 0.35),
     ("self", 0.25), ("best", 0.10)],

    # Phase 6: late RL
    [("greedy", 0.05), ("simple_bot_v2", 0.15), ("ucla_bot_v3", 0.30),
     ("self", 0.30), ("best", 0.10), ("past", 0.10)],

    # Phase 7: steady-state
    [("greedy", 0.05), ("simple_bot_v2", 0.10), ("ucla_bot_v3", 0.25),
     ("self", 0.40), ("best", 0.10), ("past", 0.10)],
]

PHASE_ADVANCE_THRESHOLD = {
    0: 0.80,  # greedy
    1: 0.75,  # greedy + simple_bot_v2
    2: 0.75,  # introduce UCLA
    3: 0.80,
    4: 0.85,
    5: 0.85,
    6: 0.85,
    7: 0.85,
}