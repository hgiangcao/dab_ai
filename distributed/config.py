import importlib.util
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.py"

spec = importlib.util.spec_from_file_location("project_config", CONFIG_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load shared config from {CONFIG_PATH}")

project_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(project_config)

ROOT_DIR = project_config.ROOT_DIR
LOGS_DIR = project_config.LOGS_DIR
REPLAY_DIR = project_config.REPLAY_DIR
REPLAY_INCOMING = project_config.REPLAY_INCOMING
REPLAY_READY = project_config.REPLAY_READY
REPLAY_TRAINING = project_config.REPLAY_TRAINING
REPLAY_USED = project_config.REPLAY_USED
REPLAY_MERGED = project_config.REPLAY_MERGED
VERSION_FILE = project_config.VERSION_FILE


def get_current_model_dir():
    return project_config.get_current_model_dir()


BATCH_SIZE = project_config.BATCH_SIZE
EPOCHS = project_config.EPOCHS
LEARNING_RATE = project_config.LEARNING_RATE
MAX_REPLAY_SIZE = project_config.MAX_REPLAY_SIZE
MIN_REPLAY_SIZE = project_config.MIN_REPLAY_SIZE
EVAL_GAMES = project_config.EVAL_GAMES
PROMOTION_THRESHOLD = project_config.PROMOTION_THRESHOLD

MCTS_NUM_SIMULATIONS = project_config.MCTS_NUM_SIMULATIONS
MCTS_C_PUCT = project_config.MCTS_C_PUCT
MCTS_TEMPERATURE = project_config.MCTS_TEMPERATURE
MCTS_DIRICHLET_ALPHA = project_config.MCTS_DIRICHLET_ALPHA
MCTS_DIRICHLET_EPS = project_config.MCTS_DIRICHLET_EPS


def get_mcts_config():
    return project_config.get_mcts_config()


def get_server_config():
    return project_config.get_server_config()


MODEL_DOWNLOAD_URL = project_config.MODEL_DOWNLOAD_URL
BEST_MODEL_DOWNLOAD_URL = project_config.BEST_MODEL_DOWNLOAD_URL
VERSION_API = project_config.VERSION_API
UPLOAD_REPLAY_API = project_config.UPLOAD_REPLAY_API