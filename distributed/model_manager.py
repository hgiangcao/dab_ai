import os
import shutil
import glob
import torch
import config


# ─────────────────────────────────────────────────────────────────────────────
# version.txt helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_version_field(field: str, default):
    """Read a single named field from version.txt."""
    if os.path.exists(config.VERSION_FILE):
        with open(config.VERSION_FILE, "r") as f:
            for line in f:
                if line.startswith(f"{field}:"):
                    val = line.split(":", 1)[1].strip()
                    try:
                        return type(default)(val)
                    except (ValueError, TypeError):
                        return default
    return default


def _write_version_field(field: str, value):
    """Write/update a single named field in version.txt."""
    lines = []
    found = False
    if os.path.exists(config.VERSION_FILE):
        with open(config.VERSION_FILE, "r") as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            if line.startswith(f"{field}:"):
                new_lines.append(f"{field}: {value}\n")
                found = True
            else:
                new_lines.append(line)
        lines = new_lines
    if not found:
        lines.append(f"{field}: {value}\n")
    with open(config.VERSION_FILE, "w") as f:
        f.writelines(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Version / phase state
# ─────────────────────────────────────────────────────────────────────────────

def get_current_version() -> int:
    return _read_version_field("last_updated_model", 0)


def increase_version() -> int:
    new_version = get_current_version() + 1
    _write_version_field("last_updated_model", new_version)
    return new_version


def set_pretrain_finished():
    _write_version_field("finish_pretrain", "True")


def get_current_phase() -> int:
    return _read_version_field("current_phase", 0)


def advance_curriculum_phase() -> int:
    current = get_current_phase()
    new_phase = min(len(config.PHASES_CONFIG) - 1, current + 1)
    _write_version_field("current_phase", new_phase)
    return new_phase


# ─────────────────────────────────────────────────────────────────────────────
# Best overall score persistence
# ─────────────────────────────────────────────────────────────────────────────

def get_best_overall_score() -> float:
    return _read_version_field("best_overall_score", 0.0)


def set_best_overall_score(score: float):
    _write_version_field("best_overall_score", f"{score:.6f}")


def get_best_min_bot_winrate() -> float:
    return _read_version_field("best_min_bot_winrate", 0.0)


def set_best_min_bot_winrate(score: float):
    _write_version_field("best_min_bot_winrate", f"{score:.6f}")


def get_best_model_checkpoint() -> int:
    """Return the last_updated_model version when best.pth.tar was last updated."""
    return _read_version_field("best_model_checkpoint", 0)


def set_best_model_checkpoint(version: int):
    """Persist which checkpoint number corresponds to the current best.pth.tar."""
    _write_version_field("best_model_checkpoint", version)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint path helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_model_dir() -> str:
    return config.get_current_model_dir()


def get_candidate_path() -> str:
    return os.path.join(get_model_dir(), "checkpoint_candidate.pth.tar")


def get_current_model_path() -> str:
    """
    Returns the path to the latest promoted checkpoint (self/current).
    Falls back to best.pth.tar if checkpoint_current does not exist yet.
    """
    current_path = os.path.join(get_model_dir(), "checkpoint_current.pth.tar")
    if os.path.exists(current_path):
        return current_path
    # Backward-compat fallback
    return get_best_model_path()


def get_best_model_path() -> str:
    return os.path.join(get_model_dir(), "best.pth.tar")


def get_latest_model_path(version=None) -> str:
    if version is None:
        version = get_current_version()
    return os.path.join(get_model_dir(), f"checkpoint_{version}.pth.tar")


def get_past_dir() -> str:
    past_dir = os.path.join(get_model_dir(), "past")
    os.makedirs(past_dir, exist_ok=True)
    return past_dir


def get_past_model_paths() -> list:
    """Return sorted list of all past checkpoint paths."""
    past_dir = get_past_dir()
    paths = sorted(glob.glob(os.path.join(past_dir, "checkpoint_*.pth.tar")))
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint management
# ─────────────────────────────────────────────────────────────────────────────

def save_latest_model(model):
    """
    Save candidate model with optimizer and scheduler states.
    Written to checkpoint_candidate.pth.tar.
    """
    filepath = get_candidate_path()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    state = {}
    if hasattr(model, "nnet"):
        state["state_dict"] = model.nnet.state_dict()
        if hasattr(model, "optimizer"):
            state["optimizer"] = model.optimizer.state_dict()
        if hasattr(model, "scheduler"):
            state["scheduler"] = model.scheduler.state_dict()
    elif hasattr(model, "state_dict"):
        state["state_dict"] = model.state_dict()
    else:
        state = model  # already a dict

    torch.save(state, filepath)
    return filepath


def promote_to_current():
    """
    Promote candidate → current/self.

    Actions:
      1. Increment version counter.
      2. Copy candidate → checkpoint_current.pth.tar   (workers' self model)
      3. Copy candidate → past/checkpoint_N.pth.tar    (history)
    """
    new_version = increase_version()
    candidate = get_candidate_path()
    model_dir = get_model_dir()
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(candidate):
        print(f"[ModelManager] WARNING: candidate not found at {candidate}")
        return new_version

    # 1. current/self checkpoint
    current_path = os.path.join(model_dir, "checkpoint_current.pth.tar")
    shutil.copyfile(candidate, current_path)
    print(f"[ModelManager] Promoted candidate → checkpoint_current  (v{new_version})")

    # 2. numbered latest checkpoint (for backward compat)
    latest_path = get_latest_model_path(new_version)
    shutil.copyfile(candidate, latest_path)

    # 3. historical past copy
    past_path = os.path.join(get_past_dir(), f"checkpoint_{new_version}.pth.tar")
    shutil.copyfile(candidate, past_path)
    print(f"[ModelManager] Saved to past → {past_path}")

    return new_version


def update_best():
    """
    Update best.pth.tar to the current candidate.
    Called independently from promote_to_current when the candidate
    scores >= BEST_UPDATE_SCORE_RATIO × current best score.
    Also records the current last_updated_model version as best_model_checkpoint.
    """
    candidate = get_candidate_path()
    best_path = get_best_model_path()
    os.makedirs(os.path.dirname(best_path), exist_ok=True)

    if os.path.exists(candidate):
        shutil.copyfile(candidate, best_path)
        current_version = get_current_version()
        set_best_model_checkpoint(current_version)
        print(f"[ModelManager] Updated best.pth.tar from candidate (checkpoint v{current_version}).")
    else:
        print(f"[ModelManager] WARNING: candidate not found, best.pth.tar not updated.")


def promote_best_model():
    """
    Legacy compatibility shim: promote candidate to both current and best.
    Prefer calling promote_to_current() + update_best() separately.
    """
    promote_to_current()
    update_best()


def load_model(filepath):
    """Load PyTorch model state_dict from file."""
    if not os.path.exists(filepath):
        return None
    state_dict = torch.load(filepath, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        return state_dict["state_dict"]
    return state_dict