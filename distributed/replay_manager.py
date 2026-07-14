import os
import shutil
import glob
import numpy as np
from datetime import datetime, timedelta
import config

# -------------------------------------------------------
# Ensure all pipeline folders exist
# -------------------------------------------------------
for _d in [config.REPLAY_INCOMING, config.REPLAY_READY,
           config.REPLAY_TRAINING, config.REPLAY_USED, config.REPLAY_MERGED]:
    os.makedirs(_d, exist_ok=True)


# -------------------------------------------------------
# Step 1: Validate uploaded file and move to ready/
# -------------------------------------------------------

def validate_replay(filepath):
    """
    Check replay integrity.
    Verify that required keys exist and arrays have matching lengths.
    Returns (True, metadata_dict) or (False, error_str).
    """
    try:
        data = np.load(filepath, allow_pickle=False)
        required_keys = ['lines', 'boxes', 'pis', 'vals']
        for k in required_keys:
            if k not in data:
                return False, f"Missing key: {k}"

        n = len(data['lines'])
        if n == 0:
            return False, "Empty replay file"
        if not (len(data['boxes']) == n == len(data['pis']) == len(data['vals'])):
            return False, "Array length mismatch"

        # Read embedded metadata if present
        meta = {
            "worker_id":     str(data['worker_id'])     if 'worker_id'     in data else "unknown",
            "model_version": int(data['model_version']) if 'model_version' in data else -1,
            "game_count":    int(data['game_count'])    if 'game_count'    in data else n,
            "timestamp":     str(data['timestamp'])     if 'timestamp'     in data else "",
        }
        return True, meta
    except Exception as e:
        return False, str(e)


def validate_and_promote():
    """
    Scan incoming/, validate each file, and move valid ones to ready/.
    Returns list of promoted filenames.
    """
    promoted = []
    files = sorted(glob.glob(os.path.join(config.REPLAY_INCOMING, "*.npz")))

    for filepath in files:
        ok, result = validate_replay(filepath)
        fname = os.path.basename(filepath)
        if ok:
            dest = os.path.join(config.REPLAY_READY, fname)
            shutil.move(filepath, dest)
            promoted.append(dest)
            print(f"  [PROMOTED] {fname} → ready/ | meta={result}")
        else:
            print(f"  [INVALID]  {fname} → discarded | reason={result}")
            os.remove(filepath)

    return promoted


# -------------------------------------------------------
# Step 2: Claim files for training (anti-race)
# -------------------------------------------------------

def claim_for_training():
    """
    Atomically move all files from ready/ → training/.
    Returns list of file paths now in training/.
    """
    files = sorted(glob.glob(os.path.join(config.REPLAY_READY, "*.npz")))
    claimed = []
    for filepath in files:
        fname = os.path.basename(filepath)
        dest = os.path.join(config.REPLAY_TRAINING, fname)
        shutil.move(filepath, dest)
        claimed.append(dest)
    if claimed:
        print(f"Claimed {len(claimed)} files for training.")
    return claimed


# -------------------------------------------------------
# Step 3: Load and merge training files
# -------------------------------------------------------

def load_replay_buffer(files=None):
    """
    Load replay data from the given list of files (or all in training/).
    Returns a list of (lines, boxes, pi, value) tuples.
    """
    if files is None:
        files = sorted(glob.glob(os.path.join(config.REPLAY_TRAINING, "*.npz")))

    buffer = []
    for filepath in files:
        try:
            data = np.load(filepath, allow_pickle=False)
            for i in range(len(data['lines'])):
                buffer.append((data['lines'][i], data['boxes'][i], data['pis'][i], data['vals'][i]))
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")

    return buffer


def merge_replay(files=None):
    """
    Merge files in training/ into merged/replay_{timestamp}.npz.
    Returns path to merged file.
    """
    if files is None:
        files = sorted(glob.glob(os.path.join(config.REPLAY_TRAINING, "*.npz")))

    if not files:
        return None

    all_lines, all_boxes, all_pis, all_vals = [], [], [], []
    for filepath in files:
        try:
            data = np.load(filepath, allow_pickle=False)
            all_lines.append(data['lines'])
            all_boxes.append(data['boxes'])
            all_pis.append(data['pis'])
            all_vals.append(data['vals'])
        except Exception as e:
            print(f"Skipping {filepath}: {e}")

    if not all_lines:
        return None

    final_lines = np.concatenate(all_lines)
    final_boxes = np.concatenate(all_boxes)
    final_pis   = np.concatenate(all_pis)
    final_vals  = np.concatenate(all_vals)

    # Truncate to MAX_REPLAY_SIZE
    if len(final_lines) > config.MAX_REPLAY_SIZE:
        final_lines = final_lines[-config.MAX_REPLAY_SIZE:]
        final_boxes = final_boxes[-config.MAX_REPLAY_SIZE:]
        final_pis   = final_pis[-config.MAX_REPLAY_SIZE:]
        final_vals  = final_vals[-config.MAX_REPLAY_SIZE:]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_path = os.path.join(config.REPLAY_MERGED, f"replay_{timestamp}.npz")
    np.savez_compressed(merged_path,
                        lines=final_lines, boxes=final_boxes,
                        pis=final_pis, vals=final_vals)
    print(f"Merged {len(final_lines)} examples → {merged_path}")
    return merged_path


# -------------------------------------------------------
# Step 4: Mark consumed files as used
# -------------------------------------------------------

def mark_used(files):
    """
    Move files from training/ → used/ after training completes.
    """
    for filepath in files:
        fname = os.path.basename(filepath)
        dest = os.path.join(config.REPLAY_USED, fname)
        if os.path.exists(filepath):
            shutil.move(filepath, dest)
    print(f"Marked {len(files)} files as used.")


# -------------------------------------------------------
# Step 5: Cleanup old used files
# -------------------------------------------------------

def cleanup_old(days=7):
    """
    Remove files in used/ older than `days` days.
    """
    cutoff = datetime.now() - timedelta(days=days)
    removed = 0
    for filepath in glob.glob(os.path.join(config.REPLAY_USED, "*.npz")):
        mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
        if mtime < cutoff:
            os.remove(filepath)
            removed += 1
    if removed:
        print(f"Cleaned up {removed} old files from used/.")