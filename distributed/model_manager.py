import os
import shutil
import torch
import config

def get_current_version():
    """
    Read version.txt

    Return:

    15 (integer of last_updated_model)
    """
    if os.path.exists(config.VERSION_FILE):
        with open(config.VERSION_FILE, "r") as f:
            for line in f:
                if line.startswith("last_updated_model:"):
                    try:
                        return int(line.split(":")[1].strip())
                    except ValueError:
                        break
    return 0

def increase_version():
    """
    Increase model version.

    Before:
    last_updated_model: 15

    After:
    last_updated_model: 16
    """
    version = get_current_version()
    new_version = version + 1
    
    if os.path.exists(config.VERSION_FILE):
        with open(config.VERSION_FILE, "r") as f:
            lines = f.readlines()
            
        with open(config.VERSION_FILE, "w") as f:
            for line in lines:
                if line.startswith("last_updated_model:"):
                    f.write(f"last_updated_model: {new_version}\n")
                else:
                    f.write(line)
    return new_version

def get_best_model_path():
    """
    Return:
    storage/models/best.pth (actually logs/run_x/best.pth.tar to match our structure)
    """
    print ("Load best model",config.get_current_model_dir(), "best.pth.tar")
    return os.path.join(config.get_current_model_dir(), "best.pth.tar")

def get_latest_model_path(version=None):
    """
    Return:
    storage/models/latest.pth (actually logs/run_x/checkpoint_{version}.pth.tar)
    """
    if version is None:
        version = get_current_version()
    
    
    print ("Load last model",config.get_current_model_dir(), f"checkpoint_{version}.pth.tar")
    return os.path.join(config.get_current_model_dir(), f"checkpoint_{version}.pth.tar")

def save_latest_model(model):
    """
    Save candidate model.
    latest.pth (actually checkpoint_{version}.pth.tar)
    """
    version = get_current_version()
    # When we are saving a new candidate, we might want to save it as version + 1 first
    # But usually, we just save to temp_latest.pth.tar or we save it to checkpoint_{v} AFTER increasing version
    # Let's save it to temp_latest.pth.tar as the candidate, or to a new checkpoint.
    # In coach.py, we save as checkpoint_{i}.pth.tar.
    # We will save to a temporary latest model path that trainer.py can use.
    filepath = os.path.join(config.get_current_model_dir(), f"checkpoint_candidate.pth.tar")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Support both raw torch models and our NNetWrapper
    if hasattr(model, 'state_dict'):
        state = {'state_dict': model.state_dict()}
    elif hasattr(model, 'nnet'):
        state = {'state_dict': model.nnet.state_dict()}
    else:
        state = model # fallback assuming it's already a dict
        
    torch.save(state, filepath)
    return filepath

def promote_best_model():
    """
    Replace:

    latest.pth
          |
          v
    best.pth

    Then:
        increase version
    """
    # 1. Increase the version
    new_version = increase_version()
    
    candidate_path = os.path.join(config.get_current_model_dir(), f"checkpoint_candidate.pth.tar")
    new_latest_path = get_latest_model_path(new_version)
    best_path = get_best_model_path()
    
    if os.path.exists(candidate_path):
        os.makedirs(os.path.dirname(best_path), exist_ok=True)
        # 2. Save it as the new latest checkpoint
        shutil.copyfile(candidate_path, new_latest_path)
        # 3. Promote it to best
        shutil.copyfile(candidate_path, best_path)

def load_model(filepath):
    """
    Load PyTorch model state_dict.
    """
    if not os.path.exists(filepath):
        return None
        
    state_dict = torch.load(filepath, map_location='cpu', weights_only=False)
    if 'state_dict' in state_dict:
        return state_dict['state_dict']
    return state_dict