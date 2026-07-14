# distributed/server.py

import os
import shutil
import uuid
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

app = FastAPI(title="AlphaZero Server")

# --------------------------------------------------------
# Configuration
# --------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

VERSION_FILE = ROOT / "version.txt"
REPLAY_DIR     = ROOT / "storage" / "replay"
REPLAY_INCOMING = REPLAY_DIR / "incoming"

# Ensure all pipeline folders exist on startup
for _d in ["incoming", "ready", "training", "used", "merged"]:
    (REPLAY_DIR / _d).mkdir(parents=True, exist_ok=True)

@app.on_event("startup")
def startup_event():
    # Initialize a random model on startup if no model exists to prevent worker deadlock
    info = get_version_info()
    run_dir = ROOT / "logs" / info["run"]
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_0 = run_dir / "checkpoint_0.pth.tar"
    best_model = run_dir / "best.pth.tar"

    if not checkpoint_0.exists() or not best_model.exists():
        print("No initial model checkpoints found. Generating initial random model...")
        from game import DotsAndBoxesGame
        from model import NNetWrapper
        from model import dotdict
        import torch
        
        nnet_args = dotdict({
            'lr': 0.0005,
            'epochs': 100,
            'batch_size': 512,
            'num_channels': 256,
            'num_res_blocks': 10,
            'device': 'cpu'
        })
        
        dummy_game = DotsAndBoxesGame(size=5)
        nnet = NNetWrapper(dummy_game, nnet_args)
        state = {'state_dict': nnet.nnet.state_dict()}
        
        torch.save(state, checkpoint_0)
        torch.save(state['state_dict'], best_model)
        
        # Ensure version.txt has last_updated_model: 0
        if VERSION_FILE.exists():
            lines = VERSION_FILE.read_text().splitlines()
            new_lines = []
            for line in lines:
                if line.startswith("last_updated_model:"):
                    new_lines.append("last_updated_model: 0")
                else:
                    new_lines.append(line)
            VERSION_FILE.write_text("\n".join(new_lines) + "\n")
        else:
            VERSION_FILE.write_text("run: run_1\nlast_updated_model: 0\ncurrent_phase: 0\n")

# --------------------------------------------------------
# Helper
# --------------------------------------------------------

def get_version_info():
    info = {"run": "run_1", "last_updated_model": "0", "current_phase": "0"}
    if VERSION_FILE.exists():
        for line in VERSION_FILE.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
    return info

def current_version():
    return get_version_info()["last_updated_model"]

def get_model_path(model_type="latest"):
    info = get_version_info()
    run_dir = ROOT / "logs" / info["run"]
    
    if model_type == "best":
        return run_dir / "best.pth.tar"
    else:
        checkpoint_id = info["last_updated_model"]
        return run_dir / f"checkpoint_{checkpoint_id}.pth.tar"

# --------------------------------------------------------
# API
# --------------------------------------------------------

@app.get("/")
def home():
    return {
        "server": "AlphaZero Distributed Server",
        "version": current_version()
    }

# --------------------------------------------------------

@app.get("/version")
def version():
    return {
        "last_updated_model": current_version()
    }

# --------------------------------------------------------

@app.get("/latest_model")
def latest_model():
    model_path = get_model_path("latest")
    if not model_path.exists():
        raise HTTPException(404, f"latest model {model_path.name} not found")

    return FileResponse(
        path=model_path,
        filename=model_path.name,
        media_type="application/octet-stream"
    )

# --------------------------------------------------------

@app.get("/best_model")
def best_model():
    model_path = get_model_path("best")
    if not model_path.exists():
        raise HTTPException(404, f"best model {model_path.name} not found")

    return FileResponse(
        path=model_path,
        filename=model_path.name,
        media_type="application/octet-stream"
    )

# --------------------------------------------------------

@app.post("/upload_replay")
async def upload_replay(
    worker: str,
    model_version: int = 0,
    file: UploadFile = File(...)
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{worker}_v{model_version}_{timestamp}.npz"
    save_path = REPLAY_INCOMING / filename

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "status": "ok",
        "file": filename,
        "folder": "incoming"
    }

# --------------------------------------------------------

@app.get("/status")
def status():

    total = 0

    for worker in REPLAY_DIR.iterdir():
        if worker.is_dir():
            total += len(list(worker.glob("*.npz")))

    return {
        "version": current_version(),
        "replay_files": total,
        "latest_model": get_model_path("latest").exists(),
        "best_model": get_model_path("best").exists()
    }

# --------------------------------------------------------

@app.get("/config")
def config():

    return {
        "num_mcts": 400,
        "cpuct": 1.5,
        "temperature": 1.0,
        "dirichlet_alpha": 0.3,
        "dirichlet_eps": 0.25
    }