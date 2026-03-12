import os
import random
import json
import logging
import numpy as np
import torch
import yaml
from pathlib import Path


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: str, name: str = "train"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"{name}.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(name)


def save_json(data: dict, path: str):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def ensure_dirs(cfg: dict):
    for key in ["checkpoints", "csv_dir", "json_dir", "plots_dir"]:
        Path(cfg["output"][key]).mkdir(parents=True, exist_ok=True)
    for sub in ["robustness", "attribution", "comparison", "extensions"]:
        Path(os.path.join(cfg["output"]["plots_dir"], sub)).mkdir(parents=True, exist_ok=True)


def save_checkpoint(model, optimizer, epoch, path, extra=None):
    state = {
        "epoch": epoch,
        "state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(model, path, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    if hasattr(model, "module"):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)
    return ckpt