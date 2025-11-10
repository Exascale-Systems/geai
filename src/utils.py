import torch
from pathlib import Path
import numpy as np
from src.model import GravInvNet

def load_model(model_path, device="cpu"):
    device = torch.device(device)
    model = GravInvNet().to(device)
    if Path(model_path).exists():
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state.get("model", state))
    model.eval()
    return model, device

