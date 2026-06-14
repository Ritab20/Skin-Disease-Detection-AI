# BEST MODEL PATH CONFIGURATION FILE

import os
import torch


# -----------------------------
# MODEL PATH
# -----------------------------
BEST_MODEL_PATH = "best_model.pth"


# -----------------------------
# DEVICE AUTO-DETECTION
# -----------------------------
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -----------------------------
# SAFE LOAD FUNCTION
# -----------------------------
def load_best_model(model):

    if not os.path.exists(BEST_MODEL_PATH):

        raise FileNotFoundError(
            f"Model file not found at {BEST_MODEL_PATH}"
        )

    model.load_state_dict(
        torch.load(
            BEST_MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)

    model.eval()

    return model