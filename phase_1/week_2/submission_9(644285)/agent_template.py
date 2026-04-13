"""
Submission template (USES trained weights).

Use this template if your agent depends on a trained neural network.
Place your saved model file (weights.pth) inside the submission folder.

The policy loads the model and uses it to predict the best action
from the observation.

The evaluator will import this file and call `policy(obs, rng)`.
"""

import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None  # stores the loaded model


def _load_once():
    """Load the trained model and weights."""
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "best_model_ep150.pth")

    import torch
    import torch.nn as nn

    class ActorCritic(nn.Module):
        def __init__(self, in_dim=18, n_actions=5):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
            )
            self.policy = nn.Linear(64, n_actions)
            self.value = nn.Linear(64, 1)

        def forward(self, x):
            x = self.shared(x)
            return self.policy(x), self.value(x)

    model = ActorCritic()
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()

    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Use the trained model to choose the best action."""
    _load_once()

    import torch
    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        logits,_ = _MODEL(x)
        logits = logits.squeeze(0).numpy()

    return ACTIONS[int(np.argmax(logits))]
