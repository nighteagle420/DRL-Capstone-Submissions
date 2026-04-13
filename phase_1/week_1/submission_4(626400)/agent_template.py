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
    wpath = os.path.join(submission_dir, "weights.pth")

    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self, obs_dim=18, n_actions=5, hidden=128):
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU()
            )

            self.lstm = nn.LSTM(
                input_size=hidden,
                hidden_size=hidden,
                batch_first=True
            )

            self.q_head = nn.Linear(hidden, n_actions)

        def forward(self, obs, hidden=None):

            if obs.dim() == 2:
                obs = obs.unsqueeze(1)

            x = self.encoder(obs)
            x, hidden = self.lstm(x, hidden)
            q_values = self.q_head(x)

            return q_values[:, -1]

    model = Net()
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()

    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Use the trained model to choose the best action."""
    _load_once()

    import torch
    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        logits = _MODEL(x).squeeze(0).numpy()

    return ACTIONS[int(np.argmax(logits))]
