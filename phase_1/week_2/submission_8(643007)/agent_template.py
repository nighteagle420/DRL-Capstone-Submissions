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

_MODEL = None  
_ATTACH = None


def _load_once():
    """Load the trained model and weights."""
    global _MODEL
    global _ATTACH
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "best_model_ep50.pth")
    apath = os.path.join(submission_dir,'best_attach_epi50.pth')

    import torch
    import torch.nn as nn

    class DDQN(nn.Module):
        def __init__(self, in_dim=19, n_actions=5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Linear(128,128),
                nn.ReLU(),
                nn.Linear(128, n_actions),
            )
        def forward(self, x):
            return self.net(x)
        
    class AttachNet(nn.Module):
        def __init__(self, in_dim=18):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)  # binary output
            )

        def forward(self, x):
            return self.net(x)  # logits

    model = DDQN()
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()

    attach = AttachNet()
    attach.load_state_dict(torch.load(apath,map_location='cpu'))
    attach.eval()

    _MODEL = model
    _ATTACH = attach


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Use the trained model to choose the best action."""
    _load_once()

    # import torch
    # x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    # with torch.no_grad():
    #     logits = _MODEL(x).squeeze(0).numpy()
    import torch

    obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        
        att = torch.sigmoid(_ATTACH(obs_t))  

        
        x = torch.cat([obs_t, att], dim=1)

        
        qvals = _MODEL(x).squeeze(0).numpy()

    return ACTIONS[int(np.argmax(qvals))]
