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
from collections import deque

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

NUM_FRAMES = 4
OBS_DIM = 18
_LAST_RNG = None
STACKED_DIM = OBS_DIM * NUM_FRAMES

_MODEL = None  # stores the loaded model
_FRAME_STACK = None


def _load_once():
    """Load the trained model and weights."""
    global _MODEL,_FRAME_STACK
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "best_model_ep4050.pth")

    import torch
    import torch.nn as nn

    class _FrameStack:
        def __init__(self, k=NUM_FRAMES, obs_dim=OBS_DIM):
            self.k = k
            self.obs_dim = obs_dim
            self.frames = deque(maxlen=k)
            self._initialized = False
    
        def reset(self):
            self.frames.clear()
            self._initialized = False
    
        def push(self, obs):
            if not self._initialized:
                for _ in range(self.k):
                    self.frames.append(obs.copy())
                self._initialized = True
            else:
                self.frames.append(obs.copy())
            return np.concatenate(list(self.frames), axis=0)

    class DDQN(nn.Module):
        def __init__(self, in_dim=STACKED_DIM, n_actions=5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, n_actions),
            )
 
        def forward(self, x):
            return self.net(x)

    model = DDQN()
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()

    _MODEL = model
    _FRAME_STACK = _FrameStack()


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Use the trained model to choose the best action."""
    
    _load_once()

    import torch

    global _FRAME_STACK, _LAST_RNG
    if rng is not _LAST_RNG:
        _FRAME_STACK.reset()
        _LAST_RNG = rng

    stacked = _FRAME_STACK.push(obs)
    x = torch.from_numpy(stacked.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        logits = _MODEL(x).squeeze(0).numpy()

    return ACTIONS[int(np.argmax(logits))]
