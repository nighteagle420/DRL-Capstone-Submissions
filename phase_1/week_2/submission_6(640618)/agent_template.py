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

Q = None  # stores the loaded model


def _load_once():
    """Load the trained model and weights."""
    global Q
    if Q is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "best_q_table2500.npy")

    

    Q = np.load(wpath)

def obs_to_state(obs):
    s = 0
    for bit in obs:
        s = (s << 1) | int(bit)
    return s


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Use the trained model to choose the best action."""
    _load_once()

    
    

    return ACTIONS[int(np.argmax(Q[obs_to_state(obs)]))]
