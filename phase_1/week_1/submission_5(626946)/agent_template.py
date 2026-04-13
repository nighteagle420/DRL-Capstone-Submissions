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
        def __init__(self,input_dim=18,output_dim=5,hidden_dims=[64,64],activation=nn.ReLU):
            super(Net,self).__init__()
            self.activation = activation()

            self.input_layer = nn.Linear(input_dim,hidden_dims[0])

            self.hidden_layers = nn.ModuleList()
            for i in range(len(hidden_dims)-1):
                hidden_layer = nn.Linear(hidden_dims[i],hidden_dims[i+1])
                self.hidden_layers.append(hidden_layer)
            self.value_output = nn.Linear(hidden_dims[-1],1)
            self.advantage_output = nn.Linear(hidden_dims[-1],output_dim)

        def forward(self,state):
            if isinstance(state, tuple):
                state = state[0]

            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)

            if state.dim() == 1:
                state = state.unsqueeze(0)
            x = state
            x = self.activation(self.input_layer(x))
            for hidden_layer in self.hidden_layers:
                x = self.activation(hidden_layer(x))

            a = self.advantage_output(x)
            v = self.value_output(x)
            v = v.expand_as(a)

            q = v + a - a.mean(1,keepdim=True).expand_as(a)
            return q

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
