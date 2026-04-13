import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = 5
OBS_RAW = 18
PREV_ACT_DIM = 5
OBS_DIM = OBS_RAW + PREV_ACT_DIM

_MODEL = None
_HIDDEN = None
_PREV_ACTION = -1
_LAST_RNG_ID = None


def _load_once():
    global _MODEL
    if _MODEL is not None:
        return

    import torch
    import torch.nn as nn

    class DuelingRecurrentDDQN(nn.Module):
        def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=256):
            super().__init__()
            self.hidden_size = hidden
            self.fc_in = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
            )
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.val_stream = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            self.adv_stream = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_actions),
            )

        def forward(self, x, h0=None):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            B, T, _ = x.shape
            if h0 is None:
                h0 = torch.zeros(1, B, self.hidden_size, device=x.device)
            feat = self.fc_in(x)
            gru_out, h_n = self.gru(feat, h0)
            val = self.val_stream(gru_out)
            adv = self.adv_stream(gru_out)
            q = val + adv - adv.mean(dim=-1, keepdim=True)
            return q, h_n

        def init_hidden(self, batch_size=1, device="cpu"):
            return torch.zeros(1, batch_size, self.hidden_size, device=device)

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "best_weights.pth")

    model = DuelingRecurrentDDQN()
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()
    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _HIDDEN, _PREV_ACTION, _LAST_RNG_ID

    _load_once()
    import torch

    if id(rng) != _LAST_RNG_ID:
        _HIDDEN = _MODEL.init_hidden(1, "cpu")
        _PREV_ACTION = -1
        _LAST_RNG_ID = id(rng)

    one_hot = np.zeros(PREV_ACT_DIM, dtype=np.float32)
    if _PREV_ACTION >= 0:
        one_hot[_PREV_ACTION] = 1.0
    aug = np.concatenate([obs.astype(np.float32), one_hot])

    with torch.no_grad():
        x = torch.tensor(aug, dtype=torch.float32).reshape(1, 1, -1)
        q_vals, _HIDDEN = _MODEL(x, _HIDDEN)
        a = int(q_vals.squeeze(0).squeeze(0).argmax().item())

    _PREV_ACTION = a

    return ACTIONS[a]