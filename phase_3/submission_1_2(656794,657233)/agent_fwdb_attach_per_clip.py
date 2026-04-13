import os
import collections
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = 5
N_FRAMES = 4
OBS_RAW = 18
OBS_AUG = OBS_RAW + 1

_STATE = None


def _load_once():
    global _STATE
    if _STATE is not None:
        return

    import torch
    import torch.nn as nn

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "best_model_ep450.pth")

    class DDQN(nn.Module):
        def __init__(self, in_dim=OBS_AUG * N_FRAMES, n_actions=N_ACTIONS):
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

    class AttachmentPredictor(nn.Module):
        def __init__(self, input_dim=OBS_RAW + N_ACTIONS, hidden_dim=64):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, 1)

        def predict_step(self, x_step, h):
            out, h_new = self.gru(x_step.unsqueeze(1), h)
            logit = self.head(out.squeeze(1))
            return logit.squeeze(-1), h_new

    q = DDQN()
    predictor = AttachmentPredictor()

    ckpt = torch.load(wpath, map_location="cpu")
    q.load_state_dict(ckpt['q'])
    predictor.load_state_dict(ckpt['predictor'])
    q.eval()
    predictor.eval()

    _STATE = {
        'q': q,
        'predictor': predictor,
        'frames': collections.deque(maxlen=N_FRAMES),
        'h': torch.zeros(1, 1, 64),
        'pred_attached': 0.0,
        'prev_obs': None,
        'prev_action': None,
        'last_rng_id': None,
        'torch': torch,
    }


def _reset_episode(obs):
    torch = _STATE['torch']
    aug = np.concatenate([obs, [0.0]])
    _STATE['frames'].clear()
    for _ in range(N_FRAMES):
        _STATE['frames'].append(aug.copy())
    _STATE['h'] = torch.zeros(1, 1, 64)
    _STATE['pred_attached'] = 0.0
    _STATE['prev_obs'] = obs.copy()
    _STATE['prev_action'] = None


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()

    torch = _STATE['torch']
    q = _STATE['q']
    predictor = _STATE['predictor']
    frames = _STATE['frames']

    rng_id = id(rng)
    if _STATE['last_rng_id'] != rng_id:
        _STATE['last_rng_id'] = rng_id
        _reset_episode(obs)
    else:
        if _STATE['prev_action'] is not None:
            act_onehot = np.zeros(N_ACTIONS, dtype=np.float32)
            act_onehot[_STATE['prev_action']] = 1.0
            gru_in = np.concatenate([obs.astype(np.float32), act_onehot])

            with torch.no_grad():
                gru_t = torch.tensor(gru_in, dtype=torch.float32).unsqueeze(0)
                logit, h_new = predictor.predict_step(gru_t, _STATE['h'])
                _STATE['h'] = h_new
                if torch.sigmoid(logit).item() > 0.5:
                    _STATE['pred_attached'] = 1.0

        aug = np.concatenate([obs, [_STATE['pred_attached']]])
        frames.append(aug.copy())

    stacked = np.concatenate(list(frames), axis=0)

    with torch.no_grad():
        x = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)
        qs = q(x).squeeze(0).numpy()

    a = int(np.argmax(qs))

    _STATE['prev_obs'] = obs.copy()
    _STATE['prev_action'] = a

    return ACTIONS[a]