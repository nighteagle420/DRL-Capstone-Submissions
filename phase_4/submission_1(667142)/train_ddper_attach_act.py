from __future__ import annotations
import argparse, random, csv, os
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

save_dir = ".checkpoints"
os.makedirs(save_dir, exist_ok=True)

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)
N_FRAMES = 4
OBS_RAW = 18
OBS_AUG = OBS_RAW + 1 + N_ACTIONS
OBS_STACKED = OBS_AUG * N_FRAMES


class DDQN(nn.Module):
    def __init__(self, in_dim=OBS_STACKED, n_actions=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class AttachmentPredictor(nn.Module):
    def __init__(self, input_dim=OBS_RAW + N_ACTIONS, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, h=None):
        out, h_new = self.gru(x, h)
        logits = self.head(out)
        return logits.squeeze(-1), h_new

    def predict_step(self, x_step, h):
        out, h_new = self.gru(x_step.unsqueeze(1), h)
        logit = self.head(out.squeeze(1))
        return logit.squeeze(-1), h_new


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class PrioritizedReplay:
    def __init__(self, cap=500000, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_steps=3000000):
        self.cap = cap
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.eps = 1e-6

        self.buf = []
        self.priorities = np.zeros(cap, dtype=np.float64)
        self.write_idx = 0
        self.size = 0
        self.max_priority = 1.0
        self._step = 0

    def add(self, t: Transition):
        if self.size < self.cap:
            self.buf.append(t)
        else:
            self.buf[self.write_idx] = t
        self.priorities[self.write_idx] = self.max_priority ** self.alpha
        self.write_idx = (self.write_idx + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch: int):
        self._step += 1
        beta = min(self.beta_end, self.beta_start + (self.beta_end - self.beta_start) * self._step / self.beta_steps)

        prios = self.priorities[:self.size]
        probs = prios / prios.sum()

        indices = np.random.choice(self.size, size=batch, replace=False, p=probs)

        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        weights = weights.astype(np.float32)

        items = [self.buf[i] for i in indices]
        s = np.stack([it.s for it in items]).astype(np.float32)
        a = np.array([it.a for it in items], dtype=np.int64)
        r = np.array([it.r for it in items], dtype=np.float32)
        s2 = np.stack([it.s2 for it in items]).astype(np.float32)
        d = np.array([it.done for it in items], dtype=np.float32)

        return s, a, r, s2, d, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + self.eps) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.priorities[idx] = priority

    def __len__(self):
        return self.size


class FrameStack:
    def __init__(self, n_frames=N_FRAMES, obs_dim=OBS_AUG):
        self.n_frames = n_frames
        self.obs_dim = obs_dim
        self.frames = deque(maxlen=n_frames)

    def reset(self, obs):
        for _ in range(self.n_frames):
            self.frames.append(obs.copy())
        return self._get()

    def push(self, obs):
        self.frames.append(obs.copy())
        return self._get()

    def _get(self):
        return np.concatenate(list(self.frames), axis=0)


class TrajectoryBuffer:
    def __init__(self, max_episodes=500):
        self.max_episodes = max_episodes
        self.episodes: Deque[List[Tuple[np.ndarray, int, int]]] = deque(maxlen=max_episodes)

    def add_episode(self, episode_data: List[Tuple[np.ndarray, int, int]]):
        if len(episode_data) > 1:
            self.episodes.append(episode_data)

    def sample_batch(self, batch_size):
        if len(self.episodes) == 0:
            return None
        indices = np.random.choice(len(self.episodes), size=min(batch_size, len(self.episodes)), replace=False)

        inputs_list = []
        labels_list = []
        lengths = []

        for idx in indices:
            ep = self.episodes[idx]

            inp_seq = []
            lab_seq = []
            for obs_raw, act_idx, attached in ep:
                act_onehot = np.zeros(N_ACTIONS, dtype=np.float32)
                act_onehot[act_idx] = 1.0
                inp_seq.append(np.concatenate([obs_raw, act_onehot]))
                lab_seq.append(float(attached))

            inputs_list.append(np.array(inp_seq, dtype=np.float32))
            labels_list.append(np.array(lab_seq, dtype=np.float32))
            lengths.append(len(ep))

        max_len = max(lengths)
        input_dim = OBS_RAW + N_ACTIONS
        batch_in = np.zeros((len(indices), max_len, input_dim), dtype=np.float32)
        batch_lab = np.zeros((len(indices), max_len), dtype=np.float32)
        mask = np.zeros((len(indices), max_len), dtype=np.float32)

        for i, (inp, lab, l) in enumerate(zip(inputs_list, labels_list, lengths)):
            batch_in[i, :l] = inp
            batch_lab[i, :l] = lab
            mask[i, :l] = 1.0

        return (
            torch.tensor(batch_in),
            torch.tensor(batch_lab),
            torch.tensor(mask),
        )

    def __len__(self):
        return len(self.episodes)


def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def augment_obs(obs_raw, attached_bit, prev_action_onehot):
    return np.concatenate([obs_raw, [float(attached_bit)], prev_action_onehot])


# def evaluate_policy(q, predictor, OBELIX, args, device, base_seed, episodes=10):
#     q.eval()
#     predictor.eval()
#     returns = []

#     for i in range(episodes):
#         env = OBELIX(
#             scaling_factor=args.scaling_factor,
#             arena_size=args.arena_size,
#             max_steps=2000,
#             wall_obstacles=args.wall_obstacles,
#             difficulty=args.difficulty,
#             box_speed=args.box_speed,
#             seed=i,
#         )
#         obs_raw = env.reset(seed=i)
#         done = False
#         ep_ret = 0.0

#         h = torch.zeros(1, 1, 64, device=device)
#         pred_attached = 0.0

#         obs_aug = augment_obs(obs_raw, pred_attached)
#         stacker = FrameStack(N_FRAMES, OBS_AUG)
#         s = stacker.reset(obs_aug)

#         while not done:
#             with torch.no_grad():
#                 st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
#                 qs = q(st)
#                 a = int(torch.argmax(qs, dim=1).item())

#             obs_raw, r, done = env.step(ACTIONS[a], render=False)
#             ep_ret += r

#             with torch.no_grad():
#                 act_onehot = np.zeros(N_ACTIONS, dtype=np.float32)
#                 act_onehot[a] = 1.0
#                 gru_in = np.concatenate([obs_raw, act_onehot])
#                 gru_t = torch.tensor(gru_in, dtype=torch.float32, device=device).unsqueeze(0)
#                 logit, h = predictor.predict_step(gru_t, h)
#                 pred_attached = float(torch.sigmoid(logit).item() > 0.5)

#             obs_aug = augment_obs(obs_raw, pred_attached)
#             s = stacker.push(obs_aug)

#         returns.append(ep_ret)

#     q.train()
#     predictor.train()
#     return np.mean(returns), np.std(returns)

def evaluate_policy(q, predictor, OBELIX, args, device, base_seed, episodes=10):
    q.eval()
    predictor.eval()

    def run_episodes(wall_obstacles):
        returns = []
        for i in range(episodes):
            env = OBELIX(
                scaling_factor=args.scaling_factor,
                arena_size=args.arena_size,
                max_steps=2000,
                wall_obstacles=wall_obstacles,
                difficulty=args.difficulty,
                box_speed=args.box_speed,
                seed=i,
            )
            obs_raw = env.reset(seed=i)
            done = False
            ep_ret = 0.0

            h = torch.zeros(1, 1, 64, device=device)
            pred_attached = 0.0
            prev_action_onehot = np.zeros(N_ACTIONS, dtype=np.float32)

            obs_aug = augment_obs(obs_raw, pred_attached, prev_action_onehot)
            stacker = FrameStack(N_FRAMES, OBS_AUG)
            s = stacker.reset(obs_aug)

            while not done:
                with torch.no_grad():
                    st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    qs = q(st)
                    a = int(torch.argmax(qs, dim=1).item())

                obs_raw, r, done = env.step(ACTIONS[a], render=False)
                ep_ret += r

                with torch.no_grad():
                    act_onehot = np.zeros(N_ACTIONS, dtype=np.float32)
                    act_onehot[a] = 1.0
                    gru_in = np.concatenate([obs_raw, act_onehot])
                    gru_t = torch.tensor(gru_in, dtype=torch.float32, device=device).unsqueeze(0)
                    logit, h = predictor.predict_step(gru_t, h)
                    pred_attached = float(torch.sigmoid(logit).item() > 0.5)

                prev_action_onehot = np.zeros(N_ACTIONS, dtype=np.float32)
                prev_action_onehot[a] = 1.0
                obs_aug = augment_obs(obs_raw, pred_attached, prev_action_onehot)
                s = stacker.push(obs_aug)

            returns.append(ep_ret)
        return np.mean(returns), np.std(returns)

    mean_wall, std_wall = run_episodes(wall_obstacles=True)
    mean_nowall, std_nowall = run_episodes(wall_obstacles=False)

    weighted_mean = 0.6 * mean_wall + 0.4 * mean_nowall
    weighted_std = 0.6 * std_wall + 0.4 * std_nowall

    q.train()
    predictor.train()
    return weighted_mean, weighted_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="ddqn_attach_weights.pth")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pred_lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--replay", type=int, default=500000)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--tau", type=float, default=0.005)

    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.02)
    ap.add_argument("--eps_decay_steps", type=int, default=1500000)

    ap.add_argument("--pred_train_interval", type=int, default=10)
    ap.add_argument("--pred_batch", type=int, default=8)
    ap.add_argument("--pred_epochs", type=int, default=4)
    ap.add_argument("--traj_buffer_size", type=int, default=500)

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.wall_obstacles = True

    print(f"WALL OBSTACLES - {args.wall_obstacles}, DIFFICULTY - {args.difficulty}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    OBELIX = import_obelix(args.obelix_py)

    q = DDQN().to(device)
    tgt = DDQN().to(device)
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    predictor = AttachmentPredictor().to(device)

    opt = optim.Adam(q.parameters(), lr=args.lr)
    pred_opt = optim.Adam(predictor.parameters(), lr=args.pred_lr)
    bce = nn.BCEWithLogitsLoss(reduction='none')

    replay = PrioritizedReplay(cap=args.replay)
    traj_buf = TrajectoryBuffer(args.traj_buffer_size)

    steps = 0

    log_file = "training_log_attach.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "epsilon", "replay_size", "steps", "pred_loss"])
    with open('eval_log.csv', "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return_mean", "return_std"])

    def eps_by_step(t):
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    best_eval_score = -float("inf")
    latest_pred_loss = 0.0

    for ep in tqdm(range(args.episodes), leave=True):
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed + ep,
        )

        obs_raw = env.reset(seed=args.seed + ep)
        attached = 0
        prev_action_onehot = np.zeros(N_ACTIONS, dtype=np.float32)
        obs_aug = augment_obs(obs_raw, attached, prev_action_onehot)
        stacker = FrameStack(N_FRAMES, OBS_AUG)
        s = stacker.reset(obs_aug)

        ep_ret = 0.0
        episode_traj = []

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)

            if np.random.rand() < eps:
                if steps < 100000:
                    probs = np.array([0.1, 0.1, 0.6, 0.1, 0.1])
                else:
                    probs = np.array([0.15, 0.2, 0.3, 0.2, 0.15])
                a = int(np.random.choice(N_ACTIONS, p=probs))
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0))
                a = int(torch.argmax(qs, dim=1).item())

            obs_raw_next, r, done = env.step(ACTIONS[a], render=False)
            ep_ret += float(r)

            if r >= 99.0 and attached == 0:
                attached = 1

            episode_traj.append((obs_raw.copy(), a, attached))

            prev_action_onehot = np.zeros(N_ACTIONS, dtype=np.float32)
            prev_action_onehot[a] = 1.0
            obs_aug_next = augment_obs(obs_raw_next, attached, prev_action_onehot)
            s2 = stacker.push(obs_aug_next)

            replay.add(Transition(s=s, a=a, r=max(r, -10.0), s2=s2, done=bool(done)))
            s = s2
            obs_raw = obs_raw_next
            steps += 1

            if len(replay) >= max(args.warmup, args.batch):
                sb, ab, rb, s2b, db, per_indices, per_weights = replay.sample(args.batch)
                sb_t = torch.tensor(sb, device=device)
                ab_t = torch.tensor(ab, device=device)
                rb_t = torch.tensor(rb, device=device)
                s2b_t = torch.tensor(s2b, device=device)
                db_t = torch.tensor(db, device=device)
                w_t = torch.tensor(per_weights, device=device)

                with torch.no_grad():
                    next_q = q(s2b_t)
                    next_a = torch.argmax(next_q, dim=1)
                    next_q_tgt = tgt(s2b_t)
                    next_val = next_q_tgt.gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y = rb_t + args.gamma * (1.0 - db_t) * next_val

                pred = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                td_errors = (pred - y).detach().cpu().numpy()
                loss = (w_t * nn.functional.smooth_l1_loss(pred, y, reduction='none')).mean()

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                replay.update_priorities(per_indices, td_errors)

                for p, tp in zip(q.parameters(), tgt.parameters()):
                    tp.data.copy_(args.tau * p.data + (1.0 - args.tau) * tp.data)

            if done:
                break

        traj_buf.add_episode(episode_traj)

        if (ep + 1) % args.pred_train_interval == 0 and len(traj_buf) >= args.pred_batch:
            total_pred_loss = 0.0
            n_batches = 0
            for _ in range(args.pred_epochs):
                batch_data = traj_buf.sample_batch(args.pred_batch)
                if batch_data is None:
                    break
                batch_in, batch_lab, mask = [x.to(device) for x in batch_data]
                logits, _ = predictor(batch_in)
                raw_loss = bce(logits, batch_lab)
                masked_loss = (raw_loss * mask).sum() / mask.sum().clamp(min=1.0)

                pred_opt.zero_grad()
                masked_loss.backward()
                nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
                pred_opt.step()

                total_pred_loss += masked_loss.item()
                n_batches += 1

            if n_batches > 0:
                latest_pred_loss = total_pred_loss / n_batches

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep + 1, ep_ret, eps_by_step(steps), len(replay), steps, f"{latest_pred_loss:.4f}"])

        if (ep + 1) % 500 == 0:
            ckpt_path = f"checkpoint_ep{ep+1}.pth"
            torch.save({
                'q': q.state_dict(),
                'predictor': predictor.state_dict(),
            }, os.path.join(save_dir, ckpt_path))

        if (ep + 1) % 50 == 0:
            tqdm.write(f"Episode {ep+1}/{args.episodes} return={ep_ret:.1f} pred_loss={latest_pred_loss:.4f}")

        if (ep + 1) % 50 == 0:
            eval_mean, eval_std = evaluate_policy(q, predictor, OBELIX, args, device, args.seed)
            tqdm.write(f"Eval @ episode {ep+1}: mean={eval_mean:.2f}, std={eval_std:.2f}")
            with open('eval_log.csv', "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ep + 1, eval_mean,eval_std])

            if eval_mean > best_eval_score:
                best_eval_score = eval_mean
                torch.save({
                    'q': q.state_dict(),
                    'predictor': predictor.state_dict(),
                }, os.path.join(save_dir, f"best_model_ep{ep+1}.pth"))
                tqdm.write(f"BEST model (eval) saved")

    torch.save({
        'q': q.state_dict(),
        'predictor': predictor.state_dict(),
    }, args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()