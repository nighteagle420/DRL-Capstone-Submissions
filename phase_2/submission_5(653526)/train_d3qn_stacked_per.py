from __future__ import annotations
import argparse, random, csv, os
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

save_dir = ".checkpoints"
os.makedirs(save_dir, exist_ok=True)

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
EXPLORE_PROBS = np.array([0.10, 0.15, 0.50, 0.15, 0.10], dtype=np.float64)
NUM_FRAMES = 4
OBS_DIM = 18
STACKED_DIM = OBS_DIM * NUM_FRAMES


class FrameStack:
    def __init__(self, k=NUM_FRAMES, obs_dim=OBS_DIM):
        self.k = k
        self.obs_dim = obs_dim
        self.frames = deque(maxlen=k)

    def reset(self, obs):
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs.copy())
        return self._get()

    def push(self, obs):
        self.frames.append(obs.copy())
        return self._get()

    def _get(self):
        return np.concatenate(list(self.frames), axis=0)


class DuelingDDQN(nn.Module):
    def __init__(self, in_dim=STACKED_DIM, n_actions=5, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        feat = self.feature(x)
        val = self.value_stream(feat)
        adv = self.advantage_stream(feat)
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class PrioritizedReplay:
    def __init__(self, cap: int = 100_000, alpha: float = 0.6):
        self.cap = cap
        self.alpha = alpha
        self.buf: list[Transition] = []
        self.priorities: np.ndarray = np.zeros(cap, dtype=np.float64)
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

    def add(self, t: Transition):
        if self.size < self.cap:
            self.buf.append(t)
        else:
            self.buf[self.pos] = t
        self.priorities[self.pos] = self.max_priority ** self.alpha
        self.pos = (self.pos + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch: int, beta: float = 0.4):
        probs = self.priorities[:self.size]
        probs = probs / probs.sum()

        idx = np.random.choice(self.size, size=batch, replace=False, p=probs)
        items = [self.buf[i] for i in idx]

        weights = (self.size * probs[idx]) ** (-beta)
        weights = weights / weights.max()

        s = np.stack([it.s for it in items]).astype(np.float32)
        a = np.array([it.a for it in items], dtype=np.int64)
        r = np.array([it.r for it in items], dtype=np.float32)
        s2 = np.stack([it.s2 for it in items]).astype(np.float32)
        d = np.array([it.done for it in items], dtype=np.float32)

        return s, a, r, s2, d, idx, weights.astype(np.float32)

    def update_priorities(self, idx, td_errors):
        for i, td in zip(idx, td_errors):
            self.priorities[i] = (abs(td) + 1e-5) ** self.alpha
            self.max_priority = max(self.max_priority, self.priorities[i])

    def __len__(self):
        return self.size


def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def evaluate_policy(q, OBELIX, args, device, base_seed, episodes=10):
    q.eval()
    returns = []

    for i in range(episodes):
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=2000,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=i,
        )

        obs = env.reset(seed=i)
        stack = FrameStack()
        s = stack.reset(obs)

        done = False
        ep_ret = 0

        while not done:
            with torch.no_grad():
                st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                qs = q(st)
                a = int(torch.argmax(qs, dim=1).item())

            obs, r, done = env.step(ACTIONS[a], render=False)
            s = stack.push(obs)
            ep_ret += r

        returns.append(ep_ret)

    q.train()
    return np.mean(returns), np.std(returns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="d3qn_weights.pth")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=2)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--replay", type=int, default=200000)
    ap.add_argument("--warmup", type=int, default=10000)
    ap.add_argument("--tau", type=float, default=0.005)

    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int, default=800000)

    ap.add_argument("--per_alpha", type=float, default=0.6)
    ap.add_argument("--per_beta_start", type=float, default=0.4)
    ap.add_argument("--per_beta_end", type=float, default=1.0)

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.wall_obstacles = True
    print(f"WALL OBSTACLES - {args.wall_obstacles}, DIFFICULTY - {args.difficulty}")
    print(f"FRAME STACK - {NUM_FRAMES}, INPUT DIM - {STACKED_DIM}")
    print(f"ARCHITECTURE - Dueling Double DQN (D3QN)")
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

    q = DuelingDDQN().to(device)
    tgt = DuelingDDQN().to(device)
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = PrioritizedReplay(args.replay, alpha=args.per_alpha)

    steps = 0
    total_steps_estimate = args.episodes * args.max_steps

    log_file = "training_log.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "epsilon", "replay_size", "steps"])

    with open("eval_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return_mean", "return_std"])

    def eps_by_step(t):
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    def beta_by_step(t):
        frac = min(1.0, t / total_steps_estimate)
        return args.per_beta_start + frac * (args.per_beta_end - args.per_beta_start)

    best_eval_score = -float("inf")

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

        raw_obs = env.reset(seed=args.seed + ep)
        stack = FrameStack()
        s = stack.reset(raw_obs)

        ep_ret = 0.0

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)

            if np.random.rand() < eps:
                a = int(np.random.choice(len(ACTIONS), p=EXPLORE_PROBS))
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0))
                a = int(torch.argmax(qs, dim=1).item())

            raw_obs, r, done = env.step(ACTIONS[a], render=False)
            s2 = stack.push(raw_obs)
            ep_ret += float(r)

            replay.add(Transition(s=s, a=a, r=r, s2=s2, done=bool(done)))
            s = s2
            steps += 1

            if len(replay) >= max(args.warmup, args.batch):
                beta = beta_by_step(steps)
                sb, ab, rb, s2b, db, idx, weights = replay.sample(args.batch, beta=beta)

                sb_t = torch.tensor(sb, device=device)
                ab_t = torch.tensor(ab, device=device)
                rb_t = torch.tensor(rb, device=device)
                s2b_t = torch.tensor(s2b, device=device)
                db_t = torch.tensor(db, device=device)
                w_t = torch.tensor(weights, device=device)

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

                replay.update_priorities(idx, td_errors)

                for tp, qp in zip(tgt.parameters(), q.parameters()):
                    tp.data.copy_(args.tau * qp.data + (1.0 - args.tau) * tp.data)

            if done:
                break

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep + 1, ep_ret, eps_by_step(steps), len(replay), steps])

        if (ep + 1) % 500 == 0:
            ckpt_path = f"checkpoint_ep{ep+1}.pth"
            torch.save(q.state_dict(), os.path.join(save_dir, ckpt_path))

        if (ep + 1) % 50 == 0:
            tqdm.write(f"Episode {ep+1}/{args.episodes} return={ep_ret:.1f}")

        if (ep + 1) % 50 == 0:
            eval_mean_score, eval_std_score = evaluate_policy(q, OBELIX, args, device, args.seed)
            tqdm.write(f"Eval @ episode {ep+1}: mean_return={eval_mean_score:.2f}, std_return={eval_std_score:.2f}")

            

            if eval_mean_score > best_eval_score:
                best_eval_score = eval_mean_score
                torch.save(q.state_dict(), os.path.join(save_dir, f"best_model_ep{ep+1}.pth"))
                tqdm.write(f"BEST model (eval) saved at ep {ep+1}")
                with open("eval_log.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([ep + 1, eval_mean_score, eval_std_score])

    torch.save(q.state_dict(), args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()