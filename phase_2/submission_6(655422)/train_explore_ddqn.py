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
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
save_dir = ".checkpoints"
os.makedirs(save_dir,exist_ok=True)

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class DDQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
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


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool

class Replay:
    def __init__(self, cap: int = 100_000):
        self.buf: Deque[Transition] = deque(maxlen=cap)
    def add(self, t: Transition):
        self.buf.append(t)
    def sample(self, batch: int):
        idx = np.random.choice(len(self.buf), size=batch, replace=False)
        items = [self.buf[i] for i in idx]
        s = np.stack([it.s for it in items]).astype(np.float32)
        a = np.array([it.a for it in items], dtype=np.int64)
        r = np.array([it.r for it in items], dtype=np.float32)
        s2 = np.stack([it.s2 for it in items]).astype(np.float32)
        d = np.array([it.done for it in items], dtype=np.float32)
        return s, a, r, s2, d
    def __len__(self): return len(self.buf)

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
        rng = np.random.default_rng(i)

        done = False
        ep_ret = 0

        while not done:
            with torch.no_grad():
                s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

                qs = q(s)
                a = int(torch.argmax(qs, dim=1).item())

            obs, r, done = env.step(ACTIONS[a], render=False)
            ep_ret += r

        returns.append(ep_ret)

    q.train()

    return np.mean(returns),np.std(returns)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="ddqnweights.pth")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--difficulty", type=int, default=2)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=3e-4) #from 5e-4
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--replay", type=int, default=100000)
    ap.add_argument("--warmup", type=int, default=8000) #from 2k
    ap.add_argument("--target_sync", type=int, default=1500) #from 2k

    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int, default=200000)

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.wall_obstacles = True
    print(f"WALL OBSTACLES - {args.wall_obstacles}, DIFFICULTY - {args.difficulty}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using {device}")
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(args.seed)
    #     torch.cuda.manual_seed_all(args.seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using {device}')

    OBELIX = import_obelix(args.obelix_py)

    q = DDQN().to(device)
    tgt = DDQN().to(device)
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    # attach_net = AttachNet().to(device)
    # attach_opt = optim.Adam(attach_net.parameters(), lr=5e-4)
    # bce = nn.BCEWithLogitsLoss()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = Replay(args.replay)

    steps = 0

    log_file = "training_log.csv"
    # with open(log_file, "a", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["episode", "return", "epsilon", "replay_size", "steps"])

    # file_exists = os.path.isfile(log_file)

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "epsilon", "replay_size", "steps"])
    with open('eval_log.csv', "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return_mean", "return_std"])
    def eps_by_step(t):
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)
    
    best_eval_score = -float("inf")

    for ep in tqdm(range(args.episodes),leave=True):
       
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed + ep,
        )

        s = env.reset(seed=args.seed + ep)
       
        ep_ret = 0.0

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)

            if np.random.rand() < eps:
                if steps<100000:
                    probs = np.array([0.1,0.1,0.6,0.1,0.1])
                else:
                    probs = np.array([0.15,0.2,0.3,0.2,0.15])
                a = int(np.random.choice(len(ACTIONS),p=probs))
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0))
                a = int(torch.argmax(qs, dim=1).item())

            s2, r, done = env.step(ACTIONS[a], render=False)
            # if r==100:
            #     ATTACHED=1
            # s2 = np.concatenate((obs2,[ATTACHED]))
            ep_ret += float(r)

            replay.add(Transition(
                s=s, a=a, r=r, s2=s2, done=bool(done),
                
            ))
            s = s2
            
            steps += 1

            if len(replay) >= max(args.warmup, args.batch):
                sb, ab, rb, s2b, db = replay.sample(args.batch)

                sb_t = torch.tensor(sb, device=device)
                ab_t = torch.tensor(ab, device=device)
                rb_t = torch.tensor(rb, device=device)
                s2b_t = torch.tensor(s2b, device=device)
                db_t = torch.tensor(db, device=device)

                with torch.no_grad():
                    next_q = q(s2b_t)
                    next_a = torch.argmax(next_q, dim=1)
                    next_q_tgt = tgt(s2b_t)
                    next_val = next_q_tgt.gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y = rb_t + args.gamma * (1.0 - db_t) * next_val

                pred = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                

                if steps % args.target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

            if done:
                break

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep + 1, ep_ret, eps_by_step(steps), len(replay), steps])

        if (ep + 1) % 500 == 0:
            ckpt_path = f"checkpoint_ep{ep+1}.pth"
        
            torch.save(q.state_dict(), os.path.join(save_dir,ckpt_path))
            

        if (ep + 1) % 50 == 0:
            tqdm.write(f"Episode {ep+1}/{args.episodes} return={ep_ret:.1f}")

        if (ep + 1) % 50 == 0:
            eval_mean_score,eval_std_score = evaluate_policy(q,OBELIX, args, device, args.seed)

            tqdm.write(f"Eval @ episode {ep+1}: mean_return={eval_mean_score:.2f}, std_return = {eval_std_score}")
            with open('eval_log.csv', "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ep+1,eval_mean_score,eval_std_score])

            if eval_mean_score > best_eval_score:
                best_eval_score = eval_mean_score
                torch.save(q.state_dict(), os.path.join(save_dir, f"best_model_ep{ep+1}.pth"))
                
                tqdm.write(f"BEST model (eval) saved")
    torch.save(q.state_dict(), 'final.pth')

    print("Saved:", args.out)

if __name__ == "__main__":
    main()