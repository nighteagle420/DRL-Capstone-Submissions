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

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

class ActorCritic(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
        )
        self.policy = nn.Linear(64, n_actions)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy(x), self.value(x)


# =========================
# UTILS
# =========================
def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

def evaluate_policy(model, OBELIX, args, device, base_seed, episodes=10):
    model.eval()
    # attach_net.eval()

    returns = []

    for i in range(episodes):
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=base_seed + 1000 + i,
        )

        obs = env.reset(seed=base_seed + 1000 + i)

        done = False
        ep_ret = 0

        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = model(obs_t)
                a = torch.argmax(logits, dim=1).item()

            obs, r, done = env.step(ACTIONS[a], render=False)
            ep_ret += r

        returns.append(ep_ret)

    model.train()

    return np.mean(returns)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="ppoweights.pth")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--replay", type=int, default=100000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--target_sync", type=int, default=2000)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)

    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int, default=500000)

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.wall_obstacles = True

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
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using {device}")

    OBELIX = import_obelix(args.obelix_py)

    model = ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    entropy_coef = 0.02
    value_coef = 0.5

    

    steps = 0

    log_file = "training_log.csv"
    # with open(log_file, "a", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["episode", "return", "epsilon", "replay_size", "steps"])

    # file_exists = os.path.isfile(log_file)

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        

        writer.writerow(["episode", "return", "steps"])
    # def eps_by_step(t):
    #     if t >= args.eps_decay_steps:
    #         return args.eps_end
    #     frac = t / args.eps_decay_steps
    #     return args.eps_start + frac * (args.eps_end - args.eps_start)
    
    best_eval_score = -float("inf")

    for ep in tqdm(range(args.episodes),leave=True):
        # ATTACHED = 0
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed + ep,
        )

        obs = env.reset(seed=args.seed + ep)
        
        states, actions, rewards, log_probs = [], [], [], []
        done = False
        ep_ret = 0.0

        while not done:
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, value = model(state)
                probs = torch.softmax(logits, dim=-1)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            obs2, r, done = env.step(ACTIONS[action.item()], render=False)

            r = r / 100.0  # reward scaling

            states.append(obs)
            actions.append(action.item())
            rewards.append(r)
            log_probs.append(log_prob.item())

            obs = obs2
            ep_ret += float(r)
            steps+=1
        returns = compute_returns(rewards, args.gamma)
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions).to(device)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        with torch.no_grad():
            _, values = model(states)
            values = values.squeeze()

        advantages = (returns - values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(args.epochs):
            logits, values_pred = model(states)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns - values_pred.squeeze()).pow(2).mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep + 1, ep_ret, steps])

        if (ep + 1) % 500 == 0:
            ckpt_path = f"checkpoint_ep{ep+1}.pth"
            # attach_path = f'attach_ep{ep+1}.pth'
            torch.save(model.state_dict(), os.path.join(save_dir,ckpt_path))
            # torch.save(attach_net.state_dict(), os.path.join(save_dir,attach_path))

        if (ep + 1) % 50 == 0:
            tqdm.write(f"Episode {ep+1}/{args.episodes} return={ep_ret:.1f}")

        if (ep + 1) % 50 == 0:
            eval_score = evaluate_policy(model,OBELIX, args, device, args.seed)

            tqdm.write(f"Eval @ episode {ep+1}: mean_return={eval_score:.2f}")

            if eval_score > best_eval_score:
                best_eval_score = eval_score
                torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_ep{ep+1}.pth"))
                # torch.save(attach_net.state_dict(), os.path.join(save_dir,f'best_attach_epi{ep+1}.pth'))
                tqdm.write(f"BEST model (eval) saved")
    torch.save(model.state_dict(), 'final.pth')
    # torch.save(attach_net.state_dict(), 'final_attach.pth')
    print("Saved:", args.out)

if __name__ == "__main__":
    main()