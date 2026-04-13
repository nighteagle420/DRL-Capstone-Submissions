from __future__ import annotations
import argparse, random, csv, os
from collections import deque
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5
OBS_RAW = 18
PREV_ACT_DIM = 5
OBS_DIM = OBS_RAW + PREV_ACT_DIM

save_dir = ".newcheckpoints_r2d2_full"
os.makedirs(save_dir, exist_ok=True)


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


@dataclass
class EpisodeData:
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    length: int = 0
    got_attachment: bool = False
    got_success: bool = False


class EpisodeReplay:
    def __init__(self, max_episodes=2000, seq_len=40, burn_in=20):
        self.max_episodes = max_episodes
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.episodes: List[EpisodeData] = []
        self.priorities: List[float] = []
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 1e-5
        self.min_priority = 1e-5

    def add_episode(self, ep: EpisodeData):
        # Need at least seq_len transitions => seq_len + 1 observations
        if ep.length < self.seq_len:
            return

        base_pri = max(self.priorities) if self.priorities else 1.0
        if ep.got_success:
            pri = base_pri * 20.0
        elif ep.got_attachment:
            pri = base_pri * 10.0
        else:
            pri = base_pri

        if len(self.episodes) >= self.max_episodes:
            evict_candidates = [
                i for i, e in enumerate(self.episodes)
                if not e.got_attachment and not e.got_success
            ]
            if evict_candidates:
                idx = min(evict_candidates, key=lambda i: self.priorities[i])
            else:
                idx = int(np.argmin(self.priorities))
            self.episodes[idx] = ep
            self.priorities[idx] = pri
        else:
            self.episodes.append(ep)
            self.priorities.append(pri)

    def sample_batch(self, batch_size):
        probs = np.array(self.priorities, dtype=np.float64) ** self.alpha
        probs /= probs.sum()

        ep_indices = np.random.choice(
            len(self.episodes), size=batch_size, p=probs, replace=True
        )

        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (len(self.episodes) * probs[ep_indices]) ** (-self.beta)
        weights /= weights.max()

        all_obs, all_acts, all_rews, all_dones = [], [], [], []

        for ei in ep_indices:
            ep = self.episodes[ei]
            max_start = ep.length - self.seq_len
            start = np.random.randint(0, max_start + 1)
            end = start + self.seq_len

            # obs has length transitions + 1, so slice end+1 is correct
            all_obs.append(np.stack(ep.obs[start:end + 1]).astype(np.float32))
            all_acts.append(ep.actions[start:end])
            all_rews.append(ep.rewards[start:end])
            all_dones.append([float(d) for d in ep.dones[start:end]])

        return (
            torch.tensor(np.stack(all_obs), dtype=torch.float32),
            torch.tensor(np.array(all_acts), dtype=torch.long),
            torch.tensor(np.array(all_rews), dtype=torch.float32),
            torch.tensor(np.array(all_dones), dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
            ep_indices,
        )

    def update_priorities(self, ep_indices, new_priorities):
        for idx, pri in zip(ep_indices, new_priorities):
            self.priorities[idx] = max(float(pri), self.min_priority)

    def __len__(self):
        return len(self.episodes)


def make_obs(obs, prev_action):
    one_hot = np.zeros(PREV_ACT_DIM, dtype=np.float32)
    if prev_action >= 0:
        one_hot[prev_action] = 1.0
    return np.concatenate([obs.astype(np.float32), one_hot])


def import_obelix(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


EXPLORE_PROBS = np.array([0.05, 0.10, 0.70, 0.10, 0.05])
TURN_ACTIONS = [0, 1, 3, 4]


def collect_episode(env_cls, args, seed, q_net, device, epsilon):
    env = env_cls(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=True,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        seed=seed,
    )
    raw_obs = env.reset(seed=seed)

    ep = EpisodeData()
    prev_action = -1
    aug = make_obs(raw_obs, prev_action)
    ep.obs.append(aug)

    h = q_net.init_hidden(1, device)
    done = False
    ep_return = 0.0

    while not done:
        if np.random.rand() < epsilon:
            if raw_obs[17] > 0.5:
                a = TURN_ACTIONS[np.random.randint(len(TURN_ACTIONS))]
            else:
                a = int(np.random.choice(N_ACTIONS, p=EXPLORE_PROBS))
        else:
            with torch.no_grad():
                obs_t = torch.tensor(
                    aug, dtype=torch.float32, device=device
                ).reshape(1, 1, -1)
                q_vals, h = q_net(obs_t, h)
                a = int(q_vals.squeeze(0).squeeze(0).argmax().item())

        raw_obs, r, done = env.step(ACTIONS[a], render=False)
        ep_return += r

        # Keep flags, but use env state when available.
        if r >= 90:
            ep.got_attachment = True
        if r >= 1900:
            ep.got_success = True

        clipped_r = max(r, -10.0)
        ep.actions.append(a)
        ep.rewards.append(clipped_r)
        ep.dones.append(done)

        prev_action = a
        aug = make_obs(raw_obs, prev_action)
        ep.obs.append(aug)

    ep.length = len(ep.actions)
    return ep, ep_return


def _burn_in_hidden(net, obs_prefix, device):
    """
    obs_prefix: [B, T, D]
    Returns hidden state after consuming prefix, detached/no-grad.
    """
    B = obs_prefix.shape[0]
    h0 = net.init_hidden(B, device)
    if obs_prefix.shape[1] == 0:
        return h0
    with torch.no_grad():
        _, h = net(obs_prefix, h0)
    return h


def train_step(q_net, tgt_net, replay, optimizer, device, args):
    obs_t, acts_t, rews_t, dones_t, weights_t, ep_indices = replay.sample_batch(args.batch_size)
    obs_t = obs_t.to(device)          # [B, SL+1, D]
    acts_t = acts_t.to(device)        # [B, SL]
    rews_t = rews_t.to(device)        # [B, SL]
    dones_t = dones_t.to(device)      # [B, SL]
    weights_t = weights_t.to(device)  # [B]

    B = obs_t.shape[0]
    SL = args.seq_len
    burn = args.burn_in

    # Current-state learn segment: states s_t for t in [burn, SL-1]
    obs_learn = obs_t[:, burn:SL, :]               # [B, SL-burn, D]
    acts_learn = acts_t[:, burn:]                  # [B, SL-burn]
    rews_learn = rews_t[:, burn:]                  # [B, SL-burn]
    dones_learn = dones_t[:, burn:]                # [B, SL-burn]

    # Next-state segment aligned to targets: s_{t+1}
    obs_next_learn = obs_t[:, burn + 1:SL + 1, :]  # [B, SL-burn, D]

    # Proper burn-in hidden states.
    # For q(s_t, a_t), hidden should have seen obs up to t-1 => prefix [:burn]
    # For q(s_{t+1}, ·), hidden should have seen obs up to t => prefix [:burn+1]
    burn_prefix = obs_t[:, :burn, :]
    burn_prefix_next = obs_t[:, :burn + 1, :]

    h_online = _burn_in_hidden(q_net, burn_prefix, device)
    h_online_next = _burn_in_hidden(q_net, burn_prefix_next, device)
    h_target_next = _burn_in_hidden(tgt_net, burn_prefix_next, device)

    # Online Q for current states (with grad)
    q_learn, _ = q_net(obs_learn, h_online)

    # Double-DQN target path (no grad)
    with torch.no_grad():
        q_next_online, _ = q_net(obs_next_learn, h_online_next)
        q_next_target, _ = tgt_net(obs_next_learn, h_target_next)

        next_actions = q_next_online.argmax(dim=-1)  # [B, T]
        next_q_val = q_next_target.gather(2, next_actions.unsqueeze(-1)).squeeze(-1)
        targets = rews_learn + args.gamma * (1.0 - dones_learn) * next_q_val

    q_taken = q_learn.gather(2, acts_learn.unsqueeze(-1)).squeeze(-1)

    td_errors = q_taken - targets
    per_seq_priority = td_errors.abs().max(dim=1).values.detach().cpu().numpy()

    # Keep same overall loss logic style: sample-wise weighting
    loss_per_sample = (td_errors ** 2).mean(dim=1)
    loss = (weights_t * loss_per_sample).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), 5.0)
    optimizer.step()

    replay.update_priorities(ep_indices, per_seq_priority)
    return loss.item()


def _run_eval(q_net, env_cls, args, device, wall_obstacles, difficulty, seeds=range(10)):
    scores = []
    attach_count = 0
    success_count = 0

    for seed in seeds:
        env = env_cls(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=2000,
            wall_obstacles=wall_obstacles,
            difficulty=difficulty,
            box_speed=args.box_speed,
            seed=seed,
        )
        raw_obs = env.reset(seed=seed)
        prev_action = -1
        h = q_net.init_hidden(1, device)
        done = False
        total = 0.0
        attached = False

        while not done:
            aug = make_obs(raw_obs, prev_action)
            with torch.no_grad():
                obs_t = torch.tensor(aug, dtype=torch.float32, device=device).reshape(1, 1, -1)
                q_vals, h = q_net(obs_t, h)
                a = int(q_vals.squeeze().argmax().item())

            raw_obs, r, done = env.step(ACTIONS[a], render=False)
            total += r
            prev_action = a

            if r >= 90 and not attached:
                attached = True
                attach_count += 1
            if r >= 1900:
                success_count += 1

        scores.append(total)

    return float(np.mean(scores)), float(np.std(scores)), attach_count, success_count


def evaluate_full(q_net, env_cls, args, device, seeds=range(10)):
    q_net.eval()
    difficulties = [0, 2, 3]
    diff_results = {}
    all_weighted = []

    for diff in difficulties:
        m_w, s_w, a_w, sc_w = _run_eval(q_net, env_cls, args, device, True, diff, seeds)
        m_nw, s_nw, a_nw, sc_nw = _run_eval(q_net, env_cls, args, device, False, diff, seeds)
        wm = 0.6 * m_w + 0.4 * m_nw
        ws = 0.6 * s_w + 0.4 * s_nw
        diff_results[diff] = {
            "wm": wm, "ws": ws,
            "m_w": m_w, "m_nw": m_nw,
            "a_w": a_w, "sc_w": sc_w,
            "a_nw": a_nw, "sc_nw": sc_nw,
        }
        all_weighted.append(wm)

    overall_mean = float(np.mean(all_weighted))
    q_net.train()
    return overall_mean, diff_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, default="./obelix.py")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--difficulty", type=int, default=2)
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=40)
    ap.add_argument("--burn_in", type=int, default=20)
    ap.add_argument("--max_replay_episodes", type=int, default=2000)
    ap.add_argument("--min_episodes", type=int, default=30)
    ap.add_argument("--target_sync", type=int, default=500)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--updates_per_episode", type=int, default=8)

    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_episodes", type=int, default=2000)

    ap.add_argument("--eval_every", type=int, default=100)

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    OBELIX = import_obelix(args.obelix_py)

    q_net = DuelingRecurrentDDQN(OBS_DIM, N_ACTIONS, args.hidden).to(device)
    tgt_net = DuelingRecurrentDDQN(OBS_DIM, N_ACTIONS, args.hidden).to(device)
    tgt_net.load_state_dict(q_net.state_dict())
    tgt_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr, eps=1e-5)
    replay = EpisodeReplay(args.max_replay_episodes, args.seq_len, args.burn_in)

    train_log = "newtrain_log_r2d2_full.csv"
    eval_log = "neweval_log_r2d2_full.csv"
    with open(train_log, "w", newline="") as f:
        csv.writer(f).writerow([
            "episode", "return", "ep_length", "epsilon",
            "attached", "success", "ddqn_loss", "replay_size",
        ])
    with open(eval_log, "w", newline="") as f:
        csv.writer(f).writerow([
            "episode", "overall_mean",
            "d0_weighted", "d0_wall", "d0_nowall", "d0_att_w", "d0_suc_w", "d0_att_nw", "d0_suc_nw",
            "d2_weighted", "d2_wall", "d2_nowall", "d2_att_w", "d2_suc_w", "d2_att_nw", "d2_suc_nw",
            "d3_weighted", "d3_wall", "d3_nowall", "d3_att_w", "d3_suc_w", "d3_att_nw", "d3_suc_nw",
            "best_eval",
        ])

    best_eval = -float("inf")
    update_count = 0

    pbar = tqdm(range(args.episodes), desc="R2D2 Full")

    for ep in pbar:
        frac = min(1.0, ep / args.eps_decay_episodes)
        epsilon = args.eps_start + frac * (args.eps_end - args.eps_start)

        ep_data, ep_return = collect_episode(
            OBELIX, args, args.seed + ep, q_net, device, epsilon
        )
        replay.add_episode(ep_data)

        ddqn_loss = 0.0
        if len(replay) >= args.min_episodes:
            for _ in range(args.updates_per_episode):
                ddqn_loss = train_step(q_net, tgt_net, replay, optimizer, device, args)
                update_count += 1
                if update_count % args.target_sync == 0:
                    tgt_net.load_state_dict(q_net.state_dict())

        tqdm.write(
            f"[ep {ep+1}] ret={ep_return:.0f} len={ep_data.length} eps={epsilon:.3f} "
            f"att={'Y' if ep_data.got_attachment else 'N'} "
            f"suc={'Y' if ep_data.got_success else 'N'} "
            f"ql={ddqn_loss:.4f} buf={len(replay)}"
        )

        with open(train_log, "a", newline="") as f:
            csv.writer(f).writerow([
                ep + 1, f"{ep_return:.1f}", ep_data.length, f"{epsilon:.4f}",
                int(ep_data.got_attachment), int(ep_data.got_success),
                f"{ddqn_loss:.6f}", len(replay),
            ])

        if (ep + 1) % args.eval_every == 0:
            overall_mean, dr = evaluate_full(q_net, OBELIX, args, device)
            tqdm.write(f"Eval @ ep {ep+1}: overall={overall_mean:.1f}")
            for d in [0, 2, 3]:
                r = dr[d]
                tqdm.write(
                    f"  d{d}: w={r['wm']:.1f} wall={r['m_w']:.1f}(att={r['a_w']}/10,suc={r['sc_w']}/10) "
                    f"nowall={r['m_nw']:.1f}(att={r['a_nw']}/10,suc={r['sc_nw']}/10)"
                )

            if overall_mean > best_eval:
                best_eval = overall_mean
                torch.save(q_net.state_dict(), os.path.join(save_dir, "best_r2d2_full.pth"))
                tqdm.write(f"  *** New best: {best_eval:.1f} ***")

            with open(eval_log, "a", newline="") as f:
                row = [ep + 1, f"{overall_mean:.2f}"]
                for d in [0, 2, 3]:
                    r = dr[d]
                    row.extend([
                        f"{r['wm']:.2f}", f"{r['m_w']:.2f}", f"{r['m_nw']:.2f}",
                        r['a_w'], r['sc_w'], r['a_nw'], r['sc_nw'],
                    ])
                row.append(f"{best_eval:.2f}")
                csv.writer(f).writerow(row)

        if (ep + 1) % 500 == 0:
            torch.save(q_net.state_dict(), os.path.join(save_dir, f"ckpt_ep{ep+1}.pth"))

    torch.save(q_net.state_dict(), os.path.join(save_dir, "final_r2d2_full.pth"))
    print("Done.")


if __name__ == "__main__":
    main()