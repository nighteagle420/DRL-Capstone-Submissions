from __future__ import annotations
import argparse, random
import numpy as np
from tqdm import tqdm

import os
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


# -----------------------------
# Load OBELIX
# -----------------------------
def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


# -----------------------------
# Decay utilities
# -----------------------------
def decayItem(initValue, finalValue, episode, max_episode, decay_stop):
    decay_duration = int(decay_stop)
    episode = min(episode, decay_duration)

    frac = episode / decay_duration
    return float(initValue + frac * (finalValue - initValue))


def generateAllItems(initValue, finalValue, max_episode, decay_stop):
    return [decayItem(initValue, finalValue, ep, max_episode, decay_stop)
            for ep in range(max_episode)]


# -----------------------------
# State encoder (bit → int)
# -----------------------------
def obs_to_state(obs):
    s = 0
    for bit in obs:
        s = (s << 1) | int(bit)
    return s


# -----------------------------
# Evaluation (greedy policy)
# -----------------------------
def evaluate_policy(env_class, Q, args, num_episodes=10):
    rewards = []

    for ep in range(num_episodes):
        env = env_class(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed + 10000 + ep,
        )

        state = env.reset()
        done = False
        total_reward = 0
        discount = 1.0

        while not done:
            s = obs_to_state(state)
            action = np.argmax(Q[s])

            next_state, r, done = env.step(ACTIONS[action], render=True)

            total_reward += discount * r
            discount *= args.gamma
            state = next_state

        rewards.append(total_reward)

    return np.mean(rewards)


# -----------------------------
# MAIN TRAINING
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # env
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=10000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    # RL params
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lambda_", type=float, default=0.7)

    ap.add_argument("--alpha_start", type=float, default=0.3)
    ap.add_argument("--alpha_end", type=float, default=0.05)

    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)

    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    args.wall_obstacles = True

    # seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # env
    OBELIX = import_obelix(args.obelix_py)

    # Q-table
    NUM_STATES = 2 ** 18
    NUM_ACTIONS = 5

    Q = np.zeros((NUM_STATES, NUM_ACTIONS))
    E = np.zeros((NUM_STATES, NUM_ACTIONS))

    # decay schedules
    decay_stop = int(0.6 * args.episodes)

    epsilons = generateAllItems(args.eps_start, args.eps_end,
                               args.episodes, decay_stop)

    alphas = generateAllItems(args.alpha_start, args.alpha_end,
                             args.episodes, decay_stop)

    # evaluation tracking
    best_mean_reward = -float("inf")
    eval_interval = 500

    # -----------------------------
    # TRAIN LOOP
    # -----------------------------
    for ep in tqdm(range(args.episodes)):

        E.fill(0)

        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed + ep,
        )

        state = env.reset()
        done = False
        discount = 1.0
        total_reward = 0

        # initial action
        if np.random.random() < epsilons[ep]:
            action = np.random.randint(NUM_ACTIONS)
        else:
            action = np.argmax(Q[obs_to_state(state)])

        while not done:

            next_state, r, done = env.step(ACTIONS[action], render=False)

            total_reward += discount * r
            discount *= args.gamma

            s = obs_to_state(state)
            s_next = obs_to_state(next_state)

            # greedy action (for Q(lambda))
            max_q = Q[s_next].max()
            greedy_actions = np.where(Q[s_next] == max_q)[0]
            greedy_action = np.random.choice(greedy_actions)

            # epsilon-greedy next action
            if np.random.random() < epsilons[ep]:
                next_action = np.random.randint(NUM_ACTIONS)
            else:
                next_action = greedy_action

            # TD error
            td_target = r + args.gamma * Q[s_next].max() * (not done)
            td_error = td_target - Q[s][action]

            # eligibility trace update
            E[s][action] += 1

            # Q update
            Q += alphas[ep] * td_error * E

            # trace decay
            if next_action == greedy_action:
                E *= args.gamma * args.lambda_
            else:
                E.fill(0)

            state, action = next_state, next_action

        # -----------------------------
        # EVALUATION
        # -----------------------------
        if (ep + 1) % eval_interval == 0:
            mean_reward = evaluate_policy(OBELIX, Q, args, num_episodes=10)

            print(f"\n[Eval @ {ep+1}] Mean Reward: {mean_reward:.2f}")

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # np.save(f"q_table.npy{ep+1}", Q)
                np.save(os.path.join(save_dir, f"best_q_table{ep+1}.npy"), Q)
                print("New best model saved!")

    # final save
    # np.save("final_q_table.npy", Q)
    np.save(os.path.join(save_dir, "final_q_table.npy"), Q)
    print("Training complete.")


if __name__ == "__main__":
    main()