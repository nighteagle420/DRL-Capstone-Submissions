import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import time

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]




# -----------------------------
# DQN MODEL (must match training)
# -----------------------------
class DQN(nn.Module):
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


# -----------------------------
# Load model
# -----------------------------
def load_model():
    base = os.path.dirname(__file__)
    path = os.path.join(base, ".checkpoints/best_model_ep50.pth")  # change if needed

    model = DQN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# -----------------------------
# Select action using DDQN
# -----------------------------
def select_action(model, obs):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        q_values = model(obs_t).squeeze(0).numpy()

    return ACTIONS[int(np.argmax(q_values))]


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument("--box_speed", type=int, default=2)

    args = parser.parse_args()
    args.wall_obstacles = True
    print(f"WALL OBSTACLES - {args.wall_obstacles}, DIFFICULTY - {args.difficulty}")

    # Initialize environment
    bot = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
    )

    # Load DDQN model
    model = load_model()

    # Reset env
    obs = bot.reset()
    bot.render_frame()

    episode_reward = 0

    # -----------------------------
    # Run episode
    # -----------------------------
    for step in range(1, args.max_steps + 1):

        action = select_action(model, obs)

        obs, reward, done = bot.step(action)
        episode_reward += reward

        print(f"Step: {step}, Action: {action}, Reward: {reward}, Total: {episode_reward}")
        time.sleep(0.05)

        if done:
            print("Episode done. Total score:", episode_reward)
            break

    print("Finished episode.")