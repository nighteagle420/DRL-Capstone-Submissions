import subprocess
import sys

AGENT_FILE = "agent_vegatach_shaped.py"
RUNS = 5
SEED = 2
MAX_STEPS = 1000

configs = [
    {"difficulty": 0, "wall": False},
    {"difficulty": 0, "wall": True},
    {"difficulty": 2, "wall": False},
    {"difficulty": 2, "wall": True},
    {"difficulty": 3, "wall": False},
    {"difficulty": 3, "wall": True},
]

results = []

for cfg in configs:
    cmd = [
        sys.executable, "evaluate.py",
        "--agent_file", AGENT_FILE,
        "--runs", str(RUNS),
        "--seed", str(SEED),
        "--max_steps", str(MAX_STEPS),
        "--difficulty", str(cfg["difficulty"]),
        # "--box_speed", "2",
    ]
    if cfg["wall"]:
        cmd.append("--wall_obstacles")

    label = f"diff={cfg['difficulty']} wall={cfg['wall']}"
    print(f"\n>>> Running: {label}")
    print(f"    {' '.join(cmd)}")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip())

    for line in proc.stdout.strip().split("\n"):
        if "mean=" in line:
            parts = line.split()
            mean = float([p for p in parts if p.startswith("mean=")][0].split("=")[1])
            std = float([p for p in parts if p.startswith("std=")][0].split("=")[1])
            results.append((label, mean, std))

print("\n" + "=" * 60)
print(f"{'Config':<30} {'Mean':>10} {'Std':>10}")
print("-" * 60)

all_means = []
for label, mean, std in results:
    print(f"{label:<30} {mean:>10.2f} {std:>10.2f}")
    all_means.append(mean)

# if all_means:
#     print("-" * 60)
#     print(f"{'Overall Average':<30} {sum(all_means)/len(all_means):>10.2f}")
# print("=" * 60)