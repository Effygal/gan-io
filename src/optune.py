#!/usr/bin/env python3

import optuna
import subprocess
import re
import argparse

SEED = 77

parser = argparse.ArgumentParser()
parser.add_argument("--trace-name", type=str, required=True,
                    help="Trace name (e.g., v521, v766, w44)")
parser.add_argument("--device", type=str, default="cuda",
                    help="Device to use: cuda, mps, cpu")
parser.add_argument("--n-trials", type=int, default=50,
                    help="Number of trials to run")
args = parser.parse_args()

TRACE_FILES = {
    "v521": "../traces/v521.txt",
    "v766": "../traces/v766.txt", 
    "v827": "../traces/v827.txt",
    "v538": "../traces/v538.txt",
    "w44": "../traces/w44_r.txt",
    "w82": "../traces/w82-r.txt",
    "w24": "../traces/w24-r.txt",
    "w11": "../traces/w11-r.txt"
}

def objective(trial):
    lrG = trial.suggest_float("lrG", 5e-5, 5e-4, log=True)
    lrD = trial.suggest_float("lrD", 1e-4, 5e-4, log=True)
    d_updates = trial.suggest_int("d_updates", 2, 3)
    g_updates = trial.suggest_int("g_updates", 1, 2)
    hidden_dim = trial.suggest_int("hidden_dim", 100, 128, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128])

    trace_file = TRACE_FILES.get(args.trace_name)
    if not trace_file:
        raise ValueError(f"Unknown trace name: {args.trace_name}. Available: {list(TRACE_FILES.keys())}")

    cmd = [
        "python", "main-early-stop.py",  # Use early-stop for faster training
        "--trace_path", trace_file,
        "--output_synth", f"../traces/{args.trace_name}-gan-synth.txt",
        "--device", args.device,
        "--batch_size", str(batch_size),
        "--num_epochs", "30",   
        "--latent_dim", "10",
        "--hidden_dim", str(hidden_dim),
        "--lrG", str(lrG),
        "--lrD", str(lrD),
        "--d_updates", str(d_updates),
        "--g_updates", str(g_updates),
        "--seq_len", "12",
        "--max_lines", "10000000"  # Limit for faster training
    ]

    print("\n===============================")
    print(f"Trial {trial.number} for {args.trace_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"lrG={lrG}, lrD={lrD}, d_up={d_updates}, g_up={g_updates}, "
          f"hidden_dim={hidden_dim}, batch_size={batch_size}")
    print("===============================")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("ERROR: subprocess returned non-zero exit code!")
        print("STDERR:", result.stderr)
        return (9999.0, 9999.0)  

    final_line = ""
    for line in result.stdout.splitlines():
        if line.startswith("[Epoch "):
            final_line = line

    if not final_line:
        print("ERROR: no epoch lines found in stdout!")
        print("Full stdout:\n", result.stdout)
        return (9999.0, 9999.0)

    match = re.search(r"D Loss:\s*([0-9\.]+)\s*\|\s*G Loss:\s*([0-9\.]+)", final_line)
    if not match:
        print("ERROR: final line not in expected format!")
        print("Final line:", final_line)
        return (9999.0, 9999.0)

    d_val = float(match.group(1))
    g_val = float(match.group(2))

    # 6) Return multi-objective tuple
    #   Let's define:
    #   Obj1 = |D - 0.5|
    #   Obj2 = G
    obj1 = abs(d_val - 0.5)
    obj2 = g_val

    print(f"Trial {trial.number} completed: D={d_val:.4f}, G={g_val:.4f}, "
          f"Obj=({obj1:.4f}, {obj2:.4f})")

    return (obj1, obj2)

study = optuna.create_study(
    directions=["minimize", "minimize"],
    sampler=optuna.samplers.TPESampler(seed=SEED)
)

print(f"Starting Optuna optimization for {args.trace_name}")
print(f"Device: {args.device}")
print(f"Trials: {args.n_trials}")

study.optimize(objective, n_trials=args.n_trials)

print("\n======= Search Finished =========")
print(f"Trace: {args.trace_name}")
print(f"Number of Pareto solutions: {len(study.best_trials)}")
print("Pareto front solutions:")
for i, t in enumerate(study.best_trials):
    print(f"Solution #{i}: values={t.values}, params={t.params}")