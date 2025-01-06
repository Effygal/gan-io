#!/usr/bin/env python3
# %%
import optuna
import subprocess
import re

SEED = 77

def objective(trial):
    lrG = trial.suggest_float("lrG", 5e-5, 5e-4, log=True)
    lrD = trial.suggest_float("lrD", 5e-5, 5e-4, log=True)
    d_updates = trial.suggest_int("d_updates", 1, 3)
    g_updates = trial.suggest_int("g_updates", 1, 3)
    hidden_dim = trial.suggest_int("hidden_dim", 100, 128, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128])

    cmd = [
        "python", "main-early-stop.py",
        "--trace_path", "../traces/w44_r.txt",
        # "--trace_path", "../traces/volume766-orig.txt",
        "--output_synth", "../traces/w44-gan-synth.txt",
        # "--output_synth", "../traces/volume766-gan-synth.txt",
        "--device", "mps",
        "--batch_size", str(batch_size),
        "--num_epochs", "50",   
        "--latent_dim", "10",
        "--hidden_dim", str(hidden_dim),
        "--lrG", str(lrG),
        "--lrD", str(lrD),
        "--d_updates", str(d_updates),
        "--g_updates", str(g_updates),
        "--seq_len", "20",
        # "--max_lines", "11368248"
        "--max_lines", "99878"
    ]

    print("\n===============================")
    print(f"Trial {trial.number} command: {' '.join(cmd)}")
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
study.optimize(objective, n_trials=500)

print("\n======= Search Finished =========")
print("Number of Pareto solutions:", len(study.best_trials))
print("Pareto front solutions:")
for i, t in enumerate(study.best_trials):
    print(f"Solution #{i}: values={t.values}, params={t.params}")