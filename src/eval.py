# %%
import numpy as np

def calculate_mmd(trace1, trace2):
    real_trace = np.loadtxt(trace1, usecols=[1, 3, 4, 5])  
    synthetic_trace = np.loadtxt(trace2, usecols=[0, 1, 2, 3])

    if real_trace.shape[0] != synthetic_trace.shape[0]:
        raise ValueError("The number of rows in the two traces must match.")

    column_displacements = np.abs(real_trace - synthetic_trace).max(axis=0)

    return column_displacements
# %%
# Example usage:
# real_trace_path = "real_trace.txt"
# synthetic_trace_path = "synthetic_trace.txt"
mmd = calculate_mmd('../traces/w44_r.txt', '../traces/w44_r_synth.txt')
print(f"The MMD between the two traces is: {mmd}")

# %%
#!/usr/bin/env python3

import optuna
import subprocess
import re

def objective(trial):
    """
    Optuna objective:
      1. Suggests hyperparameters
      2. Calls main.py with 50 epochs
      3. Parses the final log line to get D-loss & G-loss
      4. Returns a tuple for multi-objective: (obj1, obj2) = (|D - 0.5|, G)
    """
    lrG = trial.suggest_float("lrG", 1e-4, 5e-4, log=True)
    lrD = trial.suggest_float("lrD", 1e-4, 5e-4, log=True)
    d_updates = trial.suggest_int("d_updates", 1, 3)
    g_updates = trial.suggest_int("g_updates", 1, 3)
    hidden_dim = trial.suggest_int("hidden_dim", 100, 128, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128])

    cmd = [
        "python", "main.py",
        # "--trace_path", "../traces/w44_r.txt",
        "--trace_path", "../traces/volume766-orig.txt",
        "--output_synth", "../traces/v766-gan-synth.txt",
        "--device", "mps",
        "--batch_size", str(batch_size),
        "--num_epochs", "30",   
        "--latent_dim", "16",
        "--hidden_dim", str(hidden_dim),
        "--lrG", str(lrG),
        "--lrD", str(lrD),
        "--d_updates", str(d_updates),
        "--g_updates", str(g_updates),
        "--seq_len", "20",
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
        return (9999.0, 9999.0)  # multi-objective must return a tuple

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

study = optuna.create_study(directions=["minimize", "minimize"])
study.optimize(objective, n_trials=50)

print("\n======= Search Finished =========")
print("Number of Pareto solutions:", len(study.best_trials))
print("Pareto front solutions:")
for i, t in enumerate(study.best_trials):
    print(f"Solution #{i}: values={t.values}, params={t.params}")

# %%
import numpy as np

def read_trace_original(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue  # skip malformed lines
            # columns are 0-based indexing in Python:
            # parts[1] -> Timestamp
            # parts[3] -> Length
            # parts[4] -> LBA
            # parts[5] -> Latency
            # timestamp = float(parts[1])
            # length    = float(parts[3])
            # lba       = float(parts[4])
            # latency   = float(parts[5])
            timestamp = float(parts[4])
            length    = float(parts[3])
            lba       = float(parts[2])
            latency   = float(parts[0])
            data.append([timestamp, length, lba, latency])
    return np.array(data, dtype=np.float64)

def read_trace_synthetic(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            timestamp = float(parts[0])
            length    = float(parts[1])
            lba       = float(parts[2])
            latency   = float(parts[3])
            data.append([timestamp, length, lba, latency])
    return np.array(data, dtype=np.float64)

def rbf_kernel(X, Y, sigma=1.0):
    X_norm = np.sum(X**2, axis=1).reshape(-1,1)
    Y_norm = np.sum(Y**2, axis=1).reshape(1,-1)
    dist_sq = X_norm + Y_norm - 2*np.dot(X, Y.T)
    K = np.exp(-dist_sq / (2 * sigma**2))
    return K

def mmd_unbiased(X, Y, sigma=1.0):
    n = X.shape[0]
    m = Y.shape[0]
    Kxx = rbf_kernel(X, X, sigma=sigma)
    Kyy = rbf_kernel(Y, Y, sigma=sigma)
    Kxy = rbf_kernel(X, Y, sigma=sigma)

    sum_Kxx = (np.sum(Kxx) - np.sum(np.diag(Kxx))) / (n * (n-1))
    sum_Kyy = (np.sum(Kyy) - np.sum(np.diag(Kyy))) / (m * (m-1))
    sum_Kxy = np.sum(Kxy) / (n * m)

    mmd2 = sum_Kxx + sum_Kyy - 2*sum_Kxy
    return mmd2

def compute_mmd_subsample(orig_path, synth_path, sigma=1.0, max_samples=10000):
    X = read_trace_original(orig_path)
    Y = read_trace_synthetic(synth_path)

    if len(X) > max_samples:
        idx = np.random.choice(len(X), size=max_samples, replace=False)
        X = X[idx]
    if len(Y) > max_samples:
        idx = np.random.choice(len(Y), size=max_samples, replace=False)
        Y = Y[idx]

    mmd2 = mmd_unbiased(X, Y, sigma=sigma)
    return mmd2, np.sqrt(mmd2)

# orig_path = "../traces/w44_r.txt"
orig_path = "../traces/volume766-orig.txt"
# synth_path = "../traces/w44_r_synth.txt"
synth_path = "../traces/volume766-gan-synth.txt"

sigma_val = 1e6
max_samples = 10000

mmd2_val, mmd_val = compute_mmd_subsample(orig_path, orig_path, 
                                            sigma=sigma_val, 
                                            max_samples=max_samples)
print(f"MMD^2 = {mmd2_val:.6f}")
print(f"MMD   = {mmd_val:.6f}")
# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
import random

def read_trace_original(path, max_entries=10000000):
    data = []
    count = 0  
    
    with open(path, 'r') as f:
        for line_number, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) < 6:
                continue 
            
            try:
                timestamp = float(parts[1])
                length    = float(parts[3])
                lba       = float(parts[4])
                latency   = float(parts[5])
                
                data.append([timestamp, length, lba, latency])
                count += 1
                
                if count >= max_entries:
                    print(f"Reached the maximum of {max_entries} entries at line {line_number}.")
                    break
            except ValueError as e:
                print(f"ValueError at line {line_number}: {e}. Skipping this line.")
                continue  
    
    print(f"Total valid entries read: {count}")
    return np.array(data, dtype=np.float64)

def read_trace_synthetic(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            timestamp = float(parts[0])
            length    = float(parts[1])
            lba       = float(parts[2])
            latency   = float(parts[3])
            data.append([timestamp, length, lba, latency])
    return np.array(data, dtype=np.float64)

def normalize_data(X, Y):
    scaler = StandardScaler()
    combined = np.vstack((X, Y))
    scaler.fit(combined)
    X_norm = scaler.transform(X)
    Y_norm = scaler.transform(Y)
    return X_norm, Y_norm

def compute_median_sigma(X, Y, sample_size=1000, random_state=42):
    np.random.seed(random_state)
    sample_size_X = min(sample_size, X.shape[0])
    sample_size_Y = min(sample_size, Y.shape[0])
    
    indices_X = np.random.choice(X.shape[0], size=sample_size_X, replace=False)
    indices_Y = np.random.choice(Y.shape[0], size=sample_size_Y, replace=False)
    
    X_sample = X[indices_X]
    Y_sample = Y[indices_Y]
    
    combined_sample = np.vstack((X_sample, Y_sample))
    
    distances = pairwise_distances(combined_sample, metric='euclidean')
    
    triu_indices = np.triu_indices_from(distances, k=1)
    upper_tri_distances = distances[triu_indices]
    
    median_distance = np.median(upper_tri_distances)
    return median_distance

def rbf_kernel(X, Y, sigma=1.0):
    X_norm = np.sum(X**2, axis=1).reshape(-1,1)
    Y_norm = np.sum(Y**2, axis=1).reshape(1,-1)
    dist_sq = X_norm + Y_norm - 2*np.dot(X, Y.T)
    K = np.exp(-dist_sq / (2 * sigma**2))
    return K

def mmd_unbiased(X, Y, sigma=1.0):
    n = X.shape[0]
    m = Y.shape[0]
    Kxx = rbf_kernel(X, X, sigma=sigma)
    Kyy = rbf_kernel(Y, Y, sigma=sigma)
    Kxy = rbf_kernel(X, Y, sigma=sigma)

    sum_Kxx = (np.sum(Kxx) - np.sum(np.diag(Kxx))) / (n * (n-1))
    sum_Kyy = (np.sum(Kyy) - np.sum(np.diag(Kyy))) / (m * (m-1))
    sum_Kxy = np.sum(Kxy) / (n * m)

    mmd2 = sum_Kxx + sum_Kyy - 2*sum_Kxy
    return mmd2

def compute_mmd_subsample_normalized(X_norm, Y_norm, sigma=1.0, max_samples=10000, random_state=42):
    np.random.seed(random_state)
    if len(X_norm) > max_samples:
        idx_X = np.random.choice(len(X_norm), size=max_samples, replace=False)
        X_norm = X_norm[idx_X]
    if len(Y_norm) > max_samples:
        idx_Y = np.random.choice(len(Y_norm), size=max_samples, replace=False)
        Y_norm = Y_norm[idx_Y]
    mmd2 = mmd_unbiased(X_norm, Y_norm, sigma=sigma)
    return mmd2, np.sqrt(mmd2)

def visualize_feature_distributions(X, Y, feature_names, save_path=None):
    num_features = X.shape[1]
    plt.figure(figsize=(15, 4 * num_features))
    
    for i in range(num_features):
        plt.subplot(num_features, 1, i+1)
        sns.kdeplot(X[:, i], label='Original', shade=True)
        sns.kdeplot(Y[:, i], label='Synthetic', shade=True)
        plt.title(f'Feature: {feature_names[i]} Distribution')
        plt.xlabel(feature_names[i])
        plt.ylabel('Density')
        plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    orig_path = "../traces/w44_r.txt"
    synth_path = "../traces/w44_r_synth.txt"
    
    max_samples = 10000
    random_state = 42
    
    print("Reading and normalizing data...")
    X = read_trace_original(orig_path, max_entries=10000000)
    Y = read_trace_synthetic(synth_path)
    X_norm, Y_norm = normalize_data(X, Y)  # Unpack only two values
    
    print("Computing median sigma...")
    sigma_val = compute_median_sigma(X_norm, Y_norm, sample_size=1000, random_state=random_state)
    print(f"Selected sigma (median of pairwise distances): {sigma_val:.6f}")
    
    print("Computing MMD...")
    mmd2_val, mmd_val = compute_mmd_subsample_normalized(
        X_norm, Y_norm, 
        sigma=sigma_val, 
        max_samples=max_samples,
        random_state=random_state
    )
    print(f"MMD^2 = {mmd2_val:.6f}")
    print(f"MMD   = {mmd_val:.6f}")
    
    print("\nPerforming baseline comparison with shuffled data...")
    X_shuffled = X_norm.copy()
    np.random.shuffle(X_shuffled)
    
    mmd2_different, mmd_different = compute_mmd_subsample_normalized(
        X_shuffled, Y_norm, 
        sigma=sigma_val, 
        max_samples=max_samples,
        random_state=random_state
    )
    print(f"MMD^2 (Shuffled) = {mmd2_different:.6f}")
    print(f"MMD (Shuffled)   = {mmd_different:.6f}")
    
    print("\nGenerating feature distribution plots...")
    feature_names = ["Timestamp", "Length", "LBA", "Latency"]
    visualize_feature_distributions(X_norm, Y_norm, feature_names)

main()
# %%



