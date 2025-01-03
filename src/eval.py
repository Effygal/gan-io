# %%
import numpy as np

def calculate_mmd(trace1, trace2):
    # Load the specified columns from the traces
    real_trace = np.loadtxt(trace1, usecols=[1, 3, 4, 5])  # 2nd, 4th, 5th, 6th columns
    synthetic_trace = np.loadtxt(trace2, usecols=[0, 1, 2, 3])  # 1st, 2nd, 3rd, 4th columns

    # Ensure both traces have the same number of rows
    if real_trace.shape[0] != synthetic_trace.shape[0]:
        raise ValueError("The number of rows in the two traces must match.")

    # Calculate the maximum displacement for each column
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
    Optuna objective function that:
      1. Suggests hyperparameters
      2. Calls main.py with 50 epochs
      3. Parses the final log line to get D-loss & G-loss
      4. Returns a tuple for multi-objective: (obj1, obj2) = (|D - 0.5|, G)
    """
    # 1) Define your search ranges
    lrG = trial.suggest_float("lrG", 1e-4, 5e-4, log=True)
    lrD = trial.suggest_float("lrD", 1e-4, 5e-4, log=True)
    d_updates = trial.suggest_int("d_updates", 1, 3)
    g_updates = trial.suggest_int("g_updates", 1, 3)
    hidden_dim = trial.suggest_int("hidden_dim", 100, 128, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128])

    # 2) Build the command line (ensuring num_epochs=50)
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

    # 3) Run the command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)

    # 4) If there's any runtime error, mark objective as bad
    if result.returncode != 0:
        print("ERROR: subprocess returned non-zero exit code!")
        print("STDERR:", result.stderr)
        return (9999.0, 9999.0)  # multi-objective must return a tuple

    # 5) Parse the final line with "[Epoch 50/50] D Loss: X.XXXX | G Loss: YYYYY"
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
import os
import itertools
import numpy as np
import torch
from sklearn.model_selection import ParameterGrid
from sklearn.utils import shuffle

# Reuse components from your GAN implementation
from main import *

# Define hyperparameter ranges
param_grid = {
    'lrG': [1e-4, 3e-4, 5e-4],
    'lrD': [1e-4, 3e-4, 5e-4],
    'latent_dim': [16],
    'hidden_dim': [64],
    'seq_len': [12],
    'batch_size': [64, 128],
    'd_updates': [1, 2],
    'g_updates': [2, 3],
}
# param_grid = {
#     'lrG': [1e-4],
#     'lrD': [1e-4],
#     'latent_dim': [8],
#     'hidden_dim': [64],
#     'seq_len': [8],
#     'batch_size': [64],
#     'd_updates': [1],
#     'g_updates': [1],
# }

def hyperparameter_exploration(trace_path, max_lines, num_epochs, device, sample_size=100000):
    data_scaled, scalers = load_and_scale_data(trace_path, max_lines=max_lines)
    print(f"Loaded {data_scaled.shape[0]} lines for hyperparameter tuning.")
    data_scaled = shuffle(data_scaled, random_state=42)[:sample_size]

    best_config = None
    best_g_loss = float('inf')

    for params in ParameterGrid(param_grid):
        print(f"Testing configuration: {params}")
        dataset = TraceSeqDataset(data_scaled, seq_len=params['seq_len'])
        if len(dataset) == 0:
            print("Insufficient data for this configuration.")
            continue
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)

        gen = LSTMGenerator(
            latent_dim=params['latent_dim'],
            hidden_dim=params['hidden_dim'],
            seq_len=params['seq_len']
        ).to(device)

        disc = LSTMDiscriminator(
            input_dim=4,
            hidden_dim=params['hidden_dim']
        ).to(device)

        try:
            gen, disc, final_g_loss, final_d_loss = train_gan(
                gen=gen,
                disc=disc,
                dataloader=dataloader,
                device=device,
                latent_dim=params['latent_dim'],
                seq_len=params['seq_len'],
                lrG=params['lrG'],
                lrD=params['lrD'],
                num_epochs=num_epochs,
                d_updates=params['d_updates'],
                g_updates=params['g_updates']
            )
        except Exception as e:
            print(f"Error training with params {params}: {e}")
            continue

        print(f"Final G Loss: {final_g_loss:.4f}, D Loss: {final_d_loss:.4f}")

        if abs(final_d_loss - 0.5) < 0.1 and final_g_loss < best_g_loss:
            best_g_loss = final_g_loss
            best_config = params
            print("New best configuration found!")

    print("Hyperparameter exploration completed.")
    print(f"Best configuration: {best_config}")
    print(f"Best G Loss: {best_g_loss:.4f}")
    return best_config

trace_path = "../traces/w44_r.txt"
device = "mps"
best_params = hyperparameter_exploration(
    trace_path=trace_path,
    max_lines=100000,
    num_epochs=30,  # Keep it lower for tuning
    device=device
)
print(f"Best params: {best_params}")

# %%
import numpy as np

def read_trace_original(path):
    """
    Reads the original trace file and extracts columns:
      [1, 3, 4, 5] = [Timestamp, Length, LBA, Latency].
    Returns a NumPy array of shape (N, 4).
    """
    data = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue  # skip malformed lines
            # columns are 0-based indexing in Python:
            # parts[1] -> Timestamp
            # parts[3] -> Length
            # parts[4] -> LBA
            # parts[5] -> Latency
            timestamp = float(parts[1])
            length    = float(parts[3])
            lba       = float(parts[4])
            latency   = float(parts[5])
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
    """
    1. Reads the original and synthetic traces
    2. Subsamples both sets to 'max_samples' rows (or fewer if smaller)
    3. Computes MMD^2 (unbiased) and returns (MMD^2, MMD).
    """
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

orig_path = "../traces/w44_r.txt"
# synth_path = "../traces/w44_r_synth.txt"


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
    """
    Reads the synthetic trace file and extracts columns:
      [0, 1, 2, 3] = [Timestamp, Length, LBA, Latency].
    Returns a NumPy array of shape (N, 4).
    """
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
    """
    Computes the median of pairwise distances between a subset of X and Y.
    Args:
        X, Y: Normalized datasets.
        sample_size: Number of samples to draw from each dataset.
        random_state: Seed for reproducibility.
    Returns:
        sigma: Median of the sampled pairwise distances.
    """
    np.random.seed(random_state)
    # If datasets are smaller than sample_size, adjust accordingly
    sample_size_X = min(sample_size, X.shape[0])
    sample_size_Y = min(sample_size, Y.shape[0])
    
    # Randomly sample from X and Y
    indices_X = np.random.choice(X.shape[0], size=sample_size_X, replace=False)
    indices_Y = np.random.choice(Y.shape[0], size=sample_size_Y, replace=False)
    
    X_sample = X[indices_X]
    Y_sample = Y[indices_Y]
    
    # Combine samples
    combined_sample = np.vstack((X_sample, Y_sample))
    
    # Compute pairwise distances
    distances = pairwise_distances(combined_sample, metric='euclidean')
    
    # Extract the upper triangle without the diagonal
    triu_indices = np.triu_indices_from(distances, k=1)
    upper_tri_distances = distances[triu_indices]
    
    # Compute the median
    median_distance = np.median(upper_tri_distances)
    return median_distance

def rbf_kernel(X, Y, sigma=1.0):
    """
    Computes the RBF (Gaussian) kernel between X and Y with bandwidth sigma.
    """
    X_norm = np.sum(X**2, axis=1).reshape(-1,1)
    Y_norm = np.sum(Y**2, axis=1).reshape(1,-1)
    dist_sq = X_norm + Y_norm - 2*np.dot(X, Y.T)
    K = np.exp(-dist_sq / (2 * sigma**2))
    return K

def mmd_unbiased(X, Y, sigma=1.0):
    """
    Computes the unbiased estimate of MMD^2 between X and Y using the RBF kernel.
    """
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
    """
    Plots histograms and KDEs for each feature comparing original and synthetic data.
    Args:
        X, Y: Original and Synthetic normalized data.
        feature_names: List of feature names.
        save_path: If provided, saves the plots to the specified directory.
    """
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
    
    # Step 1: Read and normalize data
    print("Reading and normalizing data...")
    X = read_trace_original(orig_path, max_entries=10000000)
    Y = read_trace_synthetic(synth_path)
    X_norm, Y_norm = normalize_data(X, Y)  # Unpack only two values
    
    # Step 2: Compute median sigma using the median heuristic
    print("Computing median sigma...")
    sigma_val = compute_median_sigma(X_norm, Y_norm, sample_size=1000, random_state=random_state)
    print(f"Selected sigma (median of pairwise distances): {sigma_val:.6f}")
    
    # Step 3: Compute MMD with the selected sigma
    print("Computing MMD...")
    mmd2_val, mmd_val = compute_mmd_subsample_normalized(
        X_norm, Y_norm, 
        sigma=sigma_val, 
        max_samples=max_samples,
        random_state=random_state
    )
    print(f"MMD^2 = {mmd2_val:.6f}")
    print(f"MMD   = {mmd_val:.6f}")
    
    # Step 4: Baseline Comparison with Shuffled Data
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
    
    # Step 5: Visualization
    print("\nGenerating feature distribution plots...")
    feature_names = ["Timestamp", "Length", "LBA", "Latency"]
    visualize_feature_distributions(X_norm, Y_norm, feature_names)

main()
# %%
# Create a randomized version by shuffling the original data



