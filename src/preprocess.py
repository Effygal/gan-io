import numpy as np

def preprocess_trace(input_file):
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            t, op, addr, size = float(parts[0]), parts[1], int(parts[2]), int(parts[3])
            op = 0.0 if op.upper() == 'R' else 1.0
            data.append([t, op, addr, size])
    data = np.array(data)
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    normalized = (data - min_vals) / (max_vals - min_vals + 1e-6)
    return normalized, min_vals, max_vals

def denormalize_trace(data, min_vals, max_vals):
    return data * (max_vals - min_vals + 1e-6) + min_vals
