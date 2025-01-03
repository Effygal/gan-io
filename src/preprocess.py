# src/data_preprocessing.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_normalize_trace(trace_path):
    # 1. Load data
    data = []
    with open(trace_path, 'r') as f:
        for line in f:
            # Each line: e.g. "2 7561692514266 28 512 206848 142"
            parts = line.strip().split()
            # Convert strings -> floats
            values = list(map(float, parts))
            data.append(values)

    data = np.array(data)  # shape: (num_samples, num_features)

    # 2. Fit scaler
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)  # in [0,1]

    return data_normalized, scaler
