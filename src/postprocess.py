###############################################################################
# 2. LOAD & PER-COLUMN SCALE DATA TO [-1,1]
###############################################################################
def load_and_scale_data(trace_path, max_lines=None):
    rows = []
    with open(trace_path, 'r') as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break

            parts = line.strip().split()
            if len(parts) < 6:
                continue

            timestamp = float(parts[1])
            length    = float(parts[3])
            lba       = float(parts[4])
            latency   = float(parts[5])
            rows.append([timestamp, length, lba, latency])

    data_2d = np.array(rows, dtype=np.float32)  # shape (N, 4)

    # Separate columns
    timestamps = data_2d[:, 0].reshape(-1, 1)
    lengths    = data_2d[:, 1].reshape(-1, 1)
    lbas       = data_2d[:, 2].reshape(-1, 1)
    latencies  = data_2d[:, 3].reshape(-1, 1)

    # Fit a separate MinMaxScaler per column with feature_range=(-1,1)
    ts_scaler     = MinMaxScaler(feature_range=(-1,1))
    length_scaler = MinMaxScaler(feature_range=(-1,1))
    lba_scaler    = MinMaxScaler(feature_range=(-1,1))
    lat_scaler    = MinMaxScaler(feature_range=(-1,1))

    ts_scaled     = ts_scaler.fit_transform(timestamps)
    length_scaled = length_scaler.fit_transform(lengths)
    lba_scaled    = lba_scaler.fit_transform(lbas)
    lat_scaled    = lat_scaler.fit_transform(latencies)

    # Combine scaled columns
    data_scaled = np.hstack([ts_scaled, length_scaled, lba_scaled, lat_scaled])

    scalers = (ts_scaler, length_scaler, lba_scaler, lat_scaler)
    return data_scaled, scalers

###############################################################################
# 3. CHUNK DATA INTO MULTI-STEP SEQUENCES
###############################################################################
class TraceSeqDataset(Dataset):
    """
    Splits the entire (N,4) array into non-overlapping chunks of shape (seq_len, 4).
    If N is not divisible by seq_len, we drop the remainder.
    """
    def __init__(self, data_2d, seq_len=8):
        self.seq_len = seq_len
        self.data = data_2d
        N = len(data_2d)
        self.num_seqs = N // seq_len
        self.data_chunks = self.data[:self.num_seqs*seq_len].reshape(self.num_seqs, seq_len, 4)

    def __len__(self):
        return self.num_seqs

    def __getitem__(self, idx):
        return self.data_chunks[idx]