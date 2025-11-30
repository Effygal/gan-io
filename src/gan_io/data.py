from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


ScalerTuple = Tuple[MinMaxScaler, MinMaxScaler, MinMaxScaler, MinMaxScaler]


def load_and_scale_data(trace_path: str | Path, max_lines: Optional[int] = None) -> Tuple[np.ndarray, ScalerTuple]:
    rows: List[List[float]] = []
    path = Path(trace_path)
    with path.open("r") as handle:
        for idx, line in enumerate(handle):
            if max_lines is not None and idx >= max_lines:
                break

            parts = line.strip().split()
            if len(parts) < 6:
                continue

            timestamp = float(parts[1])
            length = float(parts[3])
            lba = float(parts[4])
            latency = float(parts[5])
            rows.append([timestamp, length, lba, latency])

    data_2d = np.array(rows, dtype=np.float32)

    timestamps = data_2d[:, 0].reshape(-1, 1)
    lengths = data_2d[:, 1].reshape(-1, 1)
    lbas = data_2d[:, 2].reshape(-1, 1)
    latencies = data_2d[:, 3].reshape(-1, 1)

    ts_scaler = MinMaxScaler(feature_range=(-1, 1))
    length_scaler = MinMaxScaler(feature_range=(-1, 1))
    lba_scaler = MinMaxScaler(feature_range=(-1, 1))
    lat_scaler = MinMaxScaler(feature_range=(-1, 1))

    ts_scaled = ts_scaler.fit_transform(timestamps)
    length_scaled = length_scaler.fit_transform(lengths)
    lba_scaled = lba_scaler.fit_transform(lbas)
    lat_scaled = lat_scaler.fit_transform(latencies)

    data_scaled = np.hstack([ts_scaled, length_scaled, lba_scaled, lat_scaled])
    return data_scaled, (ts_scaler, length_scaler, lba_scaler, lat_scaler)


class TraceSeqDataset(Dataset):
    def __init__(self, data_2d: np.ndarray, seq_len: int = 8) -> None:
        self.seq_len = seq_len
        self.data = data_2d
        total_rows = len(data_2d)
        self.num_seqs = total_rows // seq_len
        self.data_chunks = self.data[: self.num_seqs * seq_len].reshape(self.num_seqs, seq_len, 4)

    def __len__(self) -> int:
        return self.num_seqs

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.data_chunks[idx]


__all__ = ["load_and_scale_data", "TraceSeqDataset", "ScalerTuple"]
