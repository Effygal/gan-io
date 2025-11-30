from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch

from .data import ScalerTuple


def generate_synthetic(
    gen: torch.nn.Module,
    scalers: ScalerTuple,
    output_path: str | Path,
    device: torch.device,
    latent_dim: int,
    seq_len: int,
    num_entries: int,
    *,
    batch_size: int = 64,
) -> None:
    ts_scaler, length_scaler, lba_scaler, lat_scaler = scalers
    gen.eval()

    all_fakes: List[np.ndarray] = []
    rows_generated = 0

    with torch.no_grad():
        while rows_generated < num_entries:
            remaining = num_entries - rows_generated
            current_bs = min(batch_size, remaining // seq_len + 1)
            if current_bs <= 0:
                break

            noise = torch.randn(current_bs, seq_len, latent_dim, device=device)
            fake_seq = gen(noise).cpu().numpy()
            fake_seq_2d = fake_seq.reshape(-1, 4)

            fake_ts = fake_seq_2d[:, [0]]
            fake_len = fake_seq_2d[:, [1]]
            fake_lba = fake_seq_2d[:, [2]]
            fake_lat = fake_seq_2d[:, [3]]

            ts_unscaled = ts_scaler.inverse_transform(fake_ts)
            len_unscaled = length_scaler.inverse_transform(fake_len)
            lba_unscaled = lba_scaler.inverse_transform(fake_lba)
            lat_unscaled = lat_scaler.inverse_transform(fake_lat)

            combined = np.hstack([ts_unscaled, len_unscaled, lba_unscaled, lat_unscaled])
            all_fakes.append(combined)
            rows_generated += current_bs * seq_len

    output_rows = np.concatenate(all_fakes, axis=0)[:num_entries]
    path = Path(output_path)
    with path.open("w") as handle:
        for row in output_rows:
            out_str = " ".join(str(int(x)) for x in row)
            handle.write(out_str + "\n")

    print(f"Saved synthetic data to {path} (total lines = {len(output_rows)})")


__all__ = ["generate_synthetic"]
