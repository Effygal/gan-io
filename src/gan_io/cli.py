from __future__ import annotations

import argparse
import os
import random
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import TraceSeqDataset, load_and_scale_data
from .generation import generate_synthetic
from .models import LSTMDiscriminator, LSTMGenerator
from .training import train_gan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an LSTM-based GAN and emit a synthetic trace.")
    parser.add_argument("--trace_path", type=str, required=True, help="Path to the original trace file.")
    parser.add_argument("--output_synth", type=str, default="synth.txt", help="Path to save the synthetic trace.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: cpu, cuda, mps, etc.")

    parser.add_argument("--max_lines", type=int, default=None, help="Optional cap on lines read from the trace.")
    parser.add_argument("--seq_len", type=int, default=12, help="Number of rows per sequence for the LSTM.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="LSTM hidden dimension.")
    parser.add_argument("--latent_dim", type=int, default=16, help="Dimension of the noise vector per time step.")
    parser.add_argument(
        "--num_entries",
        type=int,
        default=None,
        help="Number of synthetic rows to generate. Defaults to the valid line count.",
    )

    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--lrG", type=float, default=3e-4, help="Learning rate for the Generator.")
    parser.add_argument("--lrD", type=float, default=3e-4, help="Learning rate for the Discriminator.")
    parser.add_argument("--d_updates", type=int, default=1, help="Number of Discriminator updates per iteration.")
    parser.add_argument("--g_updates", type=int, default=2, help="Number of Generator updates per iteration.")

    parser.add_argument("--save_dir", type=str, default="./models/", help="Directory to save the trained models.")
    parser.add_argument("--save_prefix", type=str, default="model", help="Prefix for the saved model files.")
    parser.add_argument("--seed", type=int, default=77, help="Random seed for reproducibility.")
    return parser


def run(args: argparse.Namespace) -> None:
    os.makedirs(args.save_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    data_scaled, scalers = load_and_scale_data(args.trace_path, max_lines=args.max_lines)
    print(f"Loaded {data_scaled.shape[0]} lines from {args.trace_path} (max_lines={args.max_lines}).")

    if args.num_entries is None:
        args.num_entries = int(data_scaled.shape[0])

    dataset = TraceSeqDataset(data_scaled, seq_len=args.seq_len)
    if len(dataset) == 0:
        raise ValueError("Not enough data to form any sequence. Try smaller seq_len or a larger trace file.")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device(args.device)

    gen = LSTMGenerator(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
        output_dim=4,
    ).to(device)

    disc = LSTMDiscriminator(input_dim=4, hidden_dim=args.hidden_dim).to(device)

    gen, disc, dloss, gloss = train_gan(
        gen=gen,
        disc=disc,
        dataloader=dataloader,
        device=device,
        latent_dim=args.latent_dim,
        seq_len=args.seq_len,
        lrG=args.lrG,
        lrD=args.lrD,
        num_epochs=args.num_epochs,
        d_updates=args.d_updates,
        g_updates=args.g_updates,
    )

    gen_save_path = os.path.join(args.save_dir, f"{args.save_prefix}_generator.pth")
    disc_save_path = os.path.join(args.save_dir, f"{args.save_prefix}_discriminator.pth")

    torch.save(gen.state_dict(), gen_save_path)
    torch.save(disc.state_dict(), disc_save_path)
    print(f"Generator saved to {gen_save_path}")
    print(f"Discriminator saved to {disc_save_path}")

    generate_synthetic(
        gen=gen,
        scalers=scalers,
        output_path=args.output_synth,
        device=device,
        latent_dim=args.latent_dim,
        seq_len=args.seq_len,
        num_entries=args.num_entries,
    )
    print("Done.")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


__all__ = ["build_parser", "main", "run"]
