from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_gan(
    gen: torch.nn.Module,
    disc: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    latent_dim: int,
    seq_len: int,
    lrG: float,
    lrD: float,
    num_epochs: int,
    d_updates: int = 1,
    g_updates: int = 1,
) -> Tuple[torch.nn.Module, torch.nn.Module, float, float]:
    criterion = nn.BCEWithLogitsLoss()
    optimizerG = optim.Adam(gen.parameters(), lr=lrG, betas=(0.5, 0.999))
    optimizerD = optim.Adam(disc.parameters(), lr=lrD, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        d_loss_sum = 0.0
        g_loss_sum = 0.0
        count_steps = 0

        for real_data in dataloader:
            real_data = real_data.to(device, dtype=torch.float)
            batch_size = real_data.size(0)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            d_loss_current = 0.0
            for _ in range(d_updates):
                optimizerD.zero_grad()

                d_out_real = disc(real_data)
                d_loss_real = criterion(d_out_real, real_labels)

                noise = torch.randn(batch_size, seq_len, latent_dim, device=device)
                fake_data = gen(noise).detach()
                d_out_fake = disc(fake_data)
                d_loss_fake = criterion(d_out_fake, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizerD.step()

                d_loss_current += d_loss.item()

            g_loss_current = 0.0
            for _ in range(g_updates):
                optimizerG.zero_grad()

                noise = torch.randn(batch_size, seq_len, latent_dim, device=device)
                fake_data = gen(noise)
                d_out_fake_for_g = disc(fake_data)
                g_loss = criterion(d_out_fake_for_g, real_labels)
                g_loss.backward()
                optimizerG.step()

                g_loss_current += g_loss.item()

            d_loss_sum += d_loss_current / d_updates
            g_loss_sum += g_loss_current / g_updates
            count_steps += 1

        print(
            f"[Epoch {epoch + 1}/{num_epochs}] "
            f"D Loss: {d_loss_sum / count_steps:.4f} | "
            f"G Loss: {g_loss_sum / count_steps:.4f}"
        )

    return gen, disc, d_loss_sum / count_steps, g_loss_sum / count_steps


__all__ = ["train_gan"]
