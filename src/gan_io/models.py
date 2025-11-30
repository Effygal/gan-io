from __future__ import annotations

import torch
import torch.nn as nn


class LSTMGenerator(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, seq_len: int, output_dim: int = 4) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim

        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(noise)
        out = self.fc_out(lstm_out)
        return torch.tanh(out)


class LSTMDiscriminator(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(sequence)
        final_feature = lstm_out[:, -1, :]
        return self.fc_out(final_feature)


__all__ = ["LSTMGenerator", "LSTMDiscriminator"]
