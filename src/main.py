#!/usr/bin/env python3

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import random
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-step LSTM-based GAN with [-1,1] scaling and flexible hyperparams.")

    parser.add_argument('--trace_path', type=str, required=True,
                        help='Path to the original trace file.')
    parser.add_argument('--output_synth', type=str, default='synth.txt',
                        help='Path to save the synthetic trace.')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use: cpu, cuda, mps, etc.')

    parser.add_argument('--max_lines', type=int, default=None,
                        help='If set, only read this many lines from the trace file.')
    parser.add_argument('--seq_len', type=int, default=12,
                        help='How many rows (time steps) in each sequence.')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='LSTM hidden dimension.')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Dimension of the random noise vector per time step.')
    parser.add_argument('--num_entries', type=int, default=100000,
                        help='Number of synthetic rows to generate.')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training batch size.')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs.')

    parser.add_argument('--lrG', type=float, default=3e-4,
                        help='Learning rate for Generator.')
    parser.add_argument('--lrD', type=float, default=3e-4,
                        help='Learning rate for Discriminator.')

    parser.add_argument('--d_updates', type=int, default=1,
                        help='How many times to train the Discriminator per iteration.')
    parser.add_argument('--g_updates', type=int, default=2,
                        help='How many times to train the Generator per iteration.')
    parser.add_argument('--seed', type=int, default=77,
                        help='Random seed for reproducibility.')
    parser.add_argument('--save_dir', type=str, default='./models/',
                        help='Directory to save the trained models.')
    parser.add_argument('--save_prefix', type=str, default='model',
                        help='Prefix for the saved model files.')

    return parser.parse_args()

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

            # for alibaba trace
            # parts = line.strip().split(',')
            # if len(parts) < 5:
            #     continue
            # latency = float(parts[0])  
            # length = float(parts[3])    
            # lba = float(parts[2])      
            # timestamp = float(parts[4])

            rows.append([timestamp, length, lba, latency])

    data_2d = np.array(rows, dtype=np.float32)  # shape (N, 4)

    timestamps = data_2d[:, 0].reshape(-1, 1)
    lengths    = data_2d[:, 1].reshape(-1, 1)
    lbas       = data_2d[:, 2].reshape(-1, 1)
    latencies  = data_2d[:, 3].reshape(-1, 1)

    ts_scaler     = MinMaxScaler(feature_range=(-1,1))
    length_scaler = MinMaxScaler(feature_range=(-1,1))
    lba_scaler    = MinMaxScaler(feature_range=(-1,1))
    lat_scaler    = MinMaxScaler(feature_range=(-1,1))

    ts_scaled     = ts_scaler.fit_transform(timestamps)
    length_scaled = length_scaler.fit_transform(lengths)
    lba_scaled    = lba_scaler.fit_transform(lbas)
    lat_scaled    = lat_scaler.fit_transform(latencies)

    data_scaled = np.hstack([ts_scaled, length_scaled, lba_scaled, lat_scaled])

    scalers = (ts_scaler, length_scaler, lba_scaler, lat_scaler)
    return data_scaled, scalers

class TraceSeqDataset(Dataset):
    
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

class LSTMGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, seq_len, output_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim

        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        lstm_out, (hn, cn) = self.lstm(z)  # (batch_size, seq_len, hidden_dim)
        out = self.fc_out(lstm_out)       # (batch_size, seq_len, 4)
        out = torch.tanh(out)             # [-1,1]
        return out

class LSTMDiscriminator(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)  # single logit

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)      # (batch_size, seq_len, hidden_dim)
        final_feature = lstm_out[:, -1, :]     # (batch_size, hidden_dim)
        logit = self.fc_out(final_feature)     # (batch_size, 1)
        return logit

def train_gan(gen, disc, dataloader, device, latent_dim, seq_len,
              lrG, lrD, num_epochs, d_updates=1, g_updates=1):
    
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

            #=============================
            # (1) Train Discriminator (d_updates times)
            #=============================
            d_loss_current = 0.0
            for _ in range(d_updates):
                optimizerD.zero_grad()

                d_out_real = disc(real_data)
                d_loss_real = criterion(d_out_real, real_labels)

                z = torch.randn(batch_size, seq_len, latent_dim, device=device)
                fake_data = gen(z).detach()  # don't backprop through G
                d_out_fake = disc(fake_data)
                d_loss_fake = criterion(d_out_fake, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizerD.step()

                d_loss_current += d_loss.item()

            #=============================
            # (2) Train Generator (g_updates times)
            #=============================
            g_loss_current = 0.0
            for _ in range(g_updates):
                optimizerG.zero_grad()

                z = torch.randn(batch_size, seq_len, latent_dim, device=device)
                fake_data = gen(z)
                d_out_fake_for_g = disc(fake_data)
                g_loss = criterion(d_out_fake_for_g, real_labels)
                g_loss.backward()
                optimizerG.step()

                g_loss_current += g_loss.item()

            d_loss_sum += d_loss_current / d_updates
            g_loss_sum += g_loss_current / g_updates
            count_steps += 1

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"D Loss: {d_loss_sum/count_steps:.4f} | "
              f"G Loss: {g_loss_sum/count_steps:.4f}")
        

    return gen, disc, d_loss_sum/count_steps, g_loss_sum/count_steps

def generate_synthetic(gen, scalers, output_path, device, latent_dim, seq_len, num_entries):
    ts_scaler, length_scaler, lba_scaler, lat_scaler = scalers
    gen.eval()

    gen_bs = 64
    all_fakes = []
    rows_generated = 0

    with torch.no_grad():
        while rows_generated < num_entries:
            current_bs = min(gen_bs, (num_entries - rows_generated) // seq_len + 1)
            if current_bs <= 0:
                break

            z = torch.randn(current_bs, seq_len, latent_dim, device=device)
            fake_seq = gen(z).cpu().numpy()  # (current_bs, seq_len, 4)

            fake_seq_2d = fake_seq.reshape(-1, 4)

            fake_ts   = fake_seq_2d[:, [0]]
            fake_len  = fake_seq_2d[:, [1]]
            fake_lba  = fake_seq_2d[:, [2]]
            fake_lat  = fake_seq_2d[:, [3]]

            ts_unscaled   = ts_scaler.inverse_transform(fake_ts)
            len_unscaled  = length_scaler.inverse_transform(fake_len)
            lba_unscaled  = lba_scaler.inverse_transform(fake_lba)
            lat_unscaled  = lat_scaler.inverse_transform(fake_lat)

            combined_unscaled = np.hstack([ts_unscaled, len_unscaled, lba_unscaled, lat_unscaled])
            all_fakes.append(combined_unscaled)
            rows_generated += current_bs * seq_len

    all_fakes = np.concatenate(all_fakes, axis=0)
    all_fakes = all_fakes[:num_entries]

    with open(output_path, 'w') as f:
        for row in all_fakes:
            out_str = ' '.join(str(int(x)) for x in row)
            f.write(out_str + '\n')

    print(f"Saved synthetic data to {output_path} (total lines = {len(all_fakes)})")

def main():
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.save_dir, exist_ok=True)
    data_scaled, scalers = load_and_scale_data(args.trace_path, max_lines=args.max_lines)
    print(f"Loaded {data_scaled.shape[0]} lines from {args.trace_path} (max_lines={args.max_lines}).")

    dataset = TraceSeqDataset(data_scaled, seq_len=args.seq_len)
    if len(dataset) == 0:
        raise ValueError("Not enough data to form any sequence. Try smaller seq_len or larger data file.")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device(args.device)

    gen = LSTMGenerator(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
        output_dim=4
    ).to(device)

    disc = LSTMDiscriminator(
        input_dim=4,
        hidden_dim=args.hidden_dim
    ).to(device)

    gen, disc, _, _ = train_gan(
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
        g_updates=args.g_updates
    )
    # gen_save_path = os.path.join(args.save_dir, f"{args.save_prefix}_generator.pth")
    # disc_save_path = os.path.join(args.save_dir, f"{args.save_prefix}_discriminator.pth")
    # torch.save(gen.state_dict(), gen_save_path)
    # torch.save(disc.state_dict(), disc_save_path)

    # generate_synthetic(
    #     gen=gen,
    #     scalers=scalers,
    #     output_path=args.output_synth,
    #     device=device,
    #     latent_dim=args.latent_dim,
    #     seq_len=args.seq_len,
    #     num_entries=args.num_entries
    # )

    print("Done.")

if __name__ == "__main__":
    main()
