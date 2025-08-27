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
import statistics

def pr_args():
    p = argparse.ArgumentParser()
    p.add_argument('--trace_path', type=str, required=True)
    p.add_argument('--output_synth', type=str, default='synth.txt')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--max_lines', type=int, default=None)
    p.add_argument('--seq_len', type=int, default=12)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--latent_dim', type=int, default=16)
    p.add_argument('--num_entries', type=int, default=100000)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_epochs', type=int, default=100)
    p.add_argument('--lrG', type=float, default=3e-4)
    p.add_argument('--lrD', type=float, default=3e-4)
    p.add_argument('--d_updates', type=int, default=1)
    p.add_argument('--g_updates', type=int, default=2)
    p.add_argument('--seed', type=int, default=77)
    p.add_argument('--save_dir', type=str, default='./models/')
    p.add_argument('--save_prefix', type=str, default='model')
    return p.parse_args()

def ld_sc(tp, mxl=None):
    rws = []
    with open(tp, 'r') as f:
        for i, ln in enumerate(f):
            if mxl is not None and i >= mxl:
                break
            
            # for cloudphysics
            pr = ln.strip().split()
            if len(pr) < 6:
                continue
            t = float(pr[1])
            lnth = float(pr[3])
            lb = float(pr[4])
            lat = float(pr[5])

            # for alibaba
            # pr = ln.strip().split(',')
            # if len(pr) < 5:
            #     continue
            # t = float(pr[4])
            # lnth = float(pr[3])
            # lb = float(pr[2])
            # lat = float(pr[0])

            rws.append([t, lnth, lb, lat])
    print(f"Read {len(rws)} lines")
    d2 = np.array(rws, dtype=np.float32)
    ts = d2[:, 0].reshape(-1, 1)
    lnth = d2[:, 1].reshape(-1, 1)
    lb = d2[:, 2].reshape(-1, 1)
    lat = d2[:, 3].reshape(-1, 1)
    s1 = MinMaxScaler(feature_range=(-1,1)).fit(ts)
    s2 = MinMaxScaler(feature_range=(-1,1)).fit(lnth)
    s3 = MinMaxScaler(feature_range=(-1,1)).fit(lb)
    s4 = MinMaxScaler(feature_range=(-1,1)).fit(lat)
    st1 = s1.transform(ts)
    st2 = s2.transform(lnth)
    st3 = s3.transform(lb)
    st4 = s4.transform(lat)
    dS = np.hstack([st1, st2, st3, st4])
    return dS, (s1, s2, s3, s4)

class TrSeqData(Dataset):
    def __init__(self, d2, sl=20):
        self.sl = sl
        self.d = d2
        N = len(d2)
        self.ns = N // sl
        self.dc = self.d[:self.ns*sl].reshape(self.ns, sl, 4)
    def __len__(self):
        return self.ns
    def __getitem__(self, i):
        return self.dc[i]

class LstmG(nn.Module):
    def __init__(self, ldim, hdim, sl, odim=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=ldim, hidden_size=hdim, batch_first=True)
        self.fc = nn.Linear(hdim, odim)
    def forward(self, z):
        o, _ = self.lstm(z)
        o = self.fc(o)
        o = torch.tanh(o)
        return o

class LstmD(nn.Module):
    def __init__(self, inp=4, hdim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size=inp, hidden_size=hdim, batch_first=True)
        self.fc = nn.Linear(hdim, 1)
    def forward(self, x):
        o, _ = self.lstm(x)
        f = o[:, -1, :]
        l = self.fc(f)
        return l

def adv_train(g, d, dl, dv, ld, sl, lrg, lrd, ep, du=1, gu=1):
    c = nn.BCEWithLogitsLoss()
    oG = optim.Adam(g.parameters(), lr=lrg, betas=(0.5, 0.999))
    oD = optim.Adam(d.parameters(), lr=lrd, betas=(0.5, 0.999))

    hist_len = 10  
    min_impr = 1e-4  
  
    pt_d = 0  
    pt_g = 0  
    max_pat = 12  

    hist_o1 = []
    hist_o2 = []

    best_o1 = float('inf')
    best_o2 = float('inf')

    for e in range(ep):
        d_sum = 0.0
        g_sum = 0.0
        c_steps = 0
        for real_data in dl:
            real_data = real_data.to(dv, dtype=torch.float)
            bs = real_data.size(0)
            rl = torch.ones(bs, 1, device=dv)
            fl = torch.zeros(bs, 1, device=dv)

            d_loss_agg = 0.0
            for _ in range(du):
                oD.zero_grad()
                dr = d(real_data)
                ldr = c(dr, rl)
                z = torch.randn(bs, sl, ld, device=dv)
                fd = g(z).detach()
                df = d(fd)
                ldf = c(df, fl)
                d_loss = ldr + ldf
                d_loss.backward()
                oD.step()
                d_loss_agg += d_loss.item()

            g_loss_agg = 0.0
            for _ in range(gu):
                oG.zero_grad()
                z = torch.randn(bs, sl, ld, device=dv)
                fd2 = g(z)
                df2 = d(fd2)
                g_loss = c(df2, rl)
                g_loss.backward()
                oG.step()
                g_loss_agg += g_loss.item()

            d_sum += d_loss_agg / du
            g_sum += g_loss_agg / gu
            c_steps += 1

        dL = d_sum / c_steps
        gL = g_sum / c_steps

        o1 = abs(dL - 0.5)
        o2 = gL

        print(f"[Epoch {e+1}/{ep}] D Loss: {dL:.4f} | G Loss: {gL:.4f}")

        hist_o1.append(o1)
        hist_o2.append(o2)
        if len(hist_o1) > hist_len:
            hist_o1.pop(0)
            hist_o2.pop(0)

        # Check improvements
        imp_d = best_o1 - o1
        imp_g = best_o2 - o2
        if imp_d > min_impr:
            pt_d = 0
            best_o1 = o1
        else:
            pt_d += 1
        if imp_g > min_impr:
            pt_g = 0
            best_o2 = o2
        else:
            pt_g += 1

        # 1) If both patience counters exceed max_pat => stop
        if pt_d >= max_pat and pt_g >= max_pat:
            print(f"EarlyStop: no improvement in both for {max_pat} epochs.")
            break

        # 2) If we have a stable plateau in last hist_len epochs
        if len(hist_o1) == hist_len:
            std_o1 = statistics.pstdev(hist_o1)
            std_o2 = statistics.pstdev(hist_o2)
            # If D is near 0.5 and stable, and G stable
            if std_o1 < 1e-3 and std_o2 < 1e-3:
                print("EarlyStop: plateau detected (std < 1e-3).")
                break
            # If G or O1 is drifting upward, we can stop
            # e.g. check slope sign; for simplicity, compare last 2 vs first 2 in the window
            if (hist_o1[-1] > hist_o1[0] + 1e-3) and (hist_o2[-1] > hist_o2[0] + 1e-3):
                print("EarlyStop: consistent upward drift in last window.")
                break

    return g, d, dL, gL

def normal_train(g, d, dl, dv, ld, sl, lrg, lrd, ep, du=1, gu=1):
    crit = nn.BCEWithLogitsLoss()
    oG = optim.Adam(g.parameters(), lr=lrg, betas=(0.5, 0.999))
    oD = optim.Adam(d.parameters(), lr=lrd, betas=(0.5, 0.999))

    prev_dloss = None
    diverge_thresh = 0.3
    close_thresh = 0.05

    for e in range(ep):
        d_sum = 0.0
        g_sum = 0.0
        steps = 0

        for real_data in dl:
            real_data = real_data.to(dv, dtype=torch.float)
            bs = real_data.size(0)
            rl = torch.ones(bs, 1, device=dv)
            fl = torch.zeros(bs, 1, device=dv)

            d_batch_loss = 0.
            for _ in range(du):
                oD.zero_grad()
                dr = d(real_data)
                ldr = crit(dr, rl)
                z = torch.randn(bs, sl, ld, device=dv)
                fd = g(z).detach()
                df = d(fd)
                ldf = crit(df, fl)
                lD = ldr + ldf
                lD.backward()
                oD.step()
                d_batch_loss += lD.item()

            g_batch_loss = 0.
            for _ in range(gu):
                oG.zero_grad()
                z = torch.randn(bs, sl, ld, device=dv)
                fd2 = g(z)
                df2 = d(fd2)
                lG = crit(df2, rl)
                lG.backward()
                oG.step()
                g_batch_loss += lG.item()

            d_sum += d_batch_loss / du
            g_sum += g_batch_loss / gu
            steps += 1

        dL = d_sum / steps
        gL = g_sum / steps
        print(f"[Epoch {e+1}/{ep}] D Loss: {dL:.4f} | G Loss: {gL:.4f}")

        # Early-stop only if we see a "sudden" jump from far-from-0.5 to near-0.5
        if prev_dloss is not None:
            if (abs(prev_dloss - 0.5) > diverge_thresh) and (abs(dL - 0.5) < close_thresh):
                print(f"EarlyStop: D-loss jumped from {prev_dloss:.3f} to near 0.5 => stop.")
                break

        prev_dloss = dL

    return g, d, dL, gL


def main():
    a = pr_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(a.save_dir, exist_ok=True)

    data2d, scs = ld_sc(a.trace_path, a.max_lines)
    if data2d.shape[0] == 0:
        raise ValueError("No valid lines in trace.")
    print(f"Loaded {data2d.shape[0]} lines")

    ds = TrSeqData(data2d, sl=a.seq_len)
    if len(ds) == 0:
        raise ValueError("No sequences formed.")
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True)

    dv = torch.device(a.device)
    gen = LstmG(a.latent_dim, a.hidden_dim, a.seq_len).to(dv)
    dis = LstmD(inp=4, hdim=a.hidden_dim).to(dv)
    gen, dis, dL, gL = normal_train(
        gen, dis, dl, dv,
        ld=a.latent_dim, sl=a.seq_len,
        lrg=a.lrG, lrd=a.lrD,
        ep=a.num_epochs,
        du=a.d_updates,
        gu=a.g_updates
    )
    print("Done.")

if __name__ == "__main__":
    main()
