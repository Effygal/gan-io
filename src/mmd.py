# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances

def read_orig(p):
    d = []
    with open(p, 'r') as f:
        for l in f:
            p = l.strip().split()
            if len(p) < 6:
                continue
            lba = int(p[4])
            length = int(p[3])
            # p = l.strip().split(',')
            # if len(p) < 5:
            #     continue
            # lba = int(p[2])
            # length = int(p[3])
            d.append([lba, length])
    return np.array(d, dtype=np.float64).reshape(-1, 2)

def read_synth(p):
    d = []
    with open(p, 'r') as f:
        for l in f:
            p = l.strip().split()
            if len(p) < 4:
                continue
            lba = int(p[2])
            length = int(p[1])
            d.append([lba, length])
    return np.array(d, dtype=np.float64).reshape(-1, 2)

def norm_data_using_original(x, y):
    """
    Fit the scaler on the original data x and then
    transform both x and y using the same scaler.
    """
    scaler = StandardScaler()
    scaler.fit(x)  # Fit only on original data
    x_norm = scaler.transform(x)
    y_norm = scaler.transform(y)
    return x_norm, y_norm

def med_sigma(x, y, samp=1000, rs=77):
    np.random.seed(rs)
    sx = min(samp, x.shape[0])
    sy = min(samp, y.shape[0])
    ix = np.random.choice(x.shape[0], size=sx, replace=False)
    iy = np.random.choice(y.shape[0], size=sy, replace=False)
    comb = np.vstack((x[ix], y[iy]))
    dist = pairwise_distances(comb, metric='euclidean')
    triu = np.triu_indices_from(dist, k=1)
    return np.median(dist[triu])

def rbf(x, y, sigma=1.0):
    x_norm = np.sum(x**2, axis=1).reshape(-1, 1)
    y_norm = np.sum(y**2, axis=1).reshape(1, -1)
    dist_sq = x_norm + y_norm - 2 * np.dot(x, y.T)
    return np.exp(-dist_sq / (2 * sigma**2))

def mmd(x, y, sigma=1.0):
    n, m = x.shape[0], y.shape[0]
    kxx = rbf(x, x, sigma=sigma)
    kyy = rbf(y, y, sigma=sigma)
    kxy = rbf(x, y, sigma=sigma)
    sum_kxx = (np.sum(kxx) - np.sum(np.diag(kxx))) / (n * (n - 1))
    sum_kyy = (np.sum(kyy) - np.sum(np.diag(kyy))) / (m * (m - 1))
    sum_kxy = np.sum(kxy) / (n * m)
    return sum_kxx + sum_kyy - 2 * sum_kxy

def mmd_norm(x, y, sigma=1.0, max_s=10000, rs=42):
    np.random.seed(rs)
    if len(x) > max_s:
        x = x[np.random.choice(len(x), size=max_s, replace=False)]
    if len(y) > max_s:
        y = y[np.random.choice(len(y), size=max_s, replace=False)]
    mmd2 = mmd(x, y, sigma=sigma)
    return mmd2, np.sqrt(mmd2)

def plot_feats(x, y, feats, save=None, bw_adjust=1.5):
    n = x.shape[1]
    plt.figure(figsize=(15, 4 * n))
    for i in range(n):
        plt.subplot(n, 1, i + 1)
        sns.kdeplot(x[:, i], label='Original', fill=True, bw_adjust=bw_adjust)
        sns.kdeplot(y[:, i], label='Synthetic', fill=True, bw_adjust=bw_adjust)
        plt.title(f'{feats[i]} Distribution')
        plt.xlabel(feats[i])
        plt.ylabel('Density')
        plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()
# %%
def main():
    # File paths
    # orig_p = "../traces/w44-r.txt"
    orig_p = "../traces/v521.txt"
    # orig_p = "../traces/volume766-orig.txt"
    # synth_p = "../traces/w44-r-synth.txt"
    # synth_p = "../traces/w44-gan-synth-trial12.txt"
    # synth_p = "../traces/w44-gan-synth-full.txt"
    # synth_p = "../traces/v766-synth.txt"
    synth_p = "../traces/v521-gan-synth.txt"
    
    print("Reading data...")
    x = read_orig(orig_p)
    y = read_synth(synth_p)
    
    print("Normalizing data using original trace statistics...")
    # Use the new normalization function that fits scaler on original data only
    x_norm, y_norm = norm_data_using_original(x, y)

    print("Computing median sigma for original-synthetic comparison...")
    sigma_val = med_sigma(x_norm, y_norm)
    print(f"Sigma (orig-synth): {sigma_val:.6f}")

    print("Computing MMD for original vs. synthetic...")
    mmd2, mmd_val = mmd_norm(x_norm, y_norm, sigma=sigma_val)
    print(f"MMD^2 = {mmd2:.6f}")
    print(f"MMD   = {mmd_val:.6f}")

    print("\nComputing MMD with random-gen baseline...")
    y_rand = np.random.randn(*y_norm.shape)
    print("Computing median sigma for original-random comparison...")
    sigma_val_rand = med_sigma(x_norm, y_rand)
    print(f"Sigma (orig-rand): {sigma_val_rand:.6f}")
    mmd2_rand, mmd_val_rand = mmd_norm(x_norm, y_rand, sigma=sigma_val_rand)
    print(f"MMD^2 (Random Baseline) = {mmd2_rand:.6f}")
    print(f"MMD   (Random Baseline) = {mmd_val_rand:.6f}")

    print("\nGenerating feature distribution plots...")
    feats = ["LBA", "Length"]
    plot_feats(x_norm, y_norm, feats, bw_adjust=1.5)

main()
# %%
#!/usr/bin/env python3
import sys, numpy as np

def rd(p, idx):
    a=[]
    with open(p) as f:
        for s in f:
            t=s.strip().replace(',',' ').split()
            if not t or len(t)<=max(idx): continue
            a.append([int(t[i]) for i in idx])
    return np.asarray(a, dtype=np.float64)

def medsig(x, S=2000, rs=0):
    r=np.random.default_rng(rs)
    if len(x)>S: x=x[r.choice(len(x),S,replace=False)]
    x2=(x*x).sum(1)
    D=np.maximum(x2[:,None]+x2[None,:]-2*x@x.T,0.0)
    iu=np.triu_indices_from(D,1)
    d=np.sqrt(D[iu]); m=np.median(d)
    return float(m if m>0 else 1.0)

def mmd2(x,y,s,B=2048):
    n,m=len(x),len(y); s2=2*s*s
    a=b=c=0.0
    for i in range(0,n,B):
        X=x[i:i+B]; x2=(X*X).sum(1)[:,None]
        for j in range(i,n,B):
            Y=x[j:j+B]; y2=(Y*Y).sum(1)[None,:]
            D=np.maximum(x2+y2-2*X@Y.T,0.0)
            if i==j: np.fill_diagonal(D, np.inf)
            K=np.exp(-D/s2); a+=K.sum() if i==j else 2*K.sum()
    a/=n*(n-1)
    for i in range(0,m,B):
        X=y[i:i+B]; x2=(X*X).sum(1)[:,None]
        for j in range(i,m,B):
            Y=y[j:j+B]; y2=(Y*Y).sum(1)[None,:]
            D=np.maximum(x2+y2-2*X@Y.T,0.0)
            if i==j: np.fill_diagonal(D, np.inf)
            K=np.exp(-D/s2); b+=K.sum() if i==j else 2*K.sum()
    b/=m*(m-1)
    for i in range(0,n,B):
        X=x[i:i+B]; x2=(X*X).sum(1)[:,None]
        for j in range(0,m,B):
            Y=y[j:j+B]; y2=(Y*Y).sum(1)[None,:]
            c+=np.exp(-np.maximum(x2+y2-2*X@Y.T,0.0)/s2).sum()
    c/=n*m
    return float(a+b-2*c)

op="../traces/v538.txt"
sp="../traces/v538-gan-synth.txt"
N=10000; rs=42

# LBA,NBLKS only
x=rd(op, idx=(4,5))
y=rd(sp, idx=(2,3))

r=np.random.default_rng(rs)
if len(x)>N: x=x[r.choice(len(x),N,replace=False)]
if len(y)>N: y=y[r.choice(len(y),N,replace=False)]

m=x.mean(0); s=x.std(0); s[s==0]=1.0
x=(x-m)/s; y=(y-m)/s
sg=medsig(x, S=min(2000,len(x)), rs=rs)

m2s=mmd2(x,y,sg)
yr=r.standard_normal(size=y.shape)
m2r=mmd2(x,yr,sg)

print(f"{x.shape} {y.shape}")
print(f"sigma {sg:.6f}")
print(f"MMD^2 synth {m2s:.6f} ; MMD {np.sqrt(max(m2s,0.0)):.6f}")
print(f"MMD^2 rand  {m2r:.6f} ; MMD {np.sqrt(max(m2r,0.0)):.6f}")