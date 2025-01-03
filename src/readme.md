# LSTM‐based GAN for I/O Trace Synthesis

This repository provides a **multi‐step LSTM** Generative Adversarial Network (GAN) to synthesize I/O trace data. The main script (`main.py`) implements:

1. **Data Preprocessing**  
2. **Generator Model**  
3. **Discriminator Model**  
4. **GAN Training Process**  
5. **Synthetic Trace Generation**  
6. **Hyperparameter Tuning Guidelines**

Below is a detailed explanation of each component, including relevant math expressions and tuning advice.

---

## 1. Data Preprocessing

### 1.1 Reading the Input Trace

We read a custom trace file containing columns:
$
[\text{timestamp},\, \text{length},\, \text{LBA},\, \text{latency}]
$
Specifically, we parse columns **[1, 3, 4, 5]** out of each line in the input trace. Optionally, we can limit how many lines we read by specifying `--max_lines` on the command line.

### 1.2 Scaling Each Column to $[-1, 1]$

We use **per‐column** `MinMaxScaler` with `feature_range=(-1,1)`. Formally, for each column $ x $, if $\min(x)$ and $\max(x)$ are the minimum and maximum values of that column in the dataset, then we map each $x_i$ to:

$
x_i' = -1 + \frac{(x_i - \min(x)) \times (1 - (-1))}{\max(x) - \min(x)}
     = -1 + 2 \cdot \frac{x_i - \min(x)}{\max(x) - \min(x)}
$

Thus the scaled data has columns strictly in $[-1, 1]$. This step helps stabilize training when we apply `tanh` in the Generator.

### 1.3 Splitting Into Multi‐Step Sequences

We group the scaled data into chunks (or subsequences) of length `seq_len` for LSTM training.  
- If $ N $ is the total number of rows, and `seq_len` is 12 by default, we get $\lfloor N / \text{seq\_len}\rfloor$ sequences.  
- Each sequence has shape $(\text{seq\_len}, 4)$.  
- We store them in a PyTorch `Dataset` (`TraceSeqDataset`) to feed batches to the GAN.

---

## 2. Generator Model

We implement a multi‐step LSTM generator `LSTMGenerator`, which receives a sequence of **noise vectors** $\mathbf{z}$ of shape $(\text{batch\_size}, \text{seq\_len}, \text{latent\_dim})$.

1. **Input:**  
   Each time step $t$ gets a random vector $\mathbf{z}_t \in \mathbb{R}^{\text{latent\_dim}}$.

2. **LSTM Unroll:**  
   We unroll an LSTM cell over the `seq_len` steps. The hidden size is `hidden_dim`.  
   Formally, for $ t \in \{1,\dots,\text{seq\_len}\} $:
  $ 
   \mathbf{h}_t, \mathbf{c}_t = \mathrm{LSTMCell}(\mathbf{z}_t, \mathbf{h}_{t-1}, \mathbf{c}_{t-1})
  $ 
   producing hidden states $\mathbf{h}_t$.

3. **Output Layer + Tanh:**  
   At each time step, we apply a fully‐connected layer to $\mathbf{h}_t$ to produce $\mathbf{o}_t \in \mathbb{R}^{4}$. Then we apply a `tanh` activation:
  $ 
   \mathbf{x}_t = \tanh(\mathbf{W}\,\mathbf{h}_t + \mathbf{b})
  $ 
   So each output step $\mathbf{x}_t \in [-1,1]^4$.  

Hence, the overall **Generator** maps random noise sequences $\{\mathbf{z}_1,\dots,\mathbf{z}_T\}$ to **synthetic data sequences** $\{\mathbf{x}_1,\dots,\mathbf{x}_T\}$, each in $[-1,1]^4$.

---

## 3. Discriminator Model

We implement a multi‐step LSTM `LSTMDiscriminator` that processes real or fake sequences of shape $(\text{batch\_size}, \text{seq\_len}, 4)$.

1. **Input:**  
   Each time step $t$ in the sequence has a 4‐dim vector $\mathbf{x}_t \in \mathbb{R}^4$.

2. **LSTM Unroll:**  
   Similarly, we unroll an LSTM over the entire sequence. At the final step $T = \text{seq\_len}$, we get a hidden state $\mathbf{h}_T$.

3. **Linear Output (Raw Logits):**  
   We pass $\mathbf{h}_T$ through a linear layer (no sigmoid) to get a single logit:
  $ 
   D(\{\mathbf{x}_t\}) = \mathbf{w}^\top \mathbf{h}_T + b
  $ 
   which is a scalar for each sequence in the batch. This logit is used with **BCEWithLogitsLoss**, so we do **not** apply any activation in `forward()`.

---

## 4. Training Process

We adopt a **GAN** framework where the Discriminator $ D $ tries to classify real sequences as **real** (label=1) and generated sequences as **fake** (label=0), while the Generator $ G $ tries to fool $ D $.

### 4.1 Loss Functions

1. **Discriminator Loss**  
   Given real samples $ \mathbf{x}^{(r)} $ with label 1 and generated samples $ \mathbf{x}^{(g)} $ with label 0, we compute a standard BCE loss **with logits**:

  $$ 
   \mathcal{L}_D = -\Big[
     \mathbb{E}[\log \sigma(D(\mathbf{x}^{(r)}))]
     + \mathbb{E}[\log (1 - \sigma(D(\mathbf{x}^{(g)})))]
   \Big]
  $$ 
   But in PyTorch, we feed raw logits into `nn.BCEWithLogitsLoss`, which handles the $\sigma(\cdot)$ internally.

2. **Generator Loss**  
   The Generator wants $ D(\mathbf{x}^{(g)}) \approx 1 $. So we do:
  $$ 
   \mathcal{L}_G = -\mathbb{E}[\log \sigma(D(\mathbf{x}^{(g)}))]
  $$ 
   i.e., the fake samples should be classified as real.  
   In code, we also pass real labels (1) for the generated data to the `BCEWithLogitsLoss`.

### 4.2 Training Updates

We train in mini‐batches from the `DataLoader`. For each mini‐batch:

1. **Discriminator update:**  
   - Sample real sequences from the dataset.  
   - Generate fake sequences from random noise.  
   - Compute Discriminator loss (real vs. fake).  
   - Update Discriminator weights (`optimizerD.step()`).

2. **Generator update(s):**  
   - Re‐sample noise or reuse the same.  
   - Generate fake sequences.  
   - Compute Generator loss (want them labeled as real).  
   - Update Generator weights (`optimizerG.step()`).  
   - We can do **multiple** Generator updates per Discriminator update (e.g., 2:1 or 3:1) to stabilize training, as suggested in the GAN paper.

### 4.3 Logging

After each epoch, we print the averaged D‐loss and G‐loss across the entire dataset. The script runs for a default of `--num_epochs=50`.

---

## 5. Trace Generation

After training, we call `generate_synthetic(...)` to produce **new** I/O traces:

1. **Generate in Batches of Sequences**  
   - For each batch, sample noise $\{z\}$ of shape $(\text{batch\_size}, \text{seq\_len}, \text{latent\_dim})$.  
   - Pass it through the Generator to get $\text{fake\_seq}\in [-1,1]$.

2. **Inverse Transform**  
   - We flatten the batch of shape $(\text{batch\_size} \times \text{seq\_len}, 4)$.  
   - For each column, we apply the inverse of `MinMaxScaler(feature_range=(-1,1))`.  
   - This yields real‐valued data approximating the distribution of the original columns.

3. **Writing the Synthetic Trace**  
   - We produce up to `--num_entries` total rows to the synthetic trace.  
   - Each row is written in plain text.

---

## 6. Hyperparameter Tuning Guidelines

GANs can be finicky to train. Below are some practical tips on how to tune the most critical hyperparameters:

### 6.1 D:G Update Ratio

- **Definition:**  
  The ratio of how many times the Discriminator (D) is updated to how many times the Generator (G) is updated in one iteration. For instance, if `G_UPDATES_PER_D = 2`, it means for every 1 Discriminator update, we do 2 Generator updates.

- **When to Increase G Updates (e.g., from 1 to 2 or 3):**  
  - If you see the **Discriminator** is quickly overpowering the Generator (D‐loss very low, G‐loss very high), you may want to train G **more** frequently so it can catch up.  
  - Also, if the Generator is not improving (G‐loss remains high), giving it extra training steps can help.

- **When to Decrease G Updates (or Increase D Updates):**  
  - If the **Generator** is dominating too fast (G‐loss very low, D‐loss very high), the Discriminator might not get enough training to classify well. Then you do more D updates to strengthen the Discriminator.  
  - Typically, if you see G‐loss dropping to near 0 while D‐loss stays near ~1.4 or higher, it might mean the Discriminator is undertrained.

- **Observing D‐loss & G‐loss:**  
  - **D‐loss near 0.5** can indicate a balanced scenario.  
  - **D‐loss >> 0.7** + **G‐loss ~ 0.0–0.3** might mean G is collapsing or D can’t keep up.  
  - Adjust ratio accordingly.

### 6.2 Batch Size

- **Effect on Training Dynamics:**  
  - A **larger batch size** (e.g., 64 → 128) tends to make training more stable (less noisy gradients), but it can also require more memory.  
  - A **smaller batch size** (e.g., 16 or 32) might let you see faster updates per iteration but with higher variance in the gradients.

- **When to Increase Batch Size:**  
  - If training is very noisy or unstable, and you have enough GPU memory, you might increase the batch size to smooth out gradient estimates.  
  - If you see widely oscillating D‐loss and G‐loss from iteration to iteration, a larger batch might help.

- **When to Decrease Batch Size:**  
  - If you’re hitting memory limits or if you notice the model converges too slowly with a large batch, try smaller batch sizes.  
  - In some tasks, small batch sizes can help the model explore more modes.

### 6.3 Learning Rate

- **Typical Range:**  
  - We often see stable results around $1\times10^{-4}$ to $3\times10^{-4}$ with Adam in many GAN tasks.  
  - However, you can sometimes go higher ($1\times10^{-3}$) or lower ($1\times10^{-5}$) depending on data complexity.

- **What the Learning Rate Influences:**  
  - **High LR** can make the network learn faster initially, but it can also cause divergence or oscillations if the updates overshoot minima.  
  - **Low LR** leads to slow but potentially more stable training. If G‐loss and D‐loss are barely changing over epochs, it might be too low.

- **How to Tune According to G‐loss & D‐loss:**  
  - If **both** losses stay around the same values for many epochs without improvement, you might **increase LR** a bit to escape a plateau.  
  - If losses fluctuate wildly (huge jumps each epoch) or quickly explode, try **decreasing LR**.

### 6.4 Sequence Length

- **Definition:**  
  - `seq_len` is how many consecutive time steps each LSTM sees as a single sample.  
  - For example, `seq_len=12` means each training sample is 12 consecutive rows from the trace.

- **Influence on Training:**  
  - **Longer seq_len** captures more temporal context but yields **fewer** total training samples (since you chunk the data in bigger blocks). This can reduce your effective dataset size and might cause overfitting or lead to heavier memory usage.  
  - **Shorter seq_len** gives more training samples but might lose some long‐term temporal patterns.

- **When to Increase seq_len:**  
  - If your data has important long‐range dependencies (e.g., patterns spanning 20–30 time steps) and you want the model to capture them.  
  - If you have enough data such that going to a higher `seq_len` still yields enough sequences.

- **When to Decrease seq_len:**  
  - If you see the training set is too small (the model quickly overfits or the Discriminator saturates).  
  - If memory usage is too high or you suspect that most patterns are short range anyway.

### 6.5 Diagnosing G‐loss and D‐loss Behavior

1. **G‐loss High, D‐loss Low:**  
   - The Discriminator is winning (classifying real vs. fake well), and the Generator can’t fool it.  
   - **Possible Actions:** Increase the G update ratio (e.g., from 1D:2G to 1D:3G), reduce the Discriminator’s learning rate, or raise the Generator’s learning rate slightly so G can catch up.

2. **G‐loss Low, D‐loss High:**  
   - The Generator is winning; the Discriminator is struggling to separate real from fake.  
   - **Possible Actions:** Increase D updates per iteration or raise the Discriminator’s LR. This ensures the Discriminator gets stronger.

3. **Both G‐loss and D‐loss Do Not Converge:**  
   - They keep oscillating or remain near random guess values (e.g., ~1.38 each for BCE).  
   - **Possible Actions:** Adjust your learning rates (lower LR if too oscillatory, higher if stuck). Consider changing the ratio of D:G updates. Possibly reduce batch size if you suspect the model is stuck in a local minimum.

4. **Equilibrium at a “Wrong” State:**  
   - Sometimes D‐loss stays near ~1.4 and G‐loss near ~0.7, neither going down. This can be a local equilibrium with little real improvement.  
   - **Possible Actions:** Drastically lower or raise LR, or adopt advanced techniques (e.g., gradient penalty or spectral norm). You might also reduce `seq_len` or modify your data preprocessing.

5. **General Tip:** Always examine actual generated samples to see if they look realistic. Losses alone can be misleading.

---

## How to Run

1. **Install Requirements**  
   Ensure you have PyTorch, NumPy, scikit‐learn, etc. installed.

2. **Prepare an Input Trace**  
   - The trace must have at least 6 columns, with columns `[1,3,4,5]` representing `[timestamp, length, LBA, latency]`.  
   - We limit reading to, e.g., 100,000 lines, use `--max_lines=100000`.

3. **Execute**  
   ```bash
    python src/main.py \
    --trace_path traces/w44_r.txt \
    --output_synth traces/w44_r_synth.txt \
    --device mps \
    --batch_size 128 \
    --num_epochs 50 \
    --latent_dim 16 \
    --hidden_dim 66 \
    --lrG 2e-05 --lrD 3e-05 \
    --d_updates 1 \
    --g_updates 2 \
    --seq_len 12 \
    --max_lines 100000


## 1. MMD

The Maximum Mean Discrepancy measures the distance between two distributions \( p \) and \( q \) given samples:
- \( X = \{x_1, x_2, \dots, x_n\} \) drawn from \( p \)
- \( Y = \{y_1, y_2, \dots, y_m\} \) drawn from \( q \)

In this project:
- \( X \) = samples from the **original** trace
- \( Y \) = samples from the **synthetic** trace

The MMD is defined in terms of a **kernel** function \( k(\cdot, \cdot) \). A popular choice is the **Radial Basis Function (RBF)** kernel, given by:

$$
k(x, y) \;=\; \exp\!\biggl( -\frac{\|x - y\|^2}{2 \sigma^2} \biggr),
$$

where \(\sigma\) is a bandwidth (scale) parameter.

---

## 2. Unbiased MMD\(^2\) Formula

The (unbiased) empirical estimate of MMD\(^2\) between two sample sets \( X \) and \( Y \) is often written as:

$$
\text{MMD}^2(X,Y) \;=\;
\frac{1}{n(n-1)} \sum_{\substack{i, j=1\\i \neq j}}^n k(x_i, x_j)
\;+\;
\frac{1}{m(m-1)} \sum_{\substack{i, j=1\\i \neq j}}^m k(y_i, y_j)
\;-\;
\frac{2}{nm} \sum_{i=1}^n \sum_{j=1}^m k(x_i, y_j).
$$

where:
- \(k(x_i, x_j)\) is the kernel evaluated on two points \(x_i\) and \(x_j\),
- \(n\) is the size of \(X\),
- \(m\) is the size of \(Y\).

Note that the **unbiased** version omits the diagonal terms \(k(x_i, x_i)\) and \(k(y_i, y_i)\). This typically gives a better unbiased estimator for \(\text{MMD}^2\).

---
## 3. Dealing with Large Datasets
### Subsampling
When the dataset is very large (millions of rows), computing the full \(N \times N\) kernel matrix is infeasible (you may encounter a `MemoryError`). Our solution is subsampling a subset of rows (by default 10,000) from each trace, which:
- Greatly reduces memory usage.
- Produces a rough approximation of the full MMD on the entire dataset.

### Kernel Approximation
Another alternative is to use Random Fourier Features or other kernel approximation techniques to compute MMD without the full pairwise distance. However, since subsampling is straightforward and commonly used, we choose that approach here.

## 4. Interpretation of MMD
MMD^2 close to 0 indicates the two distributions are similar under the RBF kernel measure.
Larger MMD^2 indicates more discrepancy.
Since we are comparing original vs. synthetic data, a lower MMD implies the synthetic distribution better matches the real distribution on these 4 columns 
[Timestamp, Length, LBA, Latency].
