# gan-io

LSTM-based GAN for synthetic I/O generations.

## Install (e.g., in uv workflow)

```bash
uv venv
source .venv/bin/activate
uv pip install --upgrade pip
uv pip install -e .
```

## Run

```bash
uv run gan-io \
  --trace_path 'traces/input_trace.txt' \
  --output_synth 'synth.txt' \
  --device cpu \
  --batch_size 128 \
  --num_epochs 100 \
  --seq_len 12 \
  --latent_dim 16 \
  --hidden_dim 128 \
  --lrG 3e-4 \
  --lrD 3e-4 \
  --d_updates 1 \
  --g_updates 2 \
  --max_lines 100000 \
  --num_entries 100000
```

### CLI Flags

- `--trace_path`: source trace file (required)
- `--output_synth`: output path for the synthetic trace
- `--device`: cpu, cuda, mps, etc.
- `--batch_size`: training batch size
- `--num_epochs`: epochs to train
- `--seq_len`: rows per LSTM sequence
- `--latent_dim`: noise vector size
- `--hidden_dim`: LSTM hidden dimension (G and D)
- `--lrG`, `--lrD`: learning rates
- `--d_updates`, `--g_updates`: update ratios
- `--max_lines`: optional cap on input lines
- `--num_entries`: total synthetic rows to emit
- `--save_dir`, `--save_prefix`: checkpoint directory/prefix
- `--seed`: RNG seed for reproducibility
