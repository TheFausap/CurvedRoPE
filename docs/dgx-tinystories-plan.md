# DGX TinyStories Plan

This is the first unattended CUDA experiment ladder for CurvedRoPE.

## Goal

Compare a standard RoPE tiny GPT against a hybrid path-RoPE tiny GPT on
TinyStories, keeping model size, tokens, optimizer, data, and schedule matched.

## Environment

Create a Python 3.12 environment on the DGX and install CUDA-enabled PyTorch.
Use the PyTorch install command appropriate for the DGX CUDA runtime, then add
the remaining dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-train.txt
```

If the system uses a different CUDA wheel channel, adjust the PyTorch index URL
before running the install.

Verify CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Data Preparation

The training script uses a byte-level vocabulary to avoid tokenizer training as
an extra experimental variable.

```bash
python experiments/train_tinystories.py \
  --prepare_data \
  --device cuda \
  --max_steps 1 \
  --eval_iters 1
```

This writes:

```text
data/tinystories_byte/train.bin
data/tinystories_byte/validation.bin
data/tinystories_byte/metadata.json
```

For a quick pipeline test, add `--max_train_stories 20000
--max_validation_stories 2000`.

## Matched Runs

Baseline:

```bash
python experiments/train_tinystories.py \
  --device cuda \
  --positional rope \
  --out_dir runs/tinystories_rope \
  --block_size 256 \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 384 \
  --batch_size 64 \
  --gradient_accumulation_steps 4 \
  --dtype bfloat16 \
  --max_steps 50000 \
  --eval_interval 1000 \
  --eval_iters 100 \
  --sample_interval 5000
```

Path hybrid:

```bash
python experiments/train_tinystories.py \
  --device cuda \
  --positional path_hybrid \
  --out_dir runs/tinystories_path_hybrid \
  --block_size 256 \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 384 \
  --path_heads 2 \
  --path_angle_scale 0.05 \
  --batch_size 64 \
  --gradient_accumulation_steps 4 \
  --dtype bfloat16 \
  --max_steps 50000 \
  --eval_interval 1000 \
  --eval_iters 100 \
  --sample_interval 5000
```

## What To Watch

- validation loss at matched steps
- tokens per second
- sample coherence every 5k steps
- `metrics.jsonl` in each run directory
- `path_gate` values in the hybrid run
- whether the hybrid catches up quickly or pays a large throughput penalty

## Analyze Results

After both runs have produced `metrics.jsonl`, generate a text verdict:

```bash
python experiments/analyze_metrics.py runs
```

Or specify runs explicitly:

```bash
python experiments/analyze_metrics.py \
  runs/tinystories_rope \
  runs/tinystories_path_hybrid
```

If `matplotlib` is installed, write a PNG chart:

```bash
python experiments/analyze_metrics.py runs --plot runs/tinystories_comparison.png
```

The verdict labels are:

- `PROMOTE`: validation loss improves enough, throughput is acceptable, and the
  path gate is active
- `PROMISING`: validation loss is close or slightly better with acceptable
  throughput
- `HOLD`: run more ablations or adjust the design before scaling
- `INSUFFICIENT`: missing baseline/candidate or eval metrics

## Promotion Criteria

The path hybrid is worth a larger FineWeb-Edu run if it gives:

- lower validation loss at matched token budget, or
- similar validation loss with better story continuity, and
- no severe throughput regression, and
- path gates move above their near-zero initialization.
