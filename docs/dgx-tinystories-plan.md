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

## Follow-Up After First Result

The first full path-hybrid run was not conclusive:

```text
rope best_val:        0.4993
path_hybrid best_val: 0.5006
path throughput:      0.481x baseline
path_gate:            0.1328
```

Interpretation: the path branch is being used, but the full implementation is
too slow and did not beat RoPE at this scale. The next ladder should test
whether a cheaper path branch can keep the useful signal while reducing the
throughput penalty.

Sparse path, last two layers only:

```bash
python experiments/train_tinystories.py \
  --device cuda \
  --positional path_hybrid \
  --out_dir runs/tinystories_path_last2 \
  --block_size 256 \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 384 \
  --path_layer_start 4 \
  --path_heads 1 \
  --path_blocks 4 \
  --path_angle_scale 0.05 \
  --batch_size 64 \
  --gradient_accumulation_steps 4 \
  --dtype bfloat16 \
  --max_steps 50000 \
  --eval_interval 1000 \
  --eval_iters 100 \
  --sample_interval 5000
```

Sparse path, alternating upper layers:

```bash
python experiments/train_tinystories.py \
  --device cuda \
  --positional path_hybrid \
  --out_dir runs/tinystories_path_upper_alt \
  --block_size 256 \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 384 \
  --path_layer_start 2 \
  --path_layer_interval 2 \
  --path_heads 1 \
  --path_blocks 4 \
  --path_angle_scale 0.05 \
  --batch_size 64 \
  --gradient_accumulation_steps 4 \
  --dtype bfloat16 \
  --max_steps 50000 \
  --eval_interval 1000 \
  --eval_iters 100 \
  --sample_interval 5000
```

If these variants remain close to RoPE while recovering much of the throughput,
then the mechanism is still alive. If they are still slower and not better,
TinyStories is probably not the right scale/task for this path design.

## Affine Path Direction

The analogy/functor framing suggests pure rotation may be too restrictive. The
next branch adds an affine translation prefix:

```text
b_t = R_t b_(t-1) + tau_t
x_t' = R_t x_t + b_t
```

This is exposed as `--positional path_affine`. It should be compared against
the current lead candidate, `path_last1`, using the same late-layer placement.

Affine last layer:

```bash
python experiments/train_tinystories.py \
  --device cuda \
  --positional path_affine \
  --out_dir runs/tinystories_path_affine_last1 \
  --block_size 256 \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 384 \
  --path_layer_start 5 \
  --path_heads 1 \
  --path_blocks 4 \
  --path_angle_scale 0.05 \
  --path_translation_scale 0.05 \
  --batch_size 64 \
  --gradient_accumulation_steps 4 \
  --dtype bfloat16 \
  --max_steps 50000 \
  --eval_interval 1000 \
  --eval_iters 100 \
  --sample_interval 5000
```

Watch both `path_gate` and `affine_gate`. If `affine_gate` stays near zero, the
model is rejecting the translation component. If it turns on while validation
improves or stays at parity, the affine/functor hypothesis deserves a larger
analogy-style diagnostic.

## What To Watch

- validation loss at matched steps
- tokens per second
- sample coherence every 5k steps
- `metrics.jsonl` in each run directory
- `path_gate` values in the hybrid run
- `affine_gate` values in affine path runs
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

For paired seed comparisons, suffix run directories with the seed, for example
`tinystories_rope_1431` and `tinystories_path_last1_1431`, then run:

```bash
python experiments/analyze_metrics.py runs \
  --paired \
  --baseline_group rope \
  --candidate_group path_last1
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

# FineWeb-Edu 1B Pilot

## Goal

Promote the replicated TinyStories `path_last1` candidate to a small
FineWeb-Edu pilot. Keep the byte-level tokenizer, model size, context length,
optimizer, and schedule unchanged so this tests the architecture rather than a
new training stack.

Use:

```text
dataset: HuggingFaceFW/fineweb-edu
config:  sample-10BT
split:   train
column:  text
```

The source dataset is train-only for this pilot, so validation is carved
deterministically from the beginning of the train stream, then training is
written from the remaining stream without overlap.

## FineWeb-Edu Data Prep

Full 1B-token pilot:

```bash
python experiments/train_tinystories.py \
  --prepare_data \
  --data_dir data/fineweb_edu_1b_byte \
  --dataset_name HuggingFaceFW/fineweb-edu \
  --dataset_config sample-10BT \
  --train_split train \
  --validation_split none \
  --text_column text \
  --max_validation_tokens 10000000 \
  --max_train_tokens 1000000000 \
  --device cuda \
  --max_steps 1 \
  --eval_iters 1
```

DGX smoke slice:

```bash
python experiments/train_tinystories.py \
  --prepare_data \
  --data_dir data/fineweb_edu_smoke_byte \
  --dataset_name HuggingFaceFW/fineweb-edu \
  --dataset_config sample-10BT \
  --train_split train \
  --validation_split none \
  --text_column text \
  --max_validation_tokens 1000000 \
  --max_train_tokens 20000000 \
  --device cuda \
  --max_steps 1 \
  --eval_iters 1
```

The generated `metadata.json` records dataset/config/splits, text column,
token counts, token caps, and whether validation was carved from the train
stream.

## FineWeb-Edu Matched Runs

RoPE baseline:

```bash
python experiments/train_tinystories.py \
  --data_dir data/fineweb_edu_1b_byte \
  --out_dir runs/fineweb_edu_1b_rope_1337 \
  --device cuda \
  --seed 1337 \
  --positional rope \
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

Path candidate:

```bash
python experiments/train_tinystories.py \
  --data_dir data/fineweb_edu_1b_byte \
  --out_dir runs/fineweb_edu_1b_path_last1_1337 \
  --device cuda \
  --seed 1337 \
  --positional path_hybrid \
  --path_layer_start 5 \
  --path_heads 1 \
  --path_blocks 4 \
  --path_angle_scale 0.05 \
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

For the first DGX smoke, run both models with the smoke data dir for `1000`
steps before launching the full 1B-token pilot.

## FineWeb-Edu Evaluation

Analyze paired runs:

```bash
python experiments/analyze_metrics.py runs \
  --paired \
  --baseline_group rope \
  --candidate_group path_last1
```

Promotion criteria:

- `path_last1` best validation loss is non-negative versus paired RoPE, or
  within `-0.05%` with noticeably better samples.
- Throughput ratio is at least `0.85x`.
- `path_gate` remains active above `0.01`.

If seed `1337` is positive or near parity, run paired seeds `1431` and `2327`
before changing architecture.
