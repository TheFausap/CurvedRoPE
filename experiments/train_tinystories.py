from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EOS_TOKEN = 256
VOCAB_SIZE = 257


@dataclass
class GPTConfig:
    vocab_size: int = VOCAB_SIZE
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    positional: str = "rope"
    path_heads: int = 2
    path_blocks: int = 0
    path_layer_start: int = 0
    path_layer_interval: int = 1
    path_angle_scale: float = 0.05
    path_translation_scale: float = 0.05


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_text(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8).astype(np.uint16)


def write_tokens(handle, tokens: np.ndarray, max_tokens: int | None, token_count: int) -> tuple[int, bool]:
    remaining = None if max_tokens is None else max_tokens - token_count
    if remaining is not None and remaining <= 0:
        return token_count, True
    if remaining is not None and len(tokens) > remaining:
        tokens = tokens[:remaining]
    tokens.tofile(handle)
    token_count += len(tokens)
    return token_count, max_tokens is not None and token_count >= max_tokens


def prepare_dataset(
    data_dir: Path,
    *,
    dataset_name: str,
    dataset_config: str | None,
    train_split: str,
    validation_split: str,
    text_column: str,
    max_train_stories: int | None = None,
    max_validation_stories: int | None = None,
    max_train_tokens: int | None = None,
    max_validation_tokens: int | None = None,
) -> None:
    try:
        from datasets import load_dataset
        from tqdm.auto import tqdm
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Dataset preparation needs `datasets` and `tqdm`. Install "
            "`requirements-train.txt` in the target environment."
        ) from exc

    data_dir.mkdir(parents=True, exist_ok=True)

    def load_stream(split: str):
        if dataset_config:
            return load_dataset(dataset_name, dataset_config, split=split, streaming=True)
        return load_dataset(dataset_name, split=split, streaming=True)

    def row_tokens(row: dict) -> np.ndarray:
        if text_column not in row:
            available = ", ".join(sorted(row.keys()))
            raise KeyError(f"Column {text_column!r} not found. Available columns: {available}")
        tokens = encode_text(str(row[text_column]))
        return np.concatenate([tokens, np.array([EOS_TOKEN], dtype=np.uint16)])

    def write_split(split: str, output_name: str, max_stories: int | None, max_tokens: int | None) -> int:
        output_path = data_dir / f"{split}.bin"
        if output_name != split:
            output_path = data_dir / f"{output_name}.bin"
        if output_path.exists():
            print(f"{output_path} already exists; skipping")
            return int(output_path.stat().st_size // np.dtype(np.uint16).itemsize)

        dataset = load_stream(split)
        token_count = 0
        with output_path.open("wb") as handle:
            iterator = tqdm(dataset, desc=f"writing {output_name}", unit="row")
            for index, row in enumerate(iterator):
                if max_stories is not None and index >= max_stories:
                    break
                token_count, done = write_tokens(handle, row_tokens(row), max_tokens, token_count)
                iterator.set_postfix(tokens=token_count)
                if done:
                    break
        return token_count

    validation_carved_from_train = validation_split.lower() == "none"

    if validation_carved_from_train:
        train_path = data_dir / "train.bin"
        validation_path = data_dir / "validation.bin"
        if max_validation_stories is None and max_validation_tokens is None:
            raise ValueError(
                "When --validation_split none, set --max_validation_tokens or "
                "--max_validation_stories so validation carving is finite."
            )
        if train_path.exists() and validation_path.exists():
            train_tokens = int(train_path.stat().st_size // np.dtype(np.uint16).itemsize)
            validation_tokens = int(validation_path.stat().st_size // np.dtype(np.uint16).itemsize)
            print(f"{train_path} and {validation_path} already exist; skipping")
        elif train_path.exists() or validation_path.exists():
            raise FileExistsError(
                f"Found only one of {train_path} and {validation_path}. "
                "Remove the partial files or choose a fresh data_dir."
            )
        else:
            dataset = load_stream(train_split)
            train_tokens = 0
            validation_tokens = 0
            validation_stories = 0
            train_stories = 0
            with validation_path.open("wb") as validation_handle, train_path.open("wb") as train_handle:
                iterator = tqdm(dataset, desc="writing validation/train", unit="row")
                for row in iterator:
                    tokens = row_tokens(row)
                    validation_story_ok = (
                        max_validation_stories is None
                        or validation_stories < max_validation_stories
                    )
                    validation_token_ok = (
                        max_validation_tokens is None
                        or validation_tokens < max_validation_tokens
                    )
                    if validation_story_ok and validation_token_ok:
                        validation_tokens, validation_done = write_tokens(
                            validation_handle,
                            tokens,
                            max_validation_tokens,
                            validation_tokens,
                        )
                        validation_stories += 1
                    else:
                        if max_train_stories is not None and train_stories >= max_train_stories:
                            break
                        train_tokens, train_done = write_tokens(
                            train_handle,
                            tokens,
                            max_train_tokens,
                            train_tokens,
                        )
                        train_stories += 1
                        if train_done:
                            break
                    iterator.set_postfix(train_tokens=train_tokens, validation_tokens=validation_tokens)
    else:
        train_tokens = write_split(train_split, "train", max_train_stories, max_train_tokens)
        validation_tokens = write_split(
            validation_split,
            "validation",
            max_validation_stories,
            max_validation_tokens,
        )

    metadata = {
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "train_split": train_split,
        "validation_split": validation_split,
        "text_column": text_column,
        "vocab_size": VOCAB_SIZE,
        "dtype": "uint16",
        "train_tokens": train_tokens,
        "validation_tokens": validation_tokens,
        "validation_carved_from_train": validation_carved_from_train,
        "max_train_tokens": max_train_tokens,
        "max_validation_tokens": max_validation_tokens,
        "max_train_stories": max_train_stories,
        "max_validation_stories": max_validation_stories,
    }
    (data_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


class BinaryTokenDataset:
    def __init__(self, path: Path, block_size: int, device: torch.device):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.device = device
        if len(self.data) <= block_size + 1:
            raise ValueError(f"{path} is too small for block_size={block_size}")

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        starts = np.random.randint(0, len(self.data) - self.block_size - 1, size=batch_size)
        x = np.stack([self.data[start : start + self.block_size] for start in starts])
        y = np.stack([self.data[start + 1 : start + self.block_size + 1] for start in starts])
        return (
            torch.from_numpy(x.astype(np.int64)).to(self.device, non_blocking=True),
            torch.from_numpy(y.astype(np.int64)).to(self.device, non_blocking=True),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        self.positional = config.positional
        self.path_heads = min(config.path_heads, config.n_head)
        self.path_blocks = config.path_blocks
        self.path_width = (self.head_dim // 3) * 3
        if self.path_blocks > 0:
            self.path_width = min(self.path_width, self.path_blocks * 3)
        self.path_angle_scale = config.path_angle_scale
        self.path_translation_scale = config.path_translation_scale

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                / self.head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.path_proj = None
        self.path_gate = None
        self.path_translation_proj = None
        self.path_translation_gate = None
        if self.positional in {"path_hybrid", "path_affine"}:
            self.path_proj = nn.Linear(config.n_embd, self.path_heads * 3, bias=False)
            self.path_gate = nn.Parameter(torch.full((self.path_heads,), -6.0))
        if self.positional == "path_affine":
            self.path_translation_proj = nn.Linear(
                config.n_embd,
                self.path_heads * self.path_width,
                bias=False,
            )
            self.path_translation_gate = nn.Parameter(torch.full((self.path_heads,), -6.0))

    def _rope_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("t,d->td", positions, self.inv_freq.to(device))
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        cos = emb.cos().to(dtype).view(1, 1, seq_len, self.head_dim)
        sin = emb.sin().to(dtype).view(1, 1, seq_len, self.head_dim)
        return cos, sin

    def _path_prefixes(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.path_proj is None:
            raise RuntimeError("path_proj is not initialized")

        batch, seq_len, _ = x.shape
        angles = torch.tanh(self.path_proj(x)).view(batch, seq_len, self.path_heads, 3)
        angles = angles * self.path_angle_scale
        ax, ay, az = angles.unbind(dim=-1)

        zeros = torch.zeros_like(ax)
        ones = torch.ones_like(ax)

        cx, sx = ax.cos(), ax.sin()
        cy, sy = ay.cos(), ay.sin()
        cz, sz = az.cos(), az.sin()

        rx = torch.stack(
            [
                torch.stack([ones, zeros, zeros], dim=-1),
                torch.stack([zeros, cx, -sx], dim=-1),
                torch.stack([zeros, sx, cx], dim=-1),
            ],
            dim=-2,
        )
        ry = torch.stack(
            [
                torch.stack([cy, zeros, sy], dim=-1),
                torch.stack([zeros, ones, zeros], dim=-1),
                torch.stack([-sy, zeros, cy], dim=-1),
            ],
            dim=-2,
        )
        rz = torch.stack(
            [
                torch.stack([cz, -sz, zeros], dim=-1),
                torch.stack([sz, cz, zeros], dim=-1),
                torch.stack([zeros, zeros, ones], dim=-1),
            ],
            dim=-2,
        )
        steps = rz @ ry @ rx

        eye = torch.eye(3, device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
        prefix = eye.expand(batch, self.path_heads, 3, 3).clone()
        translation_prefix = None
        if self.path_translation_proj is not None and self.path_width > 0:
            translations = torch.tanh(self.path_translation_proj(x))
            translations = translations.view(batch, seq_len, self.path_heads, self.path_width // 3, 3)
            translations = translations * self.path_translation_scale
            translation_prefix = torch.zeros(
                batch,
                self.path_heads,
                self.path_width // 3,
                3,
                device=x.device,
                dtype=x.dtype,
            )

        prefixes = []
        translation_prefixes = []
        for index in range(seq_len):
            prefix = steps[:, index] @ prefix
            prefixes.append(prefix)
            if translation_prefix is not None:
                translation_prefix = (
                    torch.einsum("bhij,bhkj->bhki", steps[:, index], translation_prefix)
                    + translations[:, index]
                )
                translation_prefixes.append(translation_prefix)

        prefix_tensor = torch.stack(prefixes, dim=2)
        if translation_prefix is None:
            return prefix_tensor, None
        return prefix_tensor, torch.stack(translation_prefixes, dim=2)

    def _apply_path(
        self,
        tensor: torch.Tensor,
        prefixes: torch.Tensor,
        translations: torch.Tensor | None,
    ) -> torch.Tensor:
        path_width = self.path_width
        if path_width == 0 or self.path_heads == 0:
            return tensor

        selected = tensor[:, : self.path_heads, :, :path_width]
        batch, heads, seq_len, _ = selected.shape
        blocks = selected.view(batch, heads, seq_len, path_width // 3, 3)
        rotated = torch.einsum("bhtij,bhtkj->bhtki", prefixes, blocks)
        out = tensor.clone()
        gate = torch.sigmoid(self.path_gate).to(tensor.dtype).view(1, self.path_heads, 1, 1)
        rotated = rotated.reshape(batch, heads, seq_len, path_width)
        mixed = selected + gate * (rotated - selected)
        if translations is not None and self.path_translation_gate is not None:
            translation_gate = torch.sigmoid(self.path_translation_gate)
            translation_gate = translation_gate.to(tensor.dtype).view(1, self.path_heads, 1, 1)
            mixed = mixed + translation_gate * translations.reshape(batch, heads, seq_len, path_width)
        out[:, : self.path_heads, :, :path_width] = mixed
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        q, k, v = self.c_attn(x).split(channels, dim=2)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = self._rope_cache(seq_len, x.device, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.positional in {"path_hybrid", "path_affine"}:
            prefixes, translations = self._path_prefixes(x)
            q = self._apply_path(q, prefixes, translations)
            k = self._apply_path(k, prefixes, translations)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_index: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        use_path = (
            config.positional in {"path_hybrid", "path_affine"}
            and layer_index >= config.path_layer_start
            and (layer_index - config.path_layer_start) % config.path_layer_interval == 0
        )
        block_config = config if use_path else replace(config, positional="rope")
        self.attn = CausalSelfAttention(block_config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config, index) for index in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError("Cannot forward sequence longer than block_size")

        x = self.drop(self.wte(idx))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.9) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


def estimate_loss(
    model: GPT,
    train_data: BinaryTokenDataset,
    validation_data: BinaryTokenDataset,
    *,
    batch_size: int,
    eval_iters: int,
    amp_context,
) -> dict[str, float]:
    model.eval()
    losses = {}
    with torch.no_grad():
        for split, dataset in (("train", train_data), ("validation", validation_data)):
            values = []
            for _ in range(eval_iters):
                x, y = dataset.get_batch(batch_size)
                with amp_context:
                    _, loss = model(x, y)
                values.append(float(loss.item()))
            losses[split] = sum(values) / len(values)
    model.train()
    return losses


def decode_tokens(tokens: torch.Tensor) -> str:
    values = tokens.detach().cpu().tolist()
    byte_values = [value for value in values if 0 <= value < 256]
    return bytes(byte_values).decode("utf-8", errors="ignore")


def save_checkpoint(
    out_dir: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    config: GPTConfig,
    args: argparse.Namespace,
    step: int,
    best_validation_loss: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "args": vars(args),
        "step": step,
        "best_validation_loss": best_validation_loss,
    }
    torch.save(checkpoint, out_dir / "ckpt.pt")


def load_checkpoint(path: Path, device: torch.device):
    return torch.load(path, map_location=device)


def make_amp_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "float32":
        return contextlib.nullcontext()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    return torch.autocast(device_type="cuda", dtype=dtype)


def append_metrics(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"cuda device: {torch.cuda.get_device_name(device)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print(f"precision: {args.dtype}")

    data_dir = Path(args.data_dir)
    if args.prepare_data:
        dataset_config = args.dataset_config
        if dataset_config is not None and dataset_config.lower() == "none":
            dataset_config = None
        prepare_dataset(
            data_dir,
            dataset_name=args.dataset_name,
            dataset_config=dataset_config,
            train_split=args.train_split,
            validation_split=args.validation_split,
            text_column=args.text_column,
            max_train_stories=args.max_train_stories,
            max_validation_stories=args.max_validation_stories,
            max_train_tokens=args.max_train_tokens,
            max_validation_tokens=args.max_validation_tokens,
        )

    train_path = data_dir / "train.bin"
    validation_path = data_dir / "validation.bin"
    if not train_path.exists() or not validation_path.exists():
        raise SystemExit("Missing train.bin/validation.bin. Re-run with --prepare_data.")

    config = GPTConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        positional=args.positional,
        path_heads=args.path_heads,
        path_blocks=args.path_blocks,
        path_layer_start=args.path_layer_start,
        path_layer_interval=args.path_layer_interval,
        path_angle_scale=args.path_angle_scale,
        path_translation_scale=args.path_translation_scale,
    )

    out_dir = Path(args.out_dir)
    start_step = 0
    best_validation_loss = float("inf")
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    amp_context = make_amp_context(device, args.dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.dtype == "float16")

    checkpoint_path = out_dir / "ckpt.pt"
    if args.resume and checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"]) + 1
        best_validation_loss = float(checkpoint["best_validation_loss"])
        print(f"resumed from {checkpoint_path} at step {start_step}")

    train_data = BinaryTokenDataset(train_path, args.block_size, device)
    validation_data = BinaryTokenDataset(validation_path, args.block_size, device)

    model.train()
    running_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step, args.max_steps):
        learning_rate = args.learning_rate
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        total_loss = 0.0
        for micro_step in range(args.gradient_accumulation_steps):
            x, y = train_data.get_batch(args.batch_size)
            with amp_context:
                _, loss = model(x, y)
            loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            total_loss += float(loss.item())

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            elapsed = max(time.time() - running_start, 1e-6)
            tokens = (
                args.batch_size
                * args.block_size
                * args.gradient_accumulation_steps
                * max(step - start_step + 1, 1)
            )
            msg = (
                f"step {step:06d} | loss {total_loss:.4f} | "
                f"tok/s {tokens / elapsed:,.0f}"
            )
            path_gates = [
                torch.sigmoid(module.attn.path_gate).detach().mean().item()
                for module in model.blocks
                if module.attn.path_gate is not None
            ]
            affine_gates = [
                torch.sigmoid(module.attn.path_translation_gate).detach().mean().item()
                for module in model.blocks
                if module.attn.path_translation_gate is not None
            ]
            if path_gates:
                msg += f" | path_gate {sum(path_gates) / len(path_gates):.4f}"
            if affine_gates:
                msg += f" | affine_gate {sum(affine_gates) / len(affine_gates):.4f}"
            print(msg, flush=True)
            append_metrics(
                out_dir,
                {
                    "event": "train",
                    "step": step,
                    "loss": total_loss,
                    "tokens_per_second": tokens / elapsed,
                    "path_gate": sum(path_gates) / len(path_gates) if path_gates else None,
                    "affine_gate": sum(affine_gates) / len(affine_gates) if affine_gates else None,
                    "time": time.time(),
                },
            )

        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            losses = estimate_loss(
                model,
                train_data,
                validation_data,
                batch_size=args.eval_batch_size,
                eval_iters=args.eval_iters,
                amp_context=amp_context,
            )
            print(
                f"eval {step:06d} | train {losses['train']:.4f} | "
                f"validation {losses['validation']:.4f}",
                flush=True,
            )
            if losses["validation"] < best_validation_loss:
                best_validation_loss = losses["validation"]
                save_checkpoint(out_dir, model, optimizer, config, args, step, best_validation_loss)
                print(f"saved checkpoint: {out_dir / 'ckpt.pt'}", flush=True)
            append_metrics(
                out_dir,
                {
                    "event": "eval",
                    "step": step,
                    "train_loss": losses["train"],
                    "validation_loss": losses["validation"],
                    "best_validation_loss": best_validation_loss,
                    "time": time.time(),
                },
            )

        if args.sample_interval > 0 and step % args.sample_interval == 0:
            prompt = torch.tensor([[ord("O"), ord("n"), ord("c"), ord("e")]], device=device)
            sample = model.generate(prompt, max_new_tokens=args.sample_tokens)[0]
            print("--- sample ---")
            print(decode_tokens(sample))
            print("--------------", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny GPT on TinyStories.")
    parser.add_argument("--data_dir", default="data/tinystories_byte")
    parser.add_argument("--out_dir", default="runs/tinystories_rope")
    parser.add_argument("--prepare_data", action="store_true")
    parser.add_argument("--dataset_name", default="roneneldan/TinyStories")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--validation_split", default="validation")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--max_train_stories", type=int, default=None)
    parser.add_argument("--max_validation_stories", type=int, default=None)
    parser.add_argument("--max_train_tokens", type=int, default=None)
    parser.add_argument("--max_validation_tokens", type=int, default=None)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, mps, or cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--positional", choices=["rope", "path_hybrid", "path_affine"], default="rope")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--path_heads", type=int, default=2)
    parser.add_argument("--path_blocks", type=int, default=0, help="3D blocks per path head; 0 means all possible blocks")
    parser.add_argument("--path_layer_start", type=int, default=0, help="First layer index that uses path_hybrid")
    parser.add_argument("--path_layer_interval", type=int, default=1, help="Use path every N layers after path_layer_start")
    parser.add_argument("--path_angle_scale", type=float, default=0.05)
    parser.add_argument("--path_translation_scale", type=float, default=0.05)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=100)
    parser.add_argument("--sample_interval", type=int, default=1000)
    parser.add_argument("--sample_tokens", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
