from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass
class RunSummary:
    name: str
    path: Path
    best_validation_loss: float | None
    best_step: int | None
    final_validation_loss: float | None
    final_step: int | None
    train_loss_at_final_eval: float | None
    mean_tokens_per_second: float | None
    final_path_gate: float | None
    eval_count: int
    train_count: int


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Could not parse {path}:{line_number}: {exc}") from exc
    return rows


def discover_run_dirs(paths: list[Path]) -> list[Path]:
    run_dirs = []
    for path in paths:
        if (path / "metrics.jsonl").exists():
            run_dirs.append(path)
            continue
        if path.is_dir():
            run_dirs.extend(sorted(child for child in path.iterdir() if (child / "metrics.jsonl").exists()))
    seen = set()
    unique = []
    for run_dir in run_dirs:
        resolved = run_dir.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(run_dir)
    return unique


def summarize_run(run_dir: Path) -> RunSummary:
    metrics_path = run_dir / "metrics.jsonl"
    rows = load_jsonl(metrics_path)
    eval_rows = [row for row in rows if row.get("event") == "eval"]
    train_rows = [row for row in rows if row.get("event") == "train"]

    best_validation_loss = None
    best_step = None
    final_validation_loss = None
    final_step = None
    train_loss_at_final_eval = None

    valid_eval_rows = [
        row
        for row in eval_rows
        if isinstance(row.get("validation_loss"), int | float)
        and math.isfinite(float(row["validation_loss"]))
    ]
    if valid_eval_rows:
        best_row = min(valid_eval_rows, key=lambda row: float(row["validation_loss"]))
        final_row = max(valid_eval_rows, key=lambda row: int(row.get("step", -1)))
        best_validation_loss = float(best_row["validation_loss"])
        best_step = int(best_row.get("step", -1))
        final_validation_loss = float(final_row["validation_loss"])
        final_step = int(final_row.get("step", -1))
        if isinstance(final_row.get("train_loss"), int | float):
            train_loss_at_final_eval = float(final_row["train_loss"])

    token_rates = [
        float(row["tokens_per_second"])
        for row in train_rows
        if isinstance(row.get("tokens_per_second"), int | float)
        and math.isfinite(float(row["tokens_per_second"]))
    ]
    path_gates = [
        float(row["path_gate"])
        for row in train_rows
        if isinstance(row.get("path_gate"), int | float)
        and math.isfinite(float(row["path_gate"]))
    ]

    return RunSummary(
        name=run_dir.name,
        path=run_dir,
        best_validation_loss=best_validation_loss,
        best_step=best_step,
        final_validation_loss=final_validation_loss,
        final_step=final_step,
        train_loss_at_final_eval=train_loss_at_final_eval,
        mean_tokens_per_second=mean(token_rates) if token_rates else None,
        final_path_gate=path_gates[-1] if path_gates else None,
        eval_count=len(eval_rows),
        train_count=len(train_rows),
    )


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def format_int(value: int | None) -> str:
    if value is None:
        return "-"
    return str(value)


def infer_baseline_and_candidate(summaries: list[RunSummary]) -> tuple[RunSummary | None, RunSummary | None]:
    baseline = next((run for run in summaries if "rope" in run.name and "path" not in run.name), None)
    candidate = next((run for run in summaries if "path" in run.name), None)
    if baseline is None and summaries:
        baseline = summaries[0]
    if candidate is None and len(summaries) > 1:
        candidate = summaries[1]
    return baseline, candidate


def verdict(
    baseline: RunSummary | None,
    candidate: RunSummary | None,
    *,
    min_loss_improvement: float,
    max_throughput_penalty: float,
    min_path_gate: float,
) -> tuple[str, list[str]]:
    if baseline is None or candidate is None:
        return "INSUFFICIENT", ["Need at least a baseline run and a candidate run."]
    if baseline.best_validation_loss is None or candidate.best_validation_loss is None:
        return "INSUFFICIENT", ["Both runs need eval events with validation_loss."]

    reasons = []
    loss_delta = baseline.best_validation_loss - candidate.best_validation_loss
    relative_improvement = loss_delta / baseline.best_validation_loss
    reasons.append(
        f"best validation loss delta: {loss_delta:+.4f} "
        f"({relative_improvement * 100:+.2f}% vs baseline)"
    )

    throughput_ratio = None
    if baseline.mean_tokens_per_second and candidate.mean_tokens_per_second:
        throughput_ratio = candidate.mean_tokens_per_second / baseline.mean_tokens_per_second
        reasons.append(f"throughput ratio: {throughput_ratio:.3f}x candidate/baseline")
    else:
        reasons.append("throughput ratio: unavailable")

    if candidate.final_path_gate is not None:
        reasons.append(f"final path gate: {candidate.final_path_gate:.4f}")
    else:
        reasons.append("final path gate: unavailable")

    clears_loss = relative_improvement >= min_loss_improvement
    close_loss = relative_improvement > -0.002
    clears_speed = throughput_ratio is None or throughput_ratio >= (1.0 - max_throughput_penalty)
    gate_active = candidate.final_path_gate is None or candidate.final_path_gate >= min_path_gate

    if clears_loss and clears_speed and gate_active:
        return "PROMOTE", reasons
    if close_loss and clears_speed and gate_active:
        return "PROMISING", reasons
    if not clears_speed:
        reasons.append("candidate is too slow for the current promotion threshold")
    if not gate_active:
        reasons.append("path gate did not move enough to show the mechanism is being used")
    return "HOLD", reasons


def print_table(summaries: list[RunSummary]) -> None:
    headers = [
        "run",
        "best_val",
        "best_step",
        "final_val",
        "final_step",
        "tok/s",
        "path_gate",
        "evals",
    ]
    rows = []
    for run in summaries:
        rows.append(
            [
                run.name,
                format_float(run.best_validation_loss),
                format_int(run.best_step),
                format_float(run.final_validation_loss),
                format_int(run.final_step),
                format_float(run.mean_tokens_per_second, digits=0),
                format_float(run.final_path_gate),
                str(run.eval_count),
            ]
        )
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row, strict=True)]

    print(" ".join(header.ljust(width) for header, width in zip(headers, widths, strict=True)))
    print(" ".join("-" * width for width in widths))
    for row in rows:
        print(" ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)))


def maybe_plot(summaries: list[RunSummary], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(f"\nPlot skipped: matplotlib is not installed. Text verdict is still complete.")
        return

    fig, (loss_axis, speed_axis) = plt.subplots(2, 1, figsize=(9, 7), sharex=False)

    for run in summaries:
        rows = load_jsonl(run.path / "metrics.jsonl")
        eval_rows = [row for row in rows if row.get("event") == "eval"]
        train_rows = [row for row in rows if row.get("event") == "train"]
        eval_steps = [int(row["step"]) for row in eval_rows if "step" in row and "validation_loss" in row]
        val_losses = [float(row["validation_loss"]) for row in eval_rows if "step" in row and "validation_loss" in row]
        if eval_steps:
            loss_axis.plot(eval_steps, val_losses, marker="o", linewidth=1.5, label=run.name)

        train_steps = [int(row["step"]) for row in train_rows if "step" in row and "tokens_per_second" in row]
        speeds = [float(row["tokens_per_second"]) for row in train_rows if "step" in row and "tokens_per_second" in row]
        if train_steps:
            speed_axis.plot(train_steps, speeds, linewidth=1.0, label=run.name)

    loss_axis.set_title("Validation Loss")
    loss_axis.set_xlabel("step")
    loss_axis.set_ylabel("loss")
    loss_axis.grid(True, alpha=0.25)
    loss_axis.legend()

    speed_axis.set_title("Throughput")
    speed_axis.set_xlabel("step")
    speed_axis.set_ylabel("tokens/sec")
    speed_axis.grid(True, alpha=0.25)
    speed_axis.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    print(f"\nPlot written to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze TinyStories metrics.jsonl runs.")
    parser.add_argument("paths", nargs="+", help="Run directories or parent directories containing metrics.jsonl")
    parser.add_argument("--baseline", help="Baseline run directory name or path substring")
    parser.add_argument("--candidate", help="Candidate run directory name or path substring")
    parser.add_argument("--plot", type=Path, help="Optional PNG output path")
    parser.add_argument("--min_loss_improvement", type=float, default=0.005)
    parser.add_argument("--max_throughput_penalty", type=float, default=0.25)
    parser.add_argument("--min_path_gate", type=float, default=0.01)
    return parser.parse_args()


def match_run(summaries: list[RunSummary], pattern: str | None) -> RunSummary | None:
    if pattern is None:
        return None
    return next((run for run in summaries if pattern in run.name or pattern in str(run.path)), None)


def main() -> None:
    args = parse_args()
    run_dirs = discover_run_dirs([Path(path) for path in args.paths])
    if not run_dirs:
        raise SystemExit("No metrics.jsonl files found.")

    summaries = sorted((summarize_run(run_dir) for run_dir in run_dirs), key=lambda run: run.name)
    print_table(summaries)

    inferred_baseline, inferred_candidate = infer_baseline_and_candidate(summaries)
    baseline = match_run(summaries, args.baseline) or inferred_baseline
    candidate = match_run(summaries, args.candidate) or inferred_candidate
    decision, reasons = verdict(
        baseline,
        candidate,
        min_loss_improvement=args.min_loss_improvement,
        max_throughput_penalty=args.max_throughput_penalty,
        min_path_gate=args.min_path_gate,
    )

    print()
    print(f"Baseline:  {baseline.name if baseline else '-'}")
    print(f"Candidate: {candidate.name if candidate else '-'}")
    print(f"Verdict:   {decision}")
    for reason in reasons:
        print(f"- {reason}")

    if args.plot:
        maybe_plot(summaries, args.plot)


if __name__ == "__main__":
    main()

