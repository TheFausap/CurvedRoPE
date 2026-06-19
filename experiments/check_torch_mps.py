from __future__ import annotations

import torch


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"mps built: {torch.backends.mps.is_built()}")
    print(f"mps available: {torch.backends.mps.is_available()}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x = torch.randn(4, 4, device=device)
    y = x @ x
    print(f"device: {y.device}")
    print(f"mean: {float(y.mean().cpu()):.6f}")


if __name__ == "__main__":
    main()

