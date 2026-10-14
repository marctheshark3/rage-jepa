#!/usr/bin/env python3
"""Train a fresh model using the public numerical datasets and original split."""
import argparse
from pathlib import Path
import torch
from rage_mill.jepa import JEPAConfig
from rage_mill.jepa_data import load_trajectories
from rage_mill.jepa_train import TrainConfig, train

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=200)
    args = p.parse_args()
    if args.out.exists():
        p.error("Choose a fresh output directory; frozen artifacts are never replaced")
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1] / "artifacts/npole10/jepa/data"
    training = {"npole10": load_trajectories(root / "npole10_train.npz")}
    evaluation = {"npole10": load_trajectories(root / "npole10_evaluation.npz")}
    train(training, evaluation, args.out,
          TrainConfig(epochs=args.epochs, seed=2026101310),
          JEPAConfig(latent_dim=64, hidden_dim=128), args.device)

if __name__ == "__main__":
    main()
