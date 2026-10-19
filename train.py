#!/usr/bin/env python3
"""Command-line entry point for the model training framework.

Usage:
    python train.py --config configs/example.yaml
    python train.py --config configs/example.yaml --output_dir my_output --experiment my_exp
"""
import argparse
import sys
import os

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_trainer.config import Config
from model_trainer.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ML/DL models on local parquet factor data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", "-c", required=True,
        help="Path to YAML configuration file (see configs/example.yaml).",
    )
    # Optional overrides
    p.add_argument("--output_dir",      default=None, help="Override config output.save_dir")
    p.add_argument("--experiment",      default=None, help="Override config output.experiment_name")
    p.add_argument("--device",          default=None, help="Override training.device (cpu/cuda/mps/auto)")
    p.add_argument("--normalize",       default=None, help="Override data.normalize (rolling/global/none)")
    return p.parse_args()


def main() -> None:
    print(f'Current working directory: {os.getcwd()}')
    args = parse_args()

    print(f"Loading config from: {args.config}")
    cfg = Config.from_yaml(args.config)

    # Apply CLI overrides
    if args.output_dir:
        cfg.output.save_dir = args.output_dir
    if args.experiment:
        cfg.output.experiment_name = args.experiment
    if args.device:
        cfg.training.device = args.device
    if args.normalize:
        cfg.data.normalize = args.normalize

    trainer = Trainer(cfg)
    summary = trainer.run()

    print("\nDone.")
    return summary


if __name__ == "__main__":
    main()
