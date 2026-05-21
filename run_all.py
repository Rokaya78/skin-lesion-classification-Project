"""
Runs the full pipeline in order - training, evaluation, and comparison.
Use --skip_train if you just want to re-run evaluation on already trained models.

Usage:
    python run_all.py
    python run_all.py --epochs 30 --batch_size 32
    python run_all.py --skip_train
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import os


def run(cmd: list[str]):
    print(f"\n{'='*60}")
    print("CMD: " + " ".join(cmd))
    print("="*60)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",         default="data/HAM10000_metadata.csv")
    parser.add_argument("--img_dirs",    nargs="+",
                        default=["data/HAM10000_images_part1",
                                 "data/HAM10000_images_part2"])
    parser.add_argument("--epochs",          type=int, default=30)
    parser.add_argument("--baseline_epochs", type=int, default=40,
                        help="Baseline needs more epochs since it has no pretrained weights")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--ckpt_dir",    default="checkpoints")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--unfreeze5_epoch",  type=int, default=10)
    parser.add_argument("--unfreeze10_epoch", type=int, default=20)
    parser.add_argument("--skip_train",    action="store_true",
                        help="Skip training and only run evaluation and comparison")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip the scratch CNN and only run MobileNetV2 strategies")
    args = parser.parse_args()

    py  = sys.executable
    src = os.path.join(os.path.dirname(__file__), "src")

    # shared args passed to every script
    common = [
        "--csv",         args.csv,
        "--img_dirs",    *args.img_dirs,
        "--batch_size",  str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--ckpt_dir",    args.ckpt_dir,
        "--results_dir", args.results_dir,
    ]

    mobilenet_strategies = ["feature_extraction", "progressive", "full_finetune"]
    # full_finetune uses a smaller lr to avoid destroying pretrained features
    mobilenet_lrs = {
        "feature_extraction": "1e-3",
        "progressive":        "1e-3",
        "full_finetune":      "1e-4",
    }

    # Week 1 - baseline CNN from scratch
    if not args.skip_train and not args.skip_baseline:
        print("\n>>> Week 1: Training baseline CNN from scratch")
        run([
            py, os.path.join(src, "train_baseline.py"),
            "--epochs", str(args.baseline_epochs),
            *common,
        ])

    # Week 2-3 - MobileNetV2 transfer learning strategies
    if not args.skip_train:
        for strat in mobilenet_strategies:
            week = "Week 2" if strat == "feature_extraction" else "Week 3"
            print(f"\n>>> {week}: Training MobileNetV2 - {strat}")
            train_cmd = [
                py, os.path.join(src, "train.py"),
                "--strategy", strat,
                "--epochs",   str(args.epochs),
                "--lr",       mobilenet_lrs[strat],
                *common,
            ]
            if strat == "progressive":
                train_cmd += [
                    "--unfreeze5_epoch",  str(args.unfreeze5_epoch),
                    "--unfreeze10_epoch", str(args.unfreeze10_epoch),
                ]
            run(train_cmd)

    # Week 4 - evaluate all models
    print("\n>>> Week 4: Evaluating all models")
    all_strategies = ["baseline_cnn"] + mobilenet_strategies
    for strat in all_strategies:
        ckpt = os.path.join(args.ckpt_dir, f"{strat}_best.pth")
        if not os.path.isfile(ckpt):
            print(f"  [skip] No checkpoint for {strat}")
            continue
        run([
            py, os.path.join(src, "evaluate.py"),
            "--strategy", strat,
            *common,
        ])

    # Week 4 - generate comparison plots and table
    print("\n>>> Week 4: Generating comparison report")
    run([py, os.path.join(src, "compare.py"),
         "--results_dir", args.results_dir])

    print("\n" + "="*60)
    print("Done! Results saved to:", args.results_dir)
    print("="*60)


if __name__ == "__main__":
    main()
