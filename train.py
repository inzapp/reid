import argparse

from reid import ReIDTrainer, TrainingConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="cfg/cfg.yaml")
    parser.add_argument(
        "--checkpoint-logs-only",
        action="store_true",
        help=("print training progress only at checkpoint_interval instead of "
              "after every iteration"),
    )
    args = parser.parse_args()
    ReIDTrainer(TrainingConfig(args.cfg)).train(
        checkpoint_logs_only=args.checkpoint_logs_only)
