import argparse

from reid import ReIDTrainer, TrainingConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="cfg/cfg.yaml")
    args = parser.parse_args()
    ReIDTrainer(TrainingConfig(args.cfg)).train()
