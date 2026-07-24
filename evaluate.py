import argparse

from reid import ReIDTrainer, TrainingConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="cfg.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--triplets", type=int, default=512)
    args = parser.parse_args()
    cfg = TrainingConfig(args.cfg)
    cfg.set_config("pretrained_model_path", args.model)
    loss, positive_distance, negative_distance = ReIDTrainer(cfg).evaluate(args.triplets)
    print(f"loss={loss:.4f} d_pos={positive_distance:.4f} "
          f"d_neg={negative_distance:.4f}")
