import argparse
import signal

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
    termination_signal = [None]

    def request_termination(signum, _frame):
        termination_signal[0] = signum

    # A mutable container lets the signal handler record the request without
    # running TensorFlow or checkpoint I/O from inside the handler itself.
    signal.signal(signal.SIGINT, request_termination)
    signal.signal(signal.SIGTERM, request_termination)

    trainer = ReIDTrainer(TrainingConfig(args.cfg))
    if termination_signal[0] is not None:
        raise SystemExit(128 + termination_signal[0])
    interrupted = trainer.train(
        checkpoint_logs_only=args.checkpoint_logs_only,
        should_stop=lambda: termination_signal[0] is not None,
    )
    if interrupted:
        raise SystemExit(128 + termination_signal[0])
