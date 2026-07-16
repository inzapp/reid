"""ReID training configuration and triplet-loss training loop."""

import os
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import re
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from data_loader import DataLoader
from model import Model


class TrainingConfig:
    DEFAULTS = {
        "devices": [], "pretrained_model_path": None, "model_name": "reid",
        "optimizer": "adam", "lr_policy": "cosine", "lrf": 0.05,
        "l2": 0.0005, "warm_up": 1000, "momentum": 0.9,
        "max_q_size": 256, "num_loader_workers": 4,
        "checkpoint_interval": 2000, "fix_seed": False,
        "embedding_dim": 512, "distance_margin": 0.3,
        "horizontal_flip_probability": 0.5,
        "random_erasing_probability": 0.5, "color_jitter": 0.15,
        "random_crop_padding": 10,
    }
    REQUIRED = ("train_data_path", "validation_data_path", "input_rows",
                "input_cols", "input_channels", "lr", "batch_size", "iterations")

    def __init__(self, cfg_path):
        self.cfg_path = str(cfg_path)
        with open(cfg_path, "r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
        missing = [key for key in self.REQUIRED if key not in loaded]
        if missing:
            raise ValueError("missing required config keys: " + ", ".join(missing))
        self._data = {**self.DEFAULTS, **loaded}
        for key, value in self._data.items():
            setattr(self, key, value)
        self._validate()

    def _validate(self):
        if self.input_channels not in (1, 3):
            raise ValueError("input_channels must be 1 or 3")
        if self.embedding_dim <= 0 or self.distance_margin <= 0:
            raise ValueError("embedding_dim and distance_margin must be positive")
        if self.batch_size <= 0 or self.max_q_size < self.batch_size:
            raise ValueError("max_q_size must be greater than or equal to batch_size")

    def set_config(self, key, value):
        self._data[key] = value
        setattr(self, key, value)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as stream:
            yaml.safe_dump(self._data, stream, sort_keys=False)

    def print_cfg(self):
        print(yaml.safe_dump(self._data, sort_keys=False))


class ReIDTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.fix_seed:
            random.seed(42)
            np.random.seed(42)
            tf.random.set_seed(42)

        self.strategy = self._get_strategy(cfg.devices)
        self.optimizer = self._get_optimizer()
        self.start_iteration = 0
        if cfg.pretrained_model_path:
            self.model = self._load_model(cfg.pretrained_model_path)
        else:
            self.model = Model(cfg).build(self.strategy, self.optimizer)
        self.train_loader = DataLoader(cfg, training=True)
        self.validation_loader = DataLoader(cfg, training=False)
        self.checkpoint_path = None
        self.best_loss = np.inf

    @staticmethod
    def _get_strategy(devices):
        if not devices:
            tf.config.set_visible_devices([], "GPU")
            return tf.distribute.get_strategy()
        physical = tf.config.list_physical_devices("GPU")
        invalid = [index for index in devices if index >= len(physical)]
        if invalid:
            raise ValueError(f"invalid GPU indices {invalid}; found {len(physical)} GPUs")
        tf.config.set_visible_devices([physical[index] for index in devices], "GPU")
        if len(devices) == 1:
            return tf.distribute.get_strategy()
        return tf.distribute.MirroredStrategy()

    def _get_optimizer(self):
        initial_lr = self.cfg.lr if self.cfg.lr_policy == "constant" else 0.0
        with self.strategy.scope():
            if self.cfg.optimizer.lower() == "sgd":
                return tf.keras.optimizers.SGD(initial_lr, momentum=self.cfg.momentum,
                                               nesterov=True)
            if self.cfg.optimizer.lower() == "adam":
                return tf.keras.optimizers.Adam(initial_lr, beta_1=self.cfg.momentum)
        raise ValueError("optimizer must be 'sgd' or 'adam'")

    def _load_model(self, path):
        if path == "auto":
            candidates = sorted(Path("checkpoint").glob("*/last_*_iter.h5"),
                                key=lambda item: item.stat().st_mtime)
            if not candidates:
                raise FileNotFoundError("no automatic checkpoint found")
            path = str(candidates[-1])
            self.cfg.set_config("pretrained_model_path", path)
        with self.strategy.scope():
            model = tf.keras.models.load_model(path, compile=False)
            model.compile(optimizer=self.optimizer)
        if model.output_shape[-1] != self.cfg.embedding_dim:
            raise ValueError("checkpoint embedding dimension does not match cfg.yaml")
        match = re.search(r"_(\d+)_iter", os.path.basename(path))
        self.start_iteration = int(match.group(1)) if match else 0
        return model

    @staticmethod
    def _distances(anchor_embedding, positive_embedding, negative_embedding):
        positive = tf.norm(anchor_embedding - positive_embedding, axis=1)
        negative = tf.norm(anchor_embedding - negative_embedding, axis=1)
        return positive, negative

    @tf.function
    def _train_step(self, anchor, positive, negative):
        with tf.GradientTape() as tape:
            # Three calls share weights while keeping each role explicit.
            anchor_embedding = self.model(anchor, training=True)
            positive_embedding = self.model(positive, training=True)
            negative_embedding = self.model(negative, training=True)
            positive_distance, negative_distance = self._distances(
                anchor_embedding, positive_embedding, negative_embedding)
            triplet_loss = tf.reduce_mean(tf.maximum(
                positive_distance - negative_distance + self.cfg.distance_margin, 0.0))
            regularization = (tf.add_n(self.model.losses)
                              if self.model.losses else tf.constant(0.0))
            loss = triplet_loss + regularization
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, tf.reduce_mean(positive_distance), tf.reduce_mean(negative_distance)

    def _set_learning_rate(self, iteration):
        warm_up = (int(self.cfg.iterations * self.cfg.warm_up)
                   if isinstance(self.cfg.warm_up, float) and self.cfg.warm_up <= 1.0
                   else int(self.cfg.warm_up))
        if warm_up and iteration < warm_up:
            lr = self.cfg.lr * (iteration + 1) / warm_up
        elif self.cfg.lr_policy == "constant":
            lr = self.cfg.lr
        elif self.cfg.lr_policy == "step":
            fraction = iteration / self.cfg.iterations
            lr = self.cfg.lr * (self.cfg.lrf ** (1 if fraction >= 0.8 else 0))
        elif self.cfg.lr_policy == "cosine":
            progress = (iteration - warm_up) / max(1, self.cfg.iterations - warm_up)
            cosine = 0.5 * (1.0 + np.cos(np.pi * np.clip(progress, 0.0, 1.0)))
            lr = self.cfg.lr * (self.cfg.lrf + (1.0 - self.cfg.lrf) * cosine)
        else:
            raise ValueError("lr_policy must be constant, step, or cosine")
        self.optimizer.learning_rate.assign(lr)

    def _init_checkpoint_dir(self):
        root = Path("checkpoint")
        candidate = root / self.cfg.model_name
        index = 1
        while candidate.exists():
            candidate = root / f"{self.cfg.model_name}_{index}"
            index += 1
        candidate.mkdir(parents=True)
        self.checkpoint_path = candidate
        self.cfg.save(candidate / "cfg.yaml")

    def _save(self, iteration, best=False, loss=None):
        prefix = "best" if best else "last"
        suffix = f"_loss_{loss:.4f}" if loss is not None else ""
        path = self.checkpoint_path / f"{prefix}_{iteration}_iter{suffix}.h5"
        for old in self.checkpoint_path.glob(f"{prefix}_*.h5"):
            shutil.rmtree(old) if old.is_dir() else old.unlink()
        self.model.save(path, include_optimizer=False)
        # Save again in case runtime config changed (e.g. resolved auto checkpoint).
        self.cfg.save(self.checkpoint_path / "cfg.yaml")
        return path

    def evaluate(self, triplet_count=None):
        count = triplet_count or max(self.cfg.batch_size, 128)
        triplets = self.validation_loader.sample_validation_triplets(count)
        losses, positive_distances, negative_distances = [], [], []
        for start in range(0, len(triplets), self.cfg.batch_size):
            batch = triplets[start:start + self.cfg.batch_size]
            anchor, positive, negative = (np.stack(items) for items in zip(*batch))
            ae = self.model(anchor, training=False)
            pe = self.model(positive, training=False)
            ne = self.model(negative, training=False)
            dp, dn = self._distances(ae, pe, ne)
            losses.extend(tf.maximum(dp - dn + self.cfg.distance_margin, 0.0).numpy())
            positive_distances.extend(dp.numpy())
            negative_distances.extend(dn.numpy())
        return (float(np.mean(losses)), float(np.mean(positive_distances)),
                float(np.mean(negative_distances)))

    def train(self):
        self.model.summary()
        self.cfg.print_cfg()
        print(f"train: {len(self.train_loader.data_paths)} images, "
              f"{len(self.train_loader.identities)} identities")
        self._init_checkpoint_dir()
        self.train_loader.start()
        try:
            for iteration in range(self.start_iteration, self.cfg.iterations):
                self._set_learning_rate(iteration)
                anchor, positive, negative = self.train_loader.load()
                loss, dp, dn = self._train_step(anchor, positive, negative)
                current = iteration + 1
                print(f"\r[{current}/{self.cfg.iterations}] loss={loss:.4f} "
                      f"d_pos={dp:.4f} d_neg={dn:.4f}", end="", flush=True)
                if current % 2000 == 0 or current == self.cfg.iterations:
                    self._save(current)
                if current % self.cfg.checkpoint_interval == 0:
                    validation_loss, val_dp, val_dn = self.evaluate()
                    print(f"\nvalidation loss={validation_loss:.4f} "
                          f"d_pos={val_dp:.4f} d_neg={val_dn:.4f}")
                    if validation_loss < self.best_loss:
                        self.best_loss = validation_loss
                        self._save(current, best=True, loss=validation_loss)
        finally:
            self.train_loader.stop()
        print("\ntraining completed")
