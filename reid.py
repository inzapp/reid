"""ReID training configuration and contrastive-loss training loop."""

import os
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from data_loader import DataLoader
from metrics import verification_metrics
from model import Model


class TrainingConfig:
    DEFAULTS = {
        "devices": [], "pretrained_model_path": None, "model_name": "reid",
        "optimizer": "adam", "lr_policy": "cosine", "lrf": 0.05,
        "l2": 0.0005, "warm_up": 1000, "momentum": 0.9,
        "max_q_size": 256, "num_loader_workers": 4,
        "checkpoint_interval": 2000, "fix_seed": False,
        "validation_pair_count": 100000, "evaluation_batch_size": None,
        "evaluation_pair_chunk_size": 8192,
        "query_data_path": None,
        "embedding_dim": 512, "maximum_negative_distance": 2.0,
        "verification_threshold": None,
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
        if self.embedding_dim <= 0 or self.maximum_negative_distance <= 0:
            raise ValueError(
                "embedding_dim and maximum_negative_distance must be positive")
        if (self.verification_threshold is not None
                and self.verification_threshold <= 0):
            raise ValueError("verification_threshold must be positive or null")
        if self.batch_size <= 0 or self.max_q_size < self.batch_size:
            raise ValueError("max_q_size must be greater than or equal to batch_size")
        if self.validation_pair_count <= 0:
            raise ValueError("validation_pair_count must be positive")
        if self.evaluation_batch_size is not None and self.evaluation_batch_size <= 0:
            raise ValueError("evaluation_batch_size must be positive or null")
        if self.evaluation_pair_chunk_size <= 0:
            raise ValueError("evaluation_pair_chunk_size must be positive")

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

    def _contrastive_loss(self, positive_distance, negative_distance):
        positive_loss = tf.square(positive_distance)
        negative_loss = -tf.minimum(
            negative_distance, self.cfg.maximum_negative_distance)
        return positive_loss + negative_loss

    @tf.function
    def _train_step(self, anchor, positive, negative):
        with tf.GradientTape() as tape:
            # A single call gives all three roles the same BatchNorm statistics.
            embeddings = self.model(
                tf.concat((anchor, positive, negative), axis=0), training=True)
            anchor_embedding, positive_embedding, negative_embedding = tf.split(
                embeddings, 3, axis=0)
            positive_distance, negative_distance = self._distances(anchor_embedding, positive_embedding, negative_embedding)
            contrastive_loss = tf.reduce_mean(self._contrastive_loss(positive_distance, negative_distance))
            regularization = (tf.add_n(self.model.losses) if self.model.losses else tf.constant(0.0))
            loss = contrastive_loss + regularization
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

    def _embed_paths(self, paths):
        batch_size = self.cfg.evaluation_batch_size or self.cfg.batch_size
        batches = []
        for start in range(0, len(paths), batch_size):
            images = np.stack([
                self.validation_loader.load_image(path, augment=False)
                for path in paths[start:start + batch_size]
            ])
            batches.append(self.model(images, training=False).numpy())
        return np.concatenate(batches, axis=0)

    def _query_paths(self):
        if self.cfg.query_data_path is None:
            return []
        root = Path(self.cfg.query_data_path)
        if not root.is_dir():
            raise ValueError(f"query image directory does not exist: {root}")
        return sorted(str(path) for path in root.rglob("*")
                      if path.is_file()
                      and path.suffix in {".jpg", ".jpeg", ".JPG", ".JPEG",
                                          ".png", ".PNG"}
                      and self.validation_loader._has_valid_identity(str(path)))

    def _rank1_gallery_paths(self):
        root = Path(self.cfg.validation_data_path)
        paths = sorted(str(path) for path in root.rglob("*")
                       if path.is_file()
                       and path.suffix in {".jpg", ".jpeg", ".JPG", ".JPEG",
                                           ".png", ".PNG"})
        result = []
        for path in paths:
            try:
                identity = int(self.validation_loader.identity_from_path(path))
            except ValueError:
                continue
            if identity >= 1:
                result.append(path)
        return result

    def _market1501_rank1(self, query_embeddings, query_paths,
                          gallery_embeddings, gallery_paths):
        gallery_ids = np.asarray([
            self.validation_loader.identity_from_path(path)
            for path in gallery_paths])
        gallery_cameras = np.asarray([
            self.validation_loader.camera_from_path(path)
            for path in gallery_paths])
        gallery = tf.convert_to_tensor(gallery_embeddings)
        gallery_norm = tf.reduce_sum(tf.square(gallery), axis=1)[None, :]
        correct, valid_queries = 0, 0
        batch_size = self.cfg.evaluation_batch_size or self.cfg.batch_size
        for start in range(0, len(query_paths), batch_size):
            end = min(start + batch_size, len(query_paths))
            query = tf.convert_to_tensor(query_embeddings[start:end])
            query_norm = tf.reduce_sum(tf.square(query), axis=1)[:, None]
            distances = (query_norm + gallery_norm
                         - 2.0 * tf.matmul(query, gallery, transpose_b=True))
            distances = distances.numpy()
            for offset, path in enumerate(query_paths[start:end]):
                query_id = self.validation_loader.identity_from_path(path)
                query_camera = self.validation_loader.camera_from_path(path)
                valid = ~((gallery_ids == query_id)
                          & (gallery_cameras == query_camera))
                matches = valid & (gallery_ids == query_id)
                if not np.any(matches):
                    continue
                nearest = np.argmin(np.where(valid, distances[offset], np.inf))
                correct += int(gallery_ids[nearest] == query_id)
                valid_queries += 1
        if valid_queries == 0:
            raise ValueError("no query has a valid cross-camera gallery match")
        return correct / valid_queries

    def _sample_validation_indices(self, count):
        paths = self.validation_loader.data_paths
        indices_by_id = defaultdict(list)
        indices_by_id_and_camera = defaultdict(lambda: defaultdict(list))
        for index, path in enumerate(paths):
            identity = self.validation_loader.identity_from_path(path)
            camera = self.validation_loader.camera_from_path(path)
            indices_by_id[identity].append(index)
            indices_by_id_and_camera[identity][camera].append(index)

        has_camera_metadata = any(
            camera is not None
            for cameras in indices_by_id_and_camera.values()
            for camera in cameras)
        if has_camera_metadata:
            anchor_identities = [
                identity for identity, cameras in indices_by_id_and_camera.items()
                if len([camera for camera in cameras if camera is not None]) >= 2
            ]
        else:
            anchor_identities = [
                identity for identity, indices in indices_by_id.items()
                if len(indices) >= 2
            ]
        identities = list(indices_by_id)
        if not anchor_identities:
            raise ValueError(
                "validation requires an identity present in at least two cameras")

        rng = random.Random(0)
        anchor_indices = np.empty(count, dtype=np.int64)
        positive_indices = np.empty(count, dtype=np.int64)
        negative_indices = np.empty(count, dtype=np.int64)
        for item in range(count):
            anchor_id = rng.choice(anchor_identities)
            if has_camera_metadata:
                cameras = [camera for camera in indices_by_id_and_camera[anchor_id]
                           if camera is not None]
                anchor_camera, positive_camera = rng.sample(cameras, 2)
                anchor_indices[item] = rng.choice(
                    indices_by_id_and_camera[anchor_id][anchor_camera])
                positive_indices[item] = rng.choice(
                    indices_by_id_and_camera[anchor_id][positive_camera])
            else:
                anchor_indices[item], positive_indices[item] = rng.sample(
                    indices_by_id[anchor_id], 2)

            negative_id = rng.choice(identities)
            while negative_id == anchor_id:
                negative_id = rng.choice(identities)
            negative_indices[item] = rng.choice(indices_by_id[negative_id])
        return anchor_indices, positive_indices, negative_indices

    def evaluate(self, triplet_count=None):
        count = triplet_count or self.cfg.validation_pair_count
        gallery_paths = self.validation_loader.data_paths
        query_paths = self._query_paths()
        if query_paths:
            rank1_gallery_paths = self._rank1_gallery_paths()
            rank1_gallery_embeddings = self._embed_paths(rank1_gallery_paths)
            valid_gallery = np.asarray([
                int(self.validation_loader.identity_from_path(path)) >= 1
                for path in rank1_gallery_paths])
            embeddings = rank1_gallery_embeddings[valid_gallery]
            gallery_paths = [path for path, valid in zip(rank1_gallery_paths,
                                                         valid_gallery) if valid]
        else:
            rank1_gallery_paths = None
            rank1_gallery_embeddings = None
            embeddings = self._embed_paths(gallery_paths)
        anchor_indices, positive_indices, negative_indices = (
            self._sample_validation_indices(count))
        positive_distances = np.empty(count, dtype=np.float32)
        negative_distances = np.empty(count, dtype=np.float32)
        chunk_size = self.cfg.evaluation_pair_chunk_size
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            anchor = embeddings[anchor_indices[start:end]]
            positive = embeddings[positive_indices[start:end]]
            negative = embeddings[negative_indices[start:end]]
            positive_distances[start:end] = np.linalg.norm(
                anchor - positive, axis=1)
            negative_distances[start:end] = np.linalg.norm(
                anchor - negative, axis=1)

        metrics = verification_metrics(positive_distances, negative_distances,
                                       self.cfg.verification_threshold)
        losses = (np.square(positive_distances)
                  - np.minimum(negative_distances,
                               self.cfg.maximum_negative_distance))
        metrics["loss"] = float(np.mean(losses))
        if query_paths:
            query_embeddings = self._embed_paths(query_paths)
            metrics["market1501_rank1"] = self._market1501_rank1(
                query_embeddings, query_paths, rank1_gallery_embeddings,
                rank1_gallery_paths)
        return metrics

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
                    metrics = self.evaluate()
                    validation_loss = metrics["loss"]
                    rank1 = metrics.get("market1501_rank1")
                    rank1_log = (f" Rank-1={rank1:.4f}"
                                 if rank1 is not None else "")
                    print(f"\nvalidation loss={validation_loss:.4f} "
                          f"d_pos={metrics['positive_mean_distance']:.4f} "
                          f"d_neg={metrics['negative_mean_distance']:.4f} "
                          f"TAR={metrics['tar']:.4f} FAR={metrics['far']:.4f} "
                          f"threshold_acc={metrics['threshold_accuracy']:.4f} "
                          f"AUC={metrics['roc_auc']:.4f} EER={metrics['eer']:.4f} "
                          f"TAR@FAR10%={metrics['tar_at_far_10pct']:.4f} "
                          f"TAR@FAR5%={metrics['tar_at_far_5pct']:.4f} "
                          f"TAR@FAR1%={metrics['tar_at_far_1pct']:.4f}"
                          f"{rank1_log}")
                    if validation_loss < self.best_loss:
                        self.best_loss = validation_loss
                        self._save(current, best=True, loss=validation_loss)
        finally:
            self.train_loader.stop()
        print("\ntraining completed")
