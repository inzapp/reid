"""Asynchronous, identity-aware triplet loader for person ReID."""

import os
import queue
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}


class DataLoader:
    """Prefetch augmented (anchor, positive, negative) triplets.

    Paths are indexed by identity once. Worker threads independently sample and
    decode triplets into a bounded queue, so training never scans the dataset.
    """

    def __init__(self, cfg, training=False):
        self.cfg = cfg
        self.training = training
        self.data_path = cfg.train_data_path if training else cfg.validation_data_path
        self.data_paths = self.get_data_paths()
        self.paths_by_id = self._group_paths_by_id(self.data_paths)
        self.identities = tuple(self.paths_by_id)
        self.anchor_identities = tuple(
            identity for identity, paths in self.paths_by_id.items() if len(paths) >= 2)
        self._validate_dataset()

        self.q = queue.Queue(maxsize=cfg.max_q_size)
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._executor = None
        self._started = False
        self._worker_error = None
        self._error_lock = threading.Lock()

    def get_data_paths(self):
        root = Path(self.data_path)
        if not root.is_dir():
            raise ValueError(f"image directory does not exist: {root}")
        paths = sorted(str(path) for path in root.rglob("*")
                       if path.is_file() and path.suffix in IMAGE_SUFFIXES)
        invalid_id_count = sum(not self._has_valid_identity(path)
                               for path in paths)
        split = "training" if self.training else "validation"
        print(f"{split}: excluded {invalid_id_count} images with IDs below 1")
        return [path for path in paths if self._has_valid_identity(path)]

    @staticmethod
    def identity_from_path(path):
        return os.path.basename(path).split("_", 1)[0]

    @classmethod
    def _has_valid_identity(cls, path):
        try:
            return int(cls.identity_from_path(path)) >= 1
        except ValueError:
            return False

    def _group_paths_by_id(self, paths):
        grouped = defaultdict(list)
        for path in paths:
            grouped[self.identity_from_path(path)].append(path)
        return dict(grouped)

    def _validate_dataset(self):
        if not self.data_paths:
            raise ValueError(f"no supported images found in: {self.data_path}")
        if len(self.identities) < 2:
            raise ValueError("triplet sampling requires at least two identities")
        if not self.anchor_identities:
            raise ValueError("at least one identity must have two or more images")

    def start(self):
        if self._started:
            return
        self._started = True
        self._stop_event.clear()
        self._worker_error = None
        workers = max(1, int(self.cfg.num_loader_workers))
        self._executor = ThreadPoolExecutor(max_workers=workers,
                                            thread_name_prefix="reid-loader")
        for worker_index in range(workers):
            self._executor.submit(self._producer, worker_index)

        # Warm up enough triplets for one batch, rather than blocking until the
        # whole (potentially large) queue is full.
        target = min(self.q.maxsize, self.cfg.batch_size)
        while self.q.qsize() < target:
            self._raise_worker_error()
            if self._stop_event.wait(0.05):
                self._raise_worker_error()
                raise RuntimeError("data loader stopped during prefetch")

    def stop(self):
        if not self._started:
            return
        self._stop_event.set()
        self._pause_event.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        self._executor = None
        self._started = False

    def pause(self):
        self._pause_event.set()

    def resume(self):
        self._pause_event.clear()

    def _producer(self, worker_index):
        seed = None if not self.cfg.fix_seed else 42 + worker_index
        rng = random.Random(seed)
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                self._stop_event.wait(0.05)
                continue
            try:
                triplet = self._load_random_triplet(rng)
                self.q.put(triplet, timeout=0.1)
            except queue.Full:
                continue
            except Exception as exc:
                with self._error_lock:
                    self._worker_error = exc
                self._stop_event.set()
                return

    def _load_random_triplet(self, rng):
        anchor_id = rng.choice(self.anchor_identities)
        negative_id = rng.choice(self.identities)
        while negative_id == anchor_id:
            negative_id = rng.choice(self.identities)
        anchor_path, positive_path = rng.sample(self.paths_by_id[anchor_id], 2)
        negative_path = rng.choice(self.paths_by_id[negative_id])
        return tuple(self.load_image(path, augment=self.training, rng=rng)
                     for path in (anchor_path, positive_path, negative_path))

    def load(self):
        if not self._started:
            raise RuntimeError("DataLoader.start() must be called before load()")
        items = []
        while len(items) < self.cfg.batch_size:
            self._raise_worker_error()
            try:
                items.append(self.q.get(timeout=0.2))
            except queue.Empty:
                self._raise_worker_error()
        anchors, positives, negatives = zip(*items)
        return (np.stack(anchors), np.stack(positives), np.stack(negatives))

    def _raise_worker_error(self):
        with self._error_lock:
            error = self._worker_error
        if error is not None:
            raise RuntimeError("data loader worker failed") from error

    def load_image(self, path, augment=False, rng=None):
        encoded = np.fromfile(path, dtype=np.uint8)
        flag = cv2.IMREAD_GRAYSCALE if self.cfg.input_channels == 1 else cv2.IMREAD_COLOR
        image = cv2.imdecode(encoded, flag)
        if image is None:
            raise ValueError(f"failed to decode image: {path}")
        return self.preprocess(image, augment=augment, rng=rng)

    def preprocess(self, image, augment=False, rng=None):
        rng = rng or random
        if self.cfg.input_channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image[..., None]

        if augment:
            image = self._augment(image, rng)
        else:
            image = self._resize(image)

        if image.ndim == 2:
            image = image[..., None]
        return np.ascontiguousarray(image, dtype=np.float32) / 255.0

    def _augment(self, image, rng):
        image = self._resize(image)
        padding = max(0, int(self.cfg.random_crop_padding))
        if padding:
            image = cv2.copyMakeBorder(image, padding, padding, padding, padding,
                                       cv2.BORDER_REFLECT_101)
            max_y = image.shape[0] - self.cfg.input_rows
            max_x = image.shape[1] - self.cfg.input_cols
            y = rng.randint(0, max_y)
            x = rng.randint(0, max_x)
            image = image[y:y + self.cfg.input_rows,
                          x:x + self.cfg.input_cols]

        if image.ndim == 2:
            image = image[..., None]

        if rng.random() < self.cfg.horizontal_flip_probability:
            image = np.flip(image, axis=1)

        image = image.astype(np.float32)
        jitter = float(self.cfg.color_jitter)
        if jitter > 0:
            contrast = rng.uniform(1.0 - jitter, 1.0 + jitter)
            brightness = rng.uniform(-255.0 * jitter, 255.0 * jitter)
            image = image * contrast + brightness
        image = np.clip(image, 0.0, 255.0)

        if rng.random() < self.cfg.random_erasing_probability:
            image = self._random_erasing(image, rng)
        return image

    def _resize(self, image):
        target = (self.cfg.input_cols, self.cfg.input_rows)
        interpolation = (cv2.INTER_LINEAR if image.shape[1] < target[0]
                         or image.shape[0] < target[1] else cv2.INTER_AREA)
        resized = cv2.resize(image, target, interpolation=interpolation)
        if image.ndim == 3 and resized.ndim == 2:
            resized = resized[..., None]
        return resized

    @staticmethod
    def _random_erasing(image, rng, min_area=0.02, max_area=0.25):
        height, width = image.shape[:2]
        area = height * width
        for _ in range(10):
            erase_area = rng.uniform(min_area, max_area) * area
            aspect = rng.uniform(0.3, 3.3)
            erase_h = int(round(np.sqrt(erase_area * aspect)))
            erase_w = int(round(np.sqrt(erase_area / aspect)))
            if 0 < erase_h < height and 0 < erase_w < width:
                y = rng.randint(0, height - erase_h)
                x = rng.randint(0, width - erase_w)
                image[y:y + erase_h, x:x + erase_w] = np.asarray(
                    [127.5] * image.shape[2], dtype=image.dtype)
                break
        return image

    def sample_validation_triplets(self, count, seed=0):
        rng = random.Random(seed)
        return [self._load_random_triplet(rng) for _ in range(count)]

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
