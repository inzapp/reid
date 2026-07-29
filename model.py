"""Compact CNN that emits ReID embeddings."""

import math

import tensorflow as tf


class Model:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, strategy, optimizer):
        with strategy.scope():
            model = self._build_model()
            model.compile(optimizer=optimizer)
        return model

    def _build_model(self):
        inputs = tf.keras.layers.Input((self.cfg.input_rows, self.cfg.input_cols, self.cfg.input_channels), name="input")
        x = inputs
        for filters in (32, 64, 128, 256, 512):
            x = tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same", use_bias=True, kernel_regularizer=self._regularizer())(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.Conv2D(self.cfg.embedding_dim, 1, use_bias=False, kernel_regularizer=self._regularizer(), name="embedding_conv")(x)
        x = tf.keras.layers.GlobalAveragePooling2D(name="embedding_pool")(x)
        # Keep each embedding dimension on a stable scale without requiring an
        # explicit vector-normalization operation in the edge model.
        x = tf.keras.layers.BatchNormalization(
            scale=False, name="embedding_bn")(x)
        # BN gives every dimension unit variance. Scaling by sqrt(D) keeps the
        # expected vector norm near one, so a capped distance of 2 remains
        # meaningful without per-vector L2 normalization.
        outputs = tf.keras.layers.Rescaling(
            1.0 / math.sqrt(self.cfg.embedding_dim), name="embedding")(x)
        return tf.keras.Model(inputs, outputs, name="reid_model")

    def _regularizer(self):
        return tf.keras.regularizers.l2(self.cfg.l2)
