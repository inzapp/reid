"""Compact CNN that emits ReID embeddings."""

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
        inputs = tf.keras.layers.Input(
            (self.cfg.input_rows, self.cfg.input_cols, self.cfg.input_channels),
            name="input")
        x = inputs
        for filters in (32, 64, 128, 256):
            x = self._residual_downsample(x, filters)
        x = tf.keras.layers.Conv2D(512, 3, padding="same", use_bias=False,
                                   kernel_regularizer=self._regularizer())(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Conv2D(
            self.cfg.embedding_dim, 1, use_bias=False,
            kernel_regularizer=self._regularizer(), name="embedding_conv")(x)
        outputs = tf.keras.layers.GlobalAveragePooling2D(name="embedding")(x)
        return tf.keras.Model(inputs, outputs, name="reid_model")

    def _residual_downsample(self, x, filters):
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=2, use_bias=False,
                                          kernel_regularizer=self._regularizer())(x)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

        x = tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same",
                                   use_bias=False,
                                   kernel_regularizer=self._regularizer())(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False,
                                   kernel_regularizer=self._regularizer())(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, shortcut])
        return tf.keras.layers.ReLU()(x)

    def _regularizer(self):
        return tf.keras.regularizers.l2(self.cfg.l2)
