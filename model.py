"""Edge-friendly ReID models built without normalization layers."""

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
        backbone_name = getattr(self.cfg, "backbone", "compact_cnn")
        if backbone_name == "compact_cnn":
            x = self._compact_cnn(inputs)
        elif backbone_name == "mobilenet_v2":
            x = self._mobilenet_v2(inputs)
        elif backbone_name == "mobilenet_v3_small":
            x = self._mobilenet_v3_small(inputs)
        elif backbone_name == "efficientnet_b0":
            x = self._efficientnet_b0(inputs)
        else:
            raise ValueError(f"unknown backbone: {backbone_name!r}")
        x = tf.keras.layers.Conv2D(self.cfg.embedding_dim, 1, use_bias=False, kernel_regularizer=self._regularizer(), name="embedding_conv")(x)
        outputs = tf.keras.layers.GlobalAveragePooling2D(name="embedding")(x)
        return tf.keras.Model(inputs, outputs, name="reid_model")

    def _compact_cnn(self, inputs):
        x = inputs
        for filters in (32, 64, 128, 256, 512):
            x = tf.keras.layers.Conv2D(
                filters, 3, strides=2, padding="same", use_bias=True,
                kernel_regularizer=self._regularizer())(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        return x

    def _mobilenet_v2(self, inputs):
        x = self._conv(inputs, 32, 3, strides=2)
        settings = (
            (1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2),
            (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2),
            (6, 320, 1, 1),
        )
        for expansion, filters, repeats, stride in settings:
            for repeat in range(repeats):
                x = self._mbconv(x, filters, expansion,
                                 stride if repeat == 0 else 1)
        return self._conv(x, 1280, 1)

    def _mobilenet_v3_small(self, inputs):
        x = self._conv(inputs, 16, 3, strides=2)
        settings = (
            (16, 16, 2), (72, 24, 2), (88, 24, 1), (96, 40, 2),
            (240, 40, 1), (240, 40, 1), (120, 48, 1), (144, 48, 1),
            (288, 96, 2), (576, 96, 1), (576, 96, 1),
        )
        for expanded_filters, filters, stride in settings:
            x = self._mbconv(x, filters, expanded_filters=expanded_filters,
                             stride=stride)
        return self._conv(x, 576, 1)

    def _efficientnet_b0(self, inputs):
        x = self._conv(inputs, 32, 3, strides=2)
        settings = (
            (1, 16, 1, 1), (6, 24, 2, 2), (6, 40, 2, 2),
            (6, 80, 3, 2), (6, 112, 3, 1), (6, 192, 4, 2),
            (6, 320, 1, 1),
        )
        for expansion, filters, repeats, stride in settings:
            for repeat in range(repeats):
                x = self._mbconv(x, filters, expansion,
                                 stride if repeat == 0 else 1)
        return self._conv(x, 1280, 1)

    def _mbconv(self, x, filters, expansion=None, stride=1,
                expanded_filters=None):
        input_filters = int(x.shape[-1])
        expanded_filters = expanded_filters or input_filters * expansion
        if expanded_filters != input_filters:
            x = self._conv(x, expanded_filters, 1)
        x = tf.keras.layers.DepthwiseConv2D(
            3, strides=stride, padding="same", activation="relu",
            use_bias=True, depthwise_regularizer=self._regularizer())(x)
        return self._conv(x, filters, 1, activation=None)

    def _conv(self, x, filters, kernel_size, strides=1, activation="relu"):
        return tf.keras.layers.Conv2D(
            filters, kernel_size, strides=strides, padding="same",
            activation=activation, use_bias=True,
            kernel_regularizer=self._regularizer())(x)

    def _regularizer(self):
        return tf.keras.regularizers.l2(self.cfg.l2)
