"""ReID embedding model with optional lightweight pretrained backbones."""

import tensorflow as tf


class Model:
    _APPLICATION_BACKBONES = {
        "mobilenet_v2": tf.keras.applications.MobileNetV2,
        "mobilenet_v3_small": tf.keras.applications.MobileNetV3Small,
        "efficientnet_b0": tf.keras.applications.EfficientNetB0,
    }

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
        else:
            x = self._application_backbone(inputs, backbone_name)
        x = tf.keras.layers.Conv2D(self.cfg.embedding_dim, 1, use_bias=False, kernel_regularizer=self._regularizer(), name="embedding_conv")(x)
        outputs = tf.keras.layers.GlobalAveragePooling2D(name="embedding")(x)
        return tf.keras.Model(inputs, outputs, name="reid_model")

    def _compact_cnn(self, inputs):
        x = inputs
        for filters in (32, 64, 128, 256, 512):
            x = tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same", use_bias=True, kernel_regularizer=self._regularizer())(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        return x

    def _application_backbone(self, inputs, backbone_name):
        if backbone_name not in self._APPLICATION_BACKBONES:
            choices = ", ".join(("compact_cnn", *self._APPLICATION_BACKBONES))
            raise ValueError(f"unknown backbone {backbone_name!r}; choose one of: {choices}")
        if self.cfg.input_channels != 3:
            raise ValueError("pretrained application backbones require input_channels: 3")

        # The loader emits floats in [0, 1]. With preprocessing disabled,
        # MobileNet expects [-1, 1], while EfficientNet expects [0, 255].
        if backbone_name.startswith("mobilenet_"):
            x = tf.keras.layers.Rescaling(2.0, offset=-1.0,
                                          name="backbone_preprocess")(inputs)
        else:
            x = tf.keras.layers.Rescaling(255.0,
                                          name="backbone_preprocess")(inputs)

        constructor = self._APPLICATION_BACKBONES[backbone_name]
        kwargs = {
            "include_top": False,
            "weights": getattr(self.cfg, "backbone_weights", "imagenet"),
            "input_shape": (self.cfg.input_rows, self.cfg.input_cols, 3),
        }
        if backbone_name == "mobilenet_v3_small":
            kwargs["include_preprocessing"] = False
        backbone = constructor(**kwargs)
        backbone.trainable = bool(getattr(self.cfg, "backbone_trainable", True))
        return backbone(x)

    def _regularizer(self):
        return tf.keras.regularizers.l2(self.cfg.l2)
