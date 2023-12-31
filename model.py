"""
Authors : inzapp

Github url : https://github.com/inzapp/reid

Copyright 2021 inzapp Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"),
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import tensorflow as tf


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Model:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = self.__conv_block(input_layer, 16, 3)
        x = self.__max_pool(x)

        x = self.__conv_block(x, 32, 3)
        x = self.__max_pool(x)

        x = self.__conv_block(x, 64, 3)
        x = self.__max_pool(x)

        x = self.__conv_block(x, 128, 3)
        x = self.__max_pool(x)

        x = self.__conv_block(x, 256, 3)
        x = self.__max_pool(x)

        x = self.__conv_block(x, 256, 3)
        output_layer = self.__output_layer(x)
        return tf.keras.models.Model(input_layer, output_layer)

    def __conv_block(self, x, filters, kernel_size, bn=False):
        x = self.__conv(x, filters, kernel_size, use_bias=False if bn else True)
        if bn:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def __conv(self, x, filters, kernel_size, use_bias=True):
        return tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer='he_normal',
            padding='same',
            use_bias=use_bias)(x)

    def __output_layer(self, x, name='reid_output'):
        x = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            kernel_initializer='glorot_normal',
            activation='sigmoid')(x)
        return tf.keras.layers.GlobalAveragePooling2D(name=name)(x)

    def __max_pool(self, x):
        return tf.keras.layers.MaxPool2D()(x)

