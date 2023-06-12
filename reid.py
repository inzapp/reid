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
import cv2
import random
import numpy as np
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from model import Model
from generator import DataGenerator
from lr_scheduler import LRScheduler
from ale import AbsoluteLogarithmicError


class ReID:
    def __init__(self,
                 train_image_path,
                 input_rows,
                 input_cols,
                 lr,
                 momentum,
                 label_smoothing,
                 batch_size,
                 iterations,
                 gamma=2.0,
                 warm_up=0.5,
                 lr_policy='step',
                 model_name='model',
                 checkpoint_interval=0,
                 pretrained_model_path='',
                 validation_image_path=''):
        self.input_shape = (input_rows, input_cols, 3)
        self.lr = lr
        self.warm_up = warm_up
        self.gamma = gamma
        self.momentum = momentum
        self.label_smoothing = label_smoothing
        self.batch_size = batch_size
        self.iterations = iterations
        self.lr_policy = lr_policy 
        self.model_name = model_name
        self.max_val_acc = 0.0
        self.checkpoint_interval = checkpoint_interval
        self.pretrained_iteration_count = 0
        self.checkpoint_path = 'checkpoint'

        train_image_path = self.unify_path(train_image_path)
        validation_image_path = self.unify_path(validation_image_path)

        self.train_image_paths_of, self.train_image_count = self.init_image_paths_of(train_image_path)
        self.validation_image_paths_of, self.validation_image_count = self.init_image_paths_of(validation_image_path)
        if self.train_image_count == 0:
            print(f'no images in train_image_path : {train_image_path}')
            exit(0)
        if self.validation_image_count == 0:
            print(f'no images in validation_image_path : {validation_image_path}')
            exit(0)

        self.train_data_generator = DataGenerator(
            image_paths_of=self.train_image_paths_of,
            input_shape=self.input_shape,
            batch_size=self.batch_size)
        self.validation_data_generator = DataGenerator(
            image_paths_of=self.validation_image_paths_of,
            input_shape=self.input_shape,
            batch_size=self.batch_size)
        self.train_data_generator_one_batch = DataGenerator(
            image_paths_of=self.train_image_paths_of,
            input_shape=self.input_shape,
            batch_size=1,
            augmentation=False)
        self.validation_data_generator_one_batch = DataGenerator(
            image_paths_of=self.validation_image_paths_of,
            input_shape=self.input_shape,
            batch_size=1,
            augmentation=False)

        if pretrained_model_path != '':
            if os.path.exists(pretrained_model_path) and os.path.isfile(pretrained_model_path):
                self.pretrained_iteration_count = self.parse_pretrained_iteration_count(pretrained_model_path)
                self.model = tf.keras.models.load_model(pretrained_model_path, compile=False)
            else:
                print(f'pretrained model not found : {pretrained_model_path}')
                exit(0)
        else:
            self.model = Model(input_shape=self.input_shape).build()
            self.model.save('model.h5', include_optimizer=False)

    def parse_pretrained_iteration_count(self, pretrained_model_path):
        iteration_count = 0
        sp = f'{os.path.basename(pretrained_model_path)[:-3]}'.split('_')
        for i in range(len(sp)):
            if sp[i] == 'iter' and i > 0:
                try:
                    iteration_count = int(sp[i-1])
                except:
                    pass
                break
        return iteration_count

    def unify_path(self, path):
        if path == '':
            return path
        path = path.replace('\\', '/')
        if path.endswith('/'):
            path = path[:-1]
        return path

    def init_image_paths_of(self, image_path):
        dir_path_candidates = sorted(glob(f'{image_path}/*'))
        for i in range(len(dir_path_candidates)):
            dir_path_candidates[i] = dir_path_candidates[i].replace('\\', '/')
        dir_paths = []
        for candidate in dir_path_candidates:
            basename = os.path.basename(candidate)
            if basename[0] != '_' and os.path.isdir(candidate) and len(glob(f'{candidate}/*.jpg')) > 0:
                dir_paths.append(candidate)
        image_count = 0
        image_paths_of = dict()
        for dir_path in dir_paths:
            basename = os.path.basename(dir_path)
            image_paths_of[basename] = glob(f'{dir_path}/*.jpg')
            image_count += len(image_paths_of[basename])
        return image_paths_of, image_count

    @tf.function
    def compute_gradient(self, model, optimizer, batch_x, y_true, loss_function):
        with tf.GradientTape() as tape:
            y_pred = self.model(batch_x, training=True)
            loss = tf.reduce_mean(loss_function(y_true, y_pred))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def fit(self):
        self.model.summary()
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.momentum)
        if not (os.path.exists(self.checkpoint_path) and os.path.exists(self.checkpoint_path)):
            os.makedirs(self.checkpoint_path, exist_ok=True)

        iteration_count = self.pretrained_iteration_count
        print(f'\ntrain on {self.train_image_count} samples')
        print(f'validate on {self.validation_image_count} samples\n')
        loss_function = AbsoluteLogarithmicError(gamma=self.gamma, label_smoothing=self.label_smoothing)
        lr_scheduler = LRScheduler(lr=self.lr, iterations=self.iterations, warm_up=self.warm_up, policy=self.lr_policy)
        while True:
            batch_x, batch_y = self.train_data_generator.load()
            lr_scheduler.update(optimizer, iteration_count)
            loss = self.compute_gradient(self.model, optimizer, batch_x, batch_y, loss_function)
            iteration_count += 1
            print(f'\r[iteration count : {iteration_count:6d}] loss => {loss:.4f}', end='')
            if iteration_count % 2000 == 0:
                for last_model_path in glob(f'{self.checkpoint_path}/model_last_*_iter.h5'):
                    os.remove(last_model_path)
                self.model.save(f'{self.checkpoint_path}/model_last_{iteration_count}_iter.h5', include_optimizer=False)
            if iteration_count == self.iterations:
                self.save_model(iteration_count)
                for last_model_path in glob(f'{self.checkpoint_path}/model_last_*_iter.h5'):
                    os.remove(last_model_path)
                print('train end successfully')
                exit(0)
            elif iteration_count >= int(self.iterations * self.warm_up) and self.checkpoint_interval > 0 and iteration_count % self.checkpoint_interval == 0:
                self.save_model(iteration_count)

    def save_model(self, iteration_count):
        print(f'iteration count : {iteration_count}')
        if self.validation_data_generator.flow() is None:
            self.model.save(f'{self.checkpoint_path}/{self.model_name}_{iteration_count}_iter.h5', include_optimizer=False)
        else:
            # self.evaluate_core(confidence_threshold=0.5, validation_data_generator=self.train_data_generator_one_batch)
            val_acc, val_class_score = self.evaluate_core(confidence_threshold=0.5, validation_data_generator=self.validation_data_generator_one_batch)
            model_name = f'{self.model_name}_{iteration_count}_iter_acc_{val_acc:.4f}_class_score_{val_class_score:.4f}'
            if val_acc > self.max_val_acc:
                self.max_val_acc = val_acc
                model_name = f'{self.checkpoint_path}/best_{model_name}.h5'
                print(f'[best model saved]\n')
            else:
                model_name = f'{self.checkpoint_path}/{model_name}.h5'
            self.model.save(model_name, include_optimizer=False)

    def evaluate(self, confidence_threshold=0.5):
        self.evaluate_core(confidence_threshold=confidence_threshold, validation_data_generator=self.validation_data_generator_one_batch)

    def evaluate_core(self, confidence_threshold=0.5, validation_data_generator=None):
        @tf.function
        def predict(model, x):
            return model(x, training=False)
        num_classes = self.model.output_shape[1]
        hit_counts = np.zeros(shape=(num_classes,), dtype=np.int32)
        total_counts = np.zeros(shape=(num_classes,), dtype=np.int32)
        hit_unknown_count = total_unknown_count = 0
        hit_scores = np.zeros(shape=(num_classes,), dtype=np.float32)
        unknown_score_sum = 0.0
        for batch_x, batch_y in tqdm(validation_data_generator.flow()):
            y = predict(self.model, batch_x)[0]
            max_score_index = np.argmax(y)
            max_score = y[max_score_index]
            if np.sum(batch_y[0]) == 0.0:  # case unknown using zero label
                total_unknown_count += 1
                if max_score < confidence_threshold:
                    hit_unknown_count += 1
                    unknown_score_sum += max_score
            else:  # case classification
                true_class_index = np.argmax(batch_y[0])
                total_counts[true_class_index] += 1
                if max_score_index == true_class_index:
                    hit_counts[true_class_index] += 1
                    hit_scores[true_class_index] += max_score

        print('\n')
        total_acc_sum = 0.0
        class_score_sum = 0.0
        for i in range(len(total_counts)):
            cur_class_acc = hit_counts[i] / (float(total_counts[i]) + 1e-5)
            cur_class_score = hit_scores[i] / (float(hit_counts[i]) + 1e-5)
            total_acc_sum += cur_class_acc
            class_score_sum += cur_class_score
            print(f'[class {i:2d}] acc : {cur_class_acc:.4f}, score : {cur_class_score:.4f}')

        valid_class_count = num_classes

        class_acc = total_acc_sum / valid_class_count
        class_score = class_score_sum / num_classes
        print(f'reid classifier accuracy : {class_acc:.4f}, class_score : {class_score:.4f}')
        return class_acc, class_score
