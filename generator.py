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
import cv2
import numpy as np
import tensorflow as tf
import albumentations as A

from concurrent.futures.thread import ThreadPoolExecutor


class DataGenerator:
    def __init__(self, image_paths_of, input_shape, batch_size, augmentation=True):
        self.image_paths_of = image_paths_of
        self.input_shape = input_shape
        self.input_shape_1ch = input_shape[:2] + (1,)
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.class_names = list(image_paths_of.keys())
        self.img_index_of = dict()
        self.img_count_of = dict()
        self.excepted_class_names_of = dict()
        self.class_names_for_balancing = []  # for data balancing
        self.class_name_index_for_balancing = 0
        for class_name in self.class_names:
            self.img_index_of[class_name] = 0
            self.img_count_of[class_name] = len(self.image_paths_of[class_name])
            np.random.shuffle(self.image_paths_of[class_name])
            class_names_copy = list(self.class_names)
            class_names_copy.remove(class_name)
            self.excepted_class_names_of[class_name] = class_names_copy
            for _ in self.image_paths_of[class_name]:
                self.class_names_for_balancing.append(class_name)
        self.black_img = np.zeros(shape=self.input_shape_1ch, dtype=np.float32)
        np.random.shuffle(self.class_names_for_balancing)
        self.pool = ThreadPoolExecutor(8)
        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.3, contrast_limit=0.4),
            A.GaussianBlur(p=0.5, blur_limit=(5, 5))
            # TODO : Rotate
        ])

    def load(self):
        fs = []
        for _ in range(self.batch_size):
            class_name = self.get_next_class_name()
            path_a = self.get_next_image_path_of(class_name)
            if np.random.uniform() < 0.5:
                path_b = self.get_next_image_path_of(class_name)
                fs.append(self.pool.submit(self.load_2_img, path_a, path_b, 1.0))
            else:
                diff_class_name = np.random.choice(self.excepted_class_names_of[class_name])
                path_b = self.get_next_image_path_of(diff_class_name)
                fs.append(self.pool.submit(self.load_2_img, path_a, path_b, 0.0))
        batch_x = []
        batch_y = []
        for f in fs:
            img_a, img_b, target = f.result()
            if self.augmentation:
                img_a = self.transform(image=img_a)['image']
                img_b = self.transform(image=img_b)['image']
            img_a = cv2.resize(img_a, (self.input_shape[1], self.input_shape[0]))
            img_b = cv2.resize(img_b, (self.input_shape[1], self.input_shape[0]))
            x_a = np.asarray(img_a).reshape(self.input_shape_1ch).astype('float32') / 255.0
            x_b = np.asarray(img_b).reshape(self.input_shape_1ch).astype('float32') / 255.0
            batch_x.append(np.concatenate([x_a, x_b, self.black_img], axis=2))
            batch_y.append([target])
        batch_x = np.asarray(batch_x).reshape((self.batch_size,) + self.input_shape).astype('float32')
        batch_y = np.asarray(batch_y).reshape((self.batch_size, 1)).astype('float32')
        return batch_x, batch_y

    def get_next_class_name(self):
        path = self.class_names_for_balancing[self.class_name_index_for_balancing]
        self.class_name_index_for_balancing += 1
        if self.class_name_index_for_balancing == len(self.class_names_for_balancing):
            self.class_name_index_for_balancing = 0
            np.random.shuffle(self.class_names_for_balancing)
        return path

    def get_next_image_path_of(self, class_name):
        index = self.img_index_of[class_name]
        path = self.image_paths_of[class_name][index]
        self.img_index_of[class_name] += 1
        if self.img_index_of[class_name] == self.img_count_of[class_name]:
            self.img_index_of[class_name] = 0
            np.random.shuffle(self.image_paths_of[class_name])
        return path

    def load_2_img(self, path_a, path_b, target):
        return self.load_img(path_a), self.load_img(path_b), target

    def load_img(self, path):
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

