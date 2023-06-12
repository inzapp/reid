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
from reid import ReID

if __name__ == '__main__':
    ReID(
        train_image_path=r'/train_data/imagenet/train',
        validation_image_path=r'/train_data/imagenet/validation',
        input_rows=128,
        input_cols=128,
        lr=0.001,
        gamma=2.0,
        warm_up=0.5,
        momentum=0.9,
        batch_size=32,
        iterations=300000,
        label_smoothing=0.05,
        checkpoint_interval=0).fit()

