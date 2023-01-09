#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys

import numpy as np
from PIL import Image
import torch
import mmcv
from mmcv.parallel import collate, scatter
from mmcls.datasets.pipelines import Compose
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("ImageBatcher").setLevel(logging.INFO)
log = logging.getLogger("ImageBatcher")


def resnet_preprocess(img_path):
    print(f'preprocess img {img_path}')
    cfg = mmcv.Config.fromfile('resnet18_8xb32_in1k.py')
    test_pipeline = Compose(cfg.data.test.pipeline)
    input_data = dict(img_info=dict(filename=img_path), img_prefix=None)
    input_data = test_pipeline(input_data)
    input_data = collate([input_data], samples_per_gpu=1)
    input_data = scatter(input_data, [torch.device('cuda:0')])[0]
    input_data = input_data['img'].detach().cpu().numpy()
    return input_data


class ImageBatcher:
    """
    Creates batches of pre-processed images.
    """

    def __init__(self, input, shape, dtype, max_num_images=None,
                 exact_batches=False):
        """
        Args:
            input: The input directory to read images from.
            shape: The tensor shape of the batch to prepare,
                either in NCHW or NHWC format.
            dtype: The (numpy) datatype to cast the batched data to.
            max_num_images: The maximum number of images to read from
                the directory.
            exact_batches: This defines how to handle a number of images that
                is not an exact multiple of the batch size. If false, it will
                pad the final batch with zeros to reach the batch size.
                If true, it will *remove* the last few images in excess of a
                batch size multiple, to guarantee batches are exact (useful
                for calibration).
        """
        # Find images in the given input path
        input = os.path.realpath(input)
        self.images = []

        extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        def is_image(path):
            return os.path.isfile(path) and os.path.splitext(path)[
                1].lower() in extensions

        if os.path.isdir(input):
            self.images = [os.path.join(input, f) for f in os.listdir(input) if
                           is_image(os.path.join(input, f))]
            self.images.sort()
        elif os.path.isfile(input):
            if is_image(input):
                self.images.append(input)
        self.num_images = len(self.images)
        if self.num_images < 1:
            print("No valid {} images found in {}".format("/".join(extensions),
                                                          input))
            sys.exit(1)

        # Handle Tensor Shape
        self.dtype = dtype
        self.shape = shape
        assert len(self.shape) == 4
        self.batch_size = shape[0]
        assert self.batch_size > 0
        self.format = None
        self.width = -1
        self.height = -1
        if self.shape[1] == 3:
            self.format = "NCHW"
            self.height = self.shape[2]
            self.width = self.shape[3]
        elif self.shape[3] == 3:
            self.format = "NHWC"
            self.height = self.shape[1]
            self.width = self.shape[2]
        assert all([self.format, self.width > 0, self.height > 0])

        # Adapt the number of images as needed
        if max_num_images and 0 < max_num_images < len(self.images):
            self.num_images = max_num_images
        if exact_batches:
            self.num_images = self.batch_size * (
                    self.num_images // self.batch_size)
        if self.num_images < 1:
            print("Not enough images to create batches")
            sys.exit(1)
        self.images = self.images[0: self.num_images]
        print('')
        # Subdivide the list of images into batches
        self.num_batches = 1 + int((self.num_images - 1) / self.batch_size)
        self.batches = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_images)
            self.batches.append(self.images[start:end])
        # Indices
        self.image_index = 0
        self.batch_index = 0

    def get_batch(self):
        """
        Retrieve the batches. This is a generator object, so you can use it
        within a loop as: for batch, images in batcher.get_batch(): ... Or
        outside of a batch with the next() function.

        Returns:
            A generator yielding two items per iteration: a numpy array holding
             a batch of images, and the list of paths to the images loaded
             within this batch.
        """
        for _, batch_images in enumerate(self.batches):
            batch_data = np.zeros(self.shape, dtype=self.dtype)
            for i, image in enumerate(batch_images):
                self.image_index += 1
                batch_data[i] = resnet_preprocess(image)
            self.batch_index += 1
            yield batch_data, batch_images
