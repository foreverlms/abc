import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from onnx_tf.backend import prepare
from onnx import numpy_helper

import tensorflow as tf
import onnx

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

part_2_path = "/Users/bob/docs/ByteDance/espresso/part_1.onnx"

conv3d = nn.Sequential(
    nn.Conv3d(32, 32, 3, 1, 1),
    nn.ReLU(),
    nn.Conv3d(32, 32, 3, 1, 1),
    nn.ReLU(),
    nn.Conv3d(32, 32, 3, 1, 1),
    nn.ReLU(),
    nn.Conv3d(32, 32, 3, 1, 1),
    nn.ReLU(),
    nn.Conv3d(32, 1, 3, 1, 1)
)
conv3d = conv3d.eval()

data = torch.rand(1, 32, 16, 24, 32)

print(conv3d(data).shape)
