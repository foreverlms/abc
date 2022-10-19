#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

import tensorflow as tf
import torch

dense_features = [f"rtt{i}" for i in range(1, 6)] + [f"time_gap_{i}" for i in range(1, 6)]
category_features = {
    'network_type': (7, 8),
    'socket_reused': (2, 8),
    'retry_attempts': (5, 8),
    'networktype_changed': (2, 8)
}

x = tf.random.uniform((2,3),-4,4,dtype=tf.float32)
y = tf.random.uniform((3,2),-4,4,dtype=tf.float32)

tf.einsum("ij,jk->ik",x,y)
torch.einsum()

input_tensor_name = dense_features + list(category_features.keys())
print(input_tensor_name)
output_tensor_name = ["predict/prob", "predict/label"]

deep_fm_pb_path = "/Users/bob/docs/ByteDance/PNN/converter/deepfm/optimized_model_bn.pb"

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(deep_fm_pb_path, input_tensor_name, output_tensor_name)

tflite_model = converter.convert()

with open("/Users/bob/docs/ByteDance/PNN/converter/deepfm/deepfm.tflite", "wb") as f:
    f.write(tflite_model)
