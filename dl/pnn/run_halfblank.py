#!/usr/bin/env python3
# _*_ coding: utf-8 _*_


import pnn
import cv2
import numpy as np


def foo():
    config = pnn.engine_config()

    cpu = pnn.cpu_device()

    config.devices = [cpu]

    engine = pnn.engine(config)

    print(engine.get_version())

    status = engine.init()

    model_file_path = "/Users/bob/Desktop/tmp/halfBlank_silu.pnn"

    status = engine.load_model(model_file_path)
    status = engine.resize_input_tensors({"images": [1, 3, 416, 416]})

    # input
    image = np.fromfile("/Users/bob/Desktop/tmp/half_blank_input.bin", dtype=np.float32).reshape((1, 3, 416, 416))
    print(image.shape)
    print(image.flatten()[0:10])

    inputs = engine.mutable_inputs()
    inputs[0].assign(image)

    status = engine.run()
    outputs = engine.get_outputs()
    output = outputs[0].data()

    # print(output.flatten()[0:100])

    gt = np.fromfile("/Users/bob/Desktop/tmp/half_blank_output.bin", dtype=np.float32).reshape(1, 3549, 6)

    # print(gt.flatten()[0:100])

    print(np.allclose(output, gt, rtol=1e-04, atol=1e-03))


if __name__ == "__main__":
    foo()
