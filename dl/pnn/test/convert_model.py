#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

from pnn import converter as pc


def tf_covnert():
    config = pc.ConvertConfig()

    config.type = pc.ConvertConfig.TF
    config.output_name = "yamnet"
    config.output_dir = "./"
    config.original_model_path = "/Users/bob/Downloads/mobilenetv2_fsd2018_41cls.pb"
    config.fp16_mode = False
    config.opt_level = 2
    # you can also config the input shape.

    converter = pc.Converter()
    converter.convert(config)


def onnx_convert():
    config = pc.ConvertConfig()

    config.type = pc.ConvertConfig.ONNX
    config.output_name = "yamnet"
    config.output_dir = "./"
    config.original_model_path = "/Users/bob/Downloads/"
    config.fp16_mode = False
    config.opt_level = 2
    # you can also config the input shape.

    converter = pc.Converter()
    converter.convert(config)


if __name__ == '__main__':
    onnx_convert()
