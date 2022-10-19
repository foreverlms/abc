#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

from tensorflow.python.framework import graph_util
import os
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
pb_file_path = os.getcwd()

op_names = ['conv1', 'conv1_bias_add', 'conv1_relu', 'pool1', 'matmul1', 'softmax1']


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W, name):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME", name=name)


def max_pool_2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)


def bias_add(x, y, name):
    return tf.nn.bias_add(x, y, name=name)


def relu(x, name):
    return tf.nn.relu(x, name=name)


def mat_mul(x, w, name):
    tf.matmul(x, w, name=name)


def softmax(x, name):
    tf.nn.softmax(x, name=name)


def Run():
    # input
    x = tf.placeholder(tf.float32, [None, 784], name='input')
    x_image = tf.reshape(x, [-1, 28, 28, 1], name='reshape_input')

    # Conv1 + BiasAdd1 + Relu1
    # Weight : 32x5x5x1
    W_conv1 = weight_variable([5, 5, 1, 32], 'Conv1_Weight')
    B_conv1 = bias_variable([32], 'Conv1_Bias')
    O_conv1 = relu(bias_add(conv2d(x_image, W_conv1, 'conv1'), B_conv1, 'conv1_bias_add'), 'conv1_relu')

    # Pool1
    O_pool1 = max_pool_2x2(O_conv1, 'pool1')
    print(O_pool1)

    # Reshape
    I_matmul1 = tf.reshape(O_pool1, [-1, 14 * 14 * 32], name='reshape_pool')
    print(I_matmul1)
    # Matmul
    W_matmul1 = weight_variable([14 * 14 * 32, 10], 'Matmul1_Weight')
    print(W_matmul1)
    O_matmul1 = tf.matmul(I_matmul1, W_matmul1, name='matmul1')
    print(O_matmul1)
    # Softmax
    O_softmax1 = softmax(O_matmul1, 'softmax1')

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, op_names)

        with tf.gfile.FastGFile(os.path.join(pb_file_path, 'model.pb'), mode='wb') as f:
            f.write(constant_graph.SerializeToString())


if __name__ == "__main__":
    Run()
