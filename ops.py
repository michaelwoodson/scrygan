import math
import numpy as np 
import scipy
import tensorflow as tf

from tensorflow.python.framework import ops

#image_summary = tf.summary.image
#scalar_summary = tf.summary.scalar
#histogram_summary = tf.summary.histogram
#merge_summary = tf.summary.merge
#SummaryWriter = tf.summary.FileWriter

def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                            decay=self.momentum, 
                            updates_collections=None,
                            epsilon=self.epsilon,
                            scale=True,
                            is_training=train,
                            scope=self.name)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
        x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv1d(input_, output_dim, 
       k_w=5, d_w=2, stddev=0.02,
       name="conv1d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_w, input_.get_shape()[-1], output_dim],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv1d(input_, w, stride = d_w, padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

# https://github.com/tensorflow/tensorflow/issues/8729
def conv1d_transpose(x, W, output_shape, strides):
    new_x = tf.expand_dims(x, 1)
    new_W = tf.expand_dims(W, 0)
    print("!new_x.shape: {}".format(new_x.shape))
    print("!new_W.shape: {}".format(new_W.shape))
    output_shape = (output_shape[0], 1 ,output_shape[1], output_shape[2])
    print("!output_shape: {}".format(output_shape))
    print("!strides: {}".format(strides))
    #deconv = tf.nn.conv2d_transpose(new_x, new_W, output_shape=output_shape, strides=strides, data_format="NHWC")
    deconv = tf.nn.conv2d_transpose(new_x, new_W, output_shape=output_shape, strides=strides, data_format="NHWC")
    print("!deconv.shape0: {}".format(deconv.shape))
    deconv = tf.squeeze(deconv, axis=1)
    print("!deconv.shape: {}".format(deconv.shape))
    return deconv

def deconv1d(input_, output_shape,
        k_w=5, stddev=0.02,
        name="deconv1d"):
    d_w = 2
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_w, output_shape[-1], input_.shape[-1]],
                initializer=tf.random_normal_initializer(stddev=stddev))
#        output_shape = (output_shape[0], output_shape[1], output_shape[2])
        deconv = conv1d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_w, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        #TODO: MW: is it this?: Perhaps this reshape is issue?
        print("BEFORE deconv.shape: {}".format(deconv.shape))
        #deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.nn.bias_add(deconv, biases)
        print("AFTER deconv.shape: {}".format(deconv.shape))

        return deconv, w, biases

def conv2d(input_, output_dim, 
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        return deconv, w, biases

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0):
#    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_.shape[1], output_size], tf.float32,
                    tf.truncated_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], tf.float32, initializer=tf.constant_initializer(bias_start))
        return tf.matmul(input_, matrix) + bias, matrix, bias


def save_images(images, size, image_path):
    return imsave(inverse_transform(images*256.0), size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img

def inverse_transform(images):
    return (images+1.)/2.

def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    print("[{},{},{},{}]".format(num_images, manifold_h, manifold_w, manifold_h * manifold_w))
    #assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w