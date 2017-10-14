from __future__ import division

import tensorflow as tf


def create_adam_optimizer(learning_rate, momentum):
    return tf.train.AdamOptimizer(learning_rate=learning_rate)


def create_sgd_optimizer(learning_rate, momentum):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate)


def create_rmsprop_optimizer(learning_rate, momentum):
    return tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=momentum)

def create_nadam_optimizer(learning_rate, momentum):
    return tf.contrib.opt.NadamOptimizer(learning_rate=learning_rate)


optimizer_factory = {'adam': create_adam_optimizer,
                     'nadam': create_nadam_optimizer,
                     'sgd': create_sgd_optimizer,
                     'rmsprop': create_rmsprop_optimizer}

def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    with tf.name_scope('encode'):
        mu = tf.to_float(quantization_channels - 1)
        # Perform mu-law companding transformation (ITU-T, 1988).
        # Minimum operation is here to deal with rare large amplitudes caused
        # by resampling.
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.to_int32((signal + 1) / 2 * mu + 0.5)

def one_hot(batch_size, quantization_channels, nn):
    '''One-hot encodes the waveform amplitudes.

    This allows the definition of the network as a categorical distribution
    over a finite set of possible amplitudes.
    '''
    with tf.name_scope('one_hot_encode'):
        nn = tf.one_hot(
            nn,
            depth=quantization_channels,
            dtype=tf.float32)
        nn = tf.reshape(nn, [batch_size, -1, quantization_channels])
    return nn


def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        signal = 2 * (tf.to_float(output) / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude
