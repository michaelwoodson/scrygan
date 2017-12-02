import numpy as np
import tensorflow as tf

from ops import *
from tensorflow.contrib import rnn

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class ScryGanModel(object):
    '''Implements the ScryGan network for semantic detection in audio streams.'''

    def __init__(self,
                 batch_size,
                 sample_size,
                 g_lstm_layers,
                 g_lstm_hidden,
                 d_lstm_layers,
                 d_lstm_hidden,
                 gf_dim,
                 df_dim,
                 z_dim):
        self.placeholder_cs = {}
        self.placeholder_hs = {}
        self.state_out = {}
        '''Initializes the ScryGan model. See default_params.yaml for each setting.'''
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.g_lstm_layers = g_lstm_layers
        self.g_lstm_hidden = g_lstm_hidden
        self.d_lstm_layers = d_lstm_layers
        self.d_lstm_hidden = d_lstm_hidden
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = 1
        self.z_dim = z_dim
        print("creating network [Batch Size: {:d}]".format(self.batch_size))

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn00 = batch_norm(name='g_bn00')
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn3 = batch_norm(name='g_bn4')

        self.inputs = tf.placeholder(tf.float32, [self.batch_size, 64, 64, 1], name='spectrograms')

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        #self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)

        self.G = self.generator(self.z)
        print("inputs shape: {} {}".format(self.inputs.shape, self.inputs.dtype))
        print("generator shape: {} {}".format(self.G.shape, self.G.dtype))
        self.D, self.D_logits = self.discriminator(self.inputs, reuse=False)
        self.sampler = self.generator(self.z, reuse=True)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.image("G", self.G)
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        g_x = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))
        print("D_logits_: {}".format(self.D_logits_.shape))
        print("D_: {}".format(self.D_.shape))
        self.g_loss = tf.reduce_mean(g_x)

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()
        for var in t_vars:
            print('var: {}\t{}'.format(var.name, var.shape))

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

    def lstm(self, key, inputs, n_layers, n_hidden):
        def lstm_cell():
            return tf.contrib.rnn.LSTMBlockCell(n_hidden, forget_bias=1.0)
        print('inputs shape: {}'.format(inputs.shape))
        cells = [lstm_cell() for _ in range(n_layers)]
        placeholder_cs = [tf.placeholder(tf.float32, name="{}_c_{}".format(key, i), shape=(self.batch_size, n_hidden)) for i in range(n_layers)]
        placeholder_hs = [tf.placeholder(tf.float32, name="{}_h_{}".format(key, i), shape=(self.batch_size, n_hidden)) for i in range(n_layers)]
        if n_layers == 0:
            return inputs, inputs
        elif n_layers == 1:
            state = rnn.LSTMStateTuple(placeholder_cs[0], placeholder_hs[0])
            cell = cells[0]
        else:
            state = tuple([rnn.LSTMStateTuple(c,h) for c, h in zip(placeholder_cs, placeholder_hs)])
            cell = tf.contrib.rnn.MultiRNNCell(cells)
        outputs, state_out = cell(inputs, state, scope="{}_rnn".format(key))
        return outputs, placeholder_cs, placeholder_hs, state_out

    def d_zero_state(self):
        return self.zero_state(self.d_lstm_layers, self.d_lstm_hidden)

    def g_zero_state(self):
        return self.zero_state(self.g_lstm_layers, self.g_lstm_hidden)

    def zero_state(self, layers, hidden):
        def zero_tuple():
            return rnn.LSTMStateTuple(np.zeros((self.batch_size, hidden), np.float32), np.zeros((self.batch_size, hidden), np.float32))
        if layers == 1:
            return zero_tuple()
        else:
            return [zero_tuple() for _ in range(layers)]

    def d_load_placeholders(self, key, feed_dict, states):
        self.load_placeholders(key, feed_dict, states, self.d_lstm_layers)

    def g_load_placeholders(self, key, feed_dict, states):
        self.load_placeholders(key, feed_dict, states, self.g_lstm_layers)

    def load_placeholders(self, key, feed_dict, states, layers):
        if layers == 1:
            feed_dict[self.placeholder_cs[key][0]] = states[0]
            feed_dict[self.placeholder_hs[key][0]] = states[1]
        else:
            for s, p in zip(states, self.placeholder_cs[key]):
                feed_dict[p] = s[0]
            for s, p in zip(states, self.placeholder_hs[key]):
                feed_dict[p] = s[1]

    def discriminator(self, audio, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            print("MAKING DISCRIMINATOR")
            nn = lrelu(conv2d(audio, self.df_dim, name='d_h0_conv'))
            nn = lrelu(self.d_bn1(conv2d(nn, self.df_dim*2, name='d_h1_conv')))
            nn = lrelu(self.d_bn2(conv2d(nn, self.df_dim*4, name='d_h2_conv')))
            nn = lrelu(self.d_bn3(conv2d(nn, self.df_dim*8, name='d_h3_conv')))
            flat = tf.reshape(nn, [self.batch_size, -1])

            # Project, run through lstm, concatenate projection with convoluation.
            #projection, _, _ = linear(flat, output_size=self.n_lstm_hidden, scope='d_projection_linear')
            #projection = tf.nn.relu(projection)
            #nn, placeholder_cs, placeholder_hs, state_out = self.lstm("d", projection)
            #nn = tf.concat([nn, projection], 1)

            # Concatenate lstm with flattened convolution
            #nn, placeholder_cs, placeholder_hs, state_out = self.lstm("d", flat)
            #nn = tf.concat([nn, flat], 1)

            # Force everything through lstm
            nn, placeholder_cs, placeholder_hs, state_out = self.lstm("d", flat, self.d_lstm_layers, self.d_lstm_hidden)

            nn, _, _ = linear(nn, output_size=1, scope='d_h4_lin')

            d = tf.nn.sigmoid(nn)
            self.placeholder_cs[d] = placeholder_cs
            self.placeholder_hs[d] = placeholder_hs
            self.state_out[d] = state_out

            return d, nn

    def generator(self, z, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            print("MAKING GENERATOR")
            print("z.shape: {}".format(z.shape))
            s_w = 64
            s_w2 = conv_out_size_same(s_w, 2)
            s_w4 = conv_out_size_same(s_w2, 2)
            s_w8 = conv_out_size_same(s_w4, 2)
            s_w16 = conv_out_size_same(s_w8, 2)

            nn, placeholder_cs, placeholder_hs, state_out = self.lstm("g", z, self.g_lstm_layers, self.g_lstm_hidden)
            #nn = lrelu(self.g_bn00(nn))

            #nn = tf.concat([nn, z], axis=1)

            # project `z` and reshape
            nn, self.h0_w, self.h0_b = linear(nn, output_size=self.gf_dim * s_w16 * s_w16 * 8, scope='g_h0_lin')

            self.h0 = tf.reshape(nn, [-1, s_w16, s_w16, self.gf_dim * 8])
            nn = lrelu(self.g_bn0(self.h0))
            self.h1, _, _ = deconv2d(nn, [self.batch_size, s_w8, s_w8, self.gf_dim*4], name='g_h1')
            nn = lrelu(self.g_bn1(self.h1))
            nn, _, _ = deconv2d(nn, [self.batch_size, s_w4, s_w4, self.gf_dim*2], name='g_h2')
            nn = lrelu(self.g_bn2(nn))
            nn, _, _ = deconv2d(nn, [self.batch_size, s_w2, s_w2, self.gf_dim*1], name='g_h3')
            nn = lrelu(self.g_bn3(nn))
            nn, _, _ = deconv2d(nn, [self.batch_size, s_w, s_w, self.c_dim], name='g_h4')
            #return tf.nn.tanh(nn)
            g = tf.nn.relu(nn)
            self.placeholder_cs[g] = placeholder_cs
            self.placeholder_hs[g] = placeholder_hs
            self.state_out[g] = state_out
            return g
