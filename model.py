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
                 n_lstm_layers,
                 n_lstm_hidden,
                 gf_dim,
                 df_dim,
                 z_dim):
        '''Initializes the ScryGan model. See default_params.yaml for each setting.'''
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.n_lstm_layers = n_lstm_layers
        self.n_lstm_hidden = n_lstm_hidden
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = 1
        self.z_dim = z_dim
        print("creating network [Batch Size: {:d}]".format(self.batch_size))

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.inputs = tf.placeholder(tf.float32, [self.batch_size, 64, 64, 1], name='spectrograms')

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        #self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)
        self.y = None

        self.G = self.generator(self.z, self.y)
        print("inputs shape: {}".format(self.inputs.shape))
        print("generator shape: {}".format(self.G.shape))
        self.D, self.D_logits = self.discriminator(self.inputs, self.y, reuse=False)
        self.sampler = self.sampler(self.z, self.y)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

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

    def zero_state(self):
        def zero_tuple():
            return rnn.LSTMStateTuple(np.zeros((self.batch_size, self.n_lstm_hidden), np.float32), np.zeros((self.batch_size, self.n_lstm_hidden), np.float32))
        if self.n_lstm_layers == 1:
            return zero_tuple()
        else:
            return [zero_tuple() for _ in self.cells]

    def load_placeholders(self, feed_dict, states):
        if self.n_lstm_layers == 1:
            feed_dict[self.placeholder_cs[0]] = states[0]
            feed_dict[self.placeholder_hs[0]] = states[1]
        else:
            for s, p in zip(states, self.placeholder_cs):
                feed_dict[p] = s[0]
            for s, p in zip(states, self.placeholder_hs):
                feed_dict[p] = s[1]

    def discriminator(self, audio, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            print("MAKING DISCRIMINATOR")
            print("audio.shape: {}".format(audio.shape))
            nn = lrelu(conv2d(audio, self.df_dim, name='d_h0_conv'))
            print("nn1: {}".format(nn.shape))
            nn = lrelu(self.d_bn1(conv2d(nn, self.df_dim*2, name='d_h1_conv')))
            print("nn2: {}".format(nn.shape))
            nn = lrelu(self.d_bn2(conv2d(nn, self.df_dim*4, name='d_h2_conv')))
            print("nn3: {}".format(nn.shape))
            nn = lrelu(self.d_bn3(conv2d(nn, self.df_dim*8, name='d_h3_conv')))
            print("nn4: {}".format(nn.shape))
            nn = tf.reshape(nn, [self.batch_size, -1])
            print("nn5: {}".format(nn.shape))
            nn, _, _ = linear(nn, output_size=1, scope='d_h4_lin')
            print("nn6: {}".format(nn.shape))
            print("")

            return tf.nn.sigmoid(nn), nn

    def generator(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            print("MAKING GENERATOR")
            print("z.shape: {}".format(z.shape))
            s_w = 64
            s_w2 = conv_out_size_same(s_w, 2)
            s_w4 = conv_out_size_same(s_w2, 2)
            s_w8 = conv_out_size_same(s_w4, 2)
            s_w16 = conv_out_size_same(s_w8, 2)
            print("ss: {} {} {} {} {}".format(s_w, s_w2, s_w4, s_w8, s_w16))

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, output_size=self.gf_dim * s_w16 * s_w16 * 8, scope='g_h0_lin')

            self.h0 = tf.reshape(
                self.z_, [-1, s_w16, s_w16, self.gf_dim * 8])
            nn = tf.nn.relu(self.g_bn0(self.h0))
            print("nn1: {}".format(nn.shape))

            self.h1, self.h1_w, self.h1_b = deconv2d(
                nn, [self.batch_size, s_w8, s_w8, self.gf_dim*4], name='g_h1')
            nn = tf.nn.relu(self.g_bn1(self.h1))
            print("nn2: {}".format(nn.shape))

            nn, self.h2_w, self.h2_b = deconv2d(
                nn, [self.batch_size, s_w4, s_w4, self.gf_dim*2], name='g_h2')
            nn = tf.nn.relu(self.g_bn2(nn))
            print("nn3: {}".format(nn.shape))

            nn, self.h3_w, self.h3_b = deconv2d(
                nn, [self.batch_size, s_w2, s_w2, self.gf_dim*1], name='g_h3')
            nn = tf.nn.relu(self.g_bn3(nn))
            print("nn4: {}".format(nn.shape))

            nn, self.h4_w, self.h4_b = deconv2d(
                nn, [self.batch_size, s_w, s_w, self.c_dim], name='g_h4')
            print("nn5: {}".format(nn.shape))
            print()

            #return tf.nn.tanh(nn)
            return tf.nn.relu(nn)

    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            s_w = 64
            s_w2 = conv_out_size_same(s_w, 2)
            s_w4 = conv_out_size_same(s_w2, 2)
            s_w8 = conv_out_size_same(s_w4, 2)
            s_w16 = conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            nn, _, _ = linear(z, self.gf_dim*8*s_w16*s_w16, 'g_h0_lin')
            nn = tf.reshape(nn, [-1, s_w16, s_w16, self.gf_dim * 8])
            nn = tf.nn.relu(self.g_bn0(nn, train=False))

            nn, _, _ = deconv2d(nn, [self.batch_size, s_w8, s_w8, self.gf_dim*4], name='g_h1')
            nn = tf.nn.relu(self.g_bn1(nn, train=False))

            nn, _, _ = deconv2d(nn, [self.batch_size, s_w4, s_w4, self.gf_dim*2], name='g_h2')
            nn = tf.nn.relu(self.g_bn2(nn, train=False))

            nn, _, _ = deconv2d(nn, [self.batch_size, s_w2, s_w2, self.gf_dim*1], name='g_h3')
            nn = tf.nn.relu(self.g_bn3(nn, train=False))

            nn, _, _ = deconv2d(nn, [self.batch_size, s_w, s_w, self.c_dim], name='g_h4')

            #return tf.nn.tanh(h4)
            #return tf.nn.relu(tf.nn.tanh(nn))
            return tf.nn.relu(nn)
