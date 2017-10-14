import numpy as np
import tensorflow as tf

from ops import one_hot, mu_law_encode
from tensorflow.contrib import rnn

class ScryNetModel(object):
    '''Implements the ScryNet network for semantic detection in raw audio.'''

    def __init__(self,
                 batch_size,
                 mini_batch_size,
                 n_lstm_layers,
                 n_lstm_hidden,
                 fc_hidden,
                 quantization_channels,
                 dropout,
                 layer_normalization,
                 window_size,
                 fc_stack_raw_audio,
                 fc_layers):
        '''Initializes the ScryNet model. See default_params.yaml for each setting.'''
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.n_lstm_layers = n_lstm_layers
        self.quantization_channels = quantization_channels
        self.n_lstm_hidden = n_lstm_hidden
        self.fc_hidden = fc_hidden
        self.dropout = dropout
        self.layer_normalization = layer_normalization
        self.window_size = window_size
        self.fc_stack_raw_audio = fc_stack_raw_audio
        self.fc_layers = fc_layers

    def _create_network(self, inputs, is_training):
        print("creating network [Batch Size: {:d}] [Mini Batch Size: {:d}] [LSTM Layers: {:d}] [LSTM Hidden: {:d}]".format(
            self.batch_size, self.mini_batch_size, self.n_lstm_layers, self.n_lstm_hidden))
        with tf.variable_scope('main_rnn'):
            def lstm_cell():
                if self.layer_normalization:
                    cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.n_lstm_hidden, forget_bias=1.0)
                else:
                    cell = tf.contrib.rnn.LSTMBlockCell(self.n_lstm_hidden, forget_bias=1.0)
                if is_training and self.dropout < 1:
                    return tf.contrib.rnn.DropoutWrapper(cell, state_keep_prob=self.dropout)
                else:
                    return cell
            self.cells = [lstm_cell() for _ in range(self.n_lstm_layers)]
            self.placeholder_cs = [tf.placeholder(tf.float32, shape=(self.batch_size, self.n_lstm_hidden)) for _ in range(self.n_lstm_layers)]
            self.placeholder_hs = [tf.placeholder(tf.float32, shape=(self.batch_size, self.n_lstm_hidden)) for _ in range(self.n_lstm_layers)]
            if self.n_lstm_layers == 0:
                return inputs, inputs
            elif self.n_lstm_layers == 1:
                state = rnn.LSTMStateTuple(self.placeholder_cs[0], self.placeholder_hs[0])
                nn = self.cells[0]
            else:
                state = tuple([rnn.LSTMStateTuple(c,h) for c, h in zip(self.placeholder_cs, self.placeholder_hs)])
                nn = tf.contrib.rnn.MultiRNNCell(self.cells)
            return tf.nn.dynamic_rnn(nn, inputs, dtype=tf.float32, initial_state=state)

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

    def loss(self, name='scrynet_training'):
        with tf.variable_scope(name):
            with tf.name_scope('prepare_input'):
                raw_audio_input = tf.placeholder(tf.float32, (self.batch_size, self.mini_batch_size + 1, self.window_size))
                input_slice = tf.slice(raw_audio_input, [0, 0, 0], [-1, self.mini_batch_size, -1])
            outputs, state_out = self._create_network(input_slice, True)
            with tf.variable_scope('projection'):
                if self.fc_hidden == -1:
                    nn = tf.slice(outputs, [0, 0, self.n_lstm_hidden-self.quantization_channels], [-1, -1, -1])
                else:
                    fc_input_size = self.n_lstm_hidden if self.n_lstm_layers > 0 else self.window_size
                    if self.fc_stack_raw_audio:
                        nn = tf.concat([input_slice, outputs], axis=2)
                        fc_input_size = fc_input_size + self.window_size
                    else:
                        nn = outputs
                    nn = tf.reshape(nn, [self.batch_size * self.mini_batch_size, fc_input_size])
                    if self.fc_hidden == 0:
                        nn = tf.contrib.layers.fully_connected(nn, self.quantization_channels,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.1))
                    else:
                        for i in range(self.fc_layers):
                            nn = tf.contrib.layers.fully_connected(nn, self.fc_hidden,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.1))
                        nn = tf.contrib.layers.fully_connected(nn, self.quantization_channels,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.1))
                nn = tf.reshape(nn, [self.batch_size, self.mini_batch_size, self.quantization_channels])
            with tf.variable_scope('post_processing'):
                targets = tf.slice(raw_audio_input, [0, 1, self.window_size - 1], [-1, -1, -1])
                targets = mu_law_encode(targets, self.quantization_channels)
                targets = tf.reshape(targets, [self.batch_size, self.mini_batch_size])
                print("targets: {}".format(targets))
            with tf.variable_scope('loss'):
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=nn, labels=targets)
                nn = tf.reduce_mean(losses)
                tf.summary.scalar('loss', nn)
                return nn, raw_audio_input, outputs, state_out

    def scan(self, name='scrynet'):
        with tf.variable_scope(name):
            raw_audio_input = tf.placeholder(tf.float32, (self.batch_size, self.mini_batch_size, self.window_size), name="raw_audio")
            outputs, state_out = self._create_network(tf.slice(raw_audio_input, [0, 0, 0], [-1, self.mini_batch_size, -1]), False)
            return raw_audio_input, state_out, outputs

