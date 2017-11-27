"""Training script for the ScryGan network.

This script trains a network with ScryGan using data from a wav file corpus,
"""

from __future__ import print_function

import argparse
from datetime import datetime
import yaml
import os
import sys
import time
import random

import tensorflow as tf
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from scipy import signal
from scipy import misc
from tensorflow.python.client import timeline


from model import ScryGanModel
from audio_reader import AudioReader
from ops import *

random.seed(3)

DATA_DIRECTORY = 'orchive'
LOGDIR_ROOT = 'logdir'
PARAMS = 'params.yaml'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='ScryGan training script')
    parser.add_argument('--data-dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the wav file corpus.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. ')
    parser.add_argument('--restore-from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in LOGDIR_ROOT. '
                        'Cannot use with --logdir.')
    parser.add_argument('--params', type=str, default=PARAMS,
                        help='YAML file with parameters to override defaults. Default: ' + PARAMS + '.')
    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()
    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def get_default_logdir():
    logdir = os.path.join(LOGDIR_ROOT, 'train', STARTED_DATESTRING)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    return logdir


def validate_directories(args):
    """Validate and arrange directory related arguments."""

    # Validation
    if args.logdir and args.restore_from:
        raise ValueError(
            "--logdir and --restore_from cannot be specified at the same "
            "time. This is to keep your previous model from unexpected "
            "overwrites.\n"
            "only --logdir to just continue the training from the last "
            "checkpoint.")

    logdir = args.logdir
    if logdir is None:
        logdir = get_default_logdir()
        print('Using default logdir: {}'.format(logdir))

    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are exclusive,
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir

    return {
        'logdir': logdir,
        'restore_from': restore_from
    }


def main():
    args = get_arguments()

    try:
        directories = validate_directories(args)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return

    logdir = directories['logdir']
    restore_from = directories['restore_from']

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from

    restore_params = os.path.join(restore_from, 'config.yaml')
    try:
        with open(restore_params) as f:
            scrygan_params = yaml.load(f)
    except IOError:
        print("no restore")
        with open('default_params.yaml', 'r') as f:
            scrygan_params = yaml.load(f)
        try:
            if args.params:
                with open(args.params, 'r') as f:
                    scrygan_params.update(yaml.load(f))
        except IOError:
            print("No params file found, using defaults.")
    print("Loaded params: {}".format(yaml.dump(scrygan_params)))

    batch_size = scrygan_params["batch_size"]
    sample_rate = 16000
    sample_size = scrygan_params["sample_size"]
    overlap_size = scrygan_params["overlap_size"]
    save_interval = scrygan_params["save_interval"]
    fast_z = scrygan_params["fast_z"]
    num_t = scrygan_params["num_t"]
    print("sample_size: {}".format(sample_size))
    num_steps = scrygan_params["num_steps"]
    with tf.name_scope('create_inputs'):
        reader = AudioReader(
            args.data_dir,
            batch_size=batch_size,
            sample_size=sample_size,
            overlap_size=overlap_size,
            num_t=num_t)
    model = ScryGanModel(
        batch_size=batch_size,
        sample_size=sample_size,
        **scrygan_params["model"])

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    print('discriminator shape: {}'.format(model.D.shape))
    print('d_loss shape: {}'.format(model.d_loss.shape))
    d_optim = tf.train.AdamOptimizer(scrygan_params["d_learning_rate"], beta1=0.5).minimize(model.d_loss, var_list=model.d_vars)
    print('generator shape: {}'.format(model.G.shape))
    print('g_loss shape: {}'.format(model.g_loss.shape))
    g_optim = tf.train.AdamOptimizer(scrygan_params["g_learning_rate"], beta1=0.5).minimize(model.g_loss, var_list=model.g_vars)
    init = tf.global_variables_initializer()
    sess.run(init)
    model.g_sum = tf.summary.merge([model.z_sum, model.d__sum, model.d_loss_fake_sum, model.g_loss_sum])
    model.d_sum = tf.summary.merge([model.z_sum, model.d_sum, model.d_loss_real_sum, model.d_loss_sum])
    writer = tf.summary.FileWriter(logdir, sess.graph)
    saver = tf.train.Saver(var_list=tf.trainable_variables())
    text_file = open(os.path.join(logdir, "config.yaml"), "w")
    text_file.write(yaml.dump(scrygan_params))
    text_file.close()

    saved_global_step = -1
    #try:
    #    saved_global_step = load(saver, sess, restore_from)
    #    if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
    #        saved_global_step = -1
    #except:
    #    print("Something went wrong while restoring checkpoint. "
    #        "We will terminate training to avoid accidentally overwriting "
    #        "the previous model.")
    #    raise

    step = None
    last_saved_step = saved_global_step
    #profiler = tf.profiler.Profiler(sess.graph)
    print("Seconds scanned per audio file: {:.1f}".format(sample_size / sample_rate))
    try:
        for step in range(saved_global_step + 1, num_steps):
            batch = reader.get_batch()
            start_time = time.time()
            spectrograms = []
            for idx, full_audio in enumerate(batch):
                audio_sequence = []
                for t in range(0, num_t):
                    start = t * sample_size - overlap_size * t
                    audio = full_audio[start : start + sample_size]
                    f, t, Sxx = signal.spectrogram(audio, 16000, nperseg=256, nfft=256)
                    Sxx = misc.imresize(Sxx, (64, 64))
                    audio_sequence.append(Sxx)
                spectrograms.append(audio_sequence)
            spectrograms = np.array(spectrograms)
            g_state = model.zero_state()
            d_state = model.zero_state()
            d_state_ = model.zero_state()
            slow_z_batch = np.random.uniform(-1, 1, [model.batch_size, model.z_dim - fast_z]).astype(np.float32)
            do_sampling = np.mod(step, save_interval) == 0
            for t in range(num_t):
                if fast_z > 0:
                    fast_z_batch = np.random.uniform(-1, 1, [model.batch_size, fast_z]).astype(np.float32)
                    batch_z = np.concatenate([slow_z_batch, fast_z_batch], axis=1)
                else:
                    batch_z = slow_z_batch
                #print("spectograms.shape: {}".format(spectrograms.shape))
                t_batch = spectrograms[:,t]
                #print("t_batch.shape: {}".format(t_batch.shape))
                raw_audio_batch = np.array(t_batch)
                raw_audio_batch = np.expand_dims(raw_audio_batch, axis=-1)

                # Update network
                feed_dict = {model.inputs: raw_audio_batch, model.z: batch_z}
                model.load_placeholders(model.D, feed_dict, d_state)
                model.load_placeholders(model.D_, feed_dict, d_state_)
                model.load_placeholders(model.G, feed_dict, g_state)
                _, _, errD_fake, errD_real, errG, d_summary_str, g_summary_str, d_state, d_state_, g_state, samples = sess.run([
                    d_optim,
                    g_optim,
                    model.d_loss_fake,
                    model.d_loss_real,
                    model.g_loss,
                    model.d_sum,
                    model.g_sum,
                    model.state_out[model.D],
                    model.state_out[model.D_],
                    model.state_out[model.G],
                    model.G if do_sampling else model.g_sum
                ], feed_dict=feed_dict)
                writer.add_summary(d_summary_str, step)
                writer.add_summary(g_summary_str, step)

            if do_sampling:
            #    save(saver, sess, logdir, step)
            #    last_saved_step = step
                sample_images = []
                for idx in range(24):
                    for t in range(6):
                        sample_images.append(spectrograms[idx,t,:,:])
                save_images(np.array(sample_images).reshape([144,64,64,1]), (12,12),
                    os.path.join(logdir, 'sample_{:04d}.png'.format(step)))
                print("training sample saved")
                sample_images = np.zeros((24,6,64,64,1))
                sampler_state = model.zero_state()
                si = []
                #slow_z_batch = np.random.uniform(-1, 1, [model.batch_size, model.z_dim - fast_z]).astype(np.float32)
                for t in range(6):
                #    if fast_z > 0:
                #        fast_z_batch = np.random.uniform(-1, 1, [model.batch_size, fast_z]).astype(np.float32)
                #        sample_z = np.concatenate([slow_z_batch, fast_z_batch], axis=1)
                #    else:
                #        sample_z = slow_z_batch
                    sb = []
                #    feed_dict = {model.z: sample_z}
                #    model.load_placeholders(model.sampler, feed_dict, sampler_state)
                #    samples, sampler_state = sess.run([model.sampler, model.state_out[model.sampler]], feed_dict=feed_dict)
                    for idx in range(24):
                        sample_images[idx, t] = samples[idx]
                        sb.append(samples[idx])
                    si.append(sb)
                trythis = []
                for idx in range(24):
                    for t in range(6):
                        trythis.append(si[t][idx])
                save_images(np.array(trythis).reshape([144,64,64,1]), image_manifold_size(samples.shape[0]),
                    os.path.join(logdir, 'train_{:04d}.png'.format(step)))
                #save_images(sample_images.reshape([144,64,64,1]), image_manifold_size(samples.shape[0]),
                #    os.path.join(logdir, 'train_{:04d}.png'.format(step)))
            print("Epoch: [%03d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                % (step, time.time() - start_time, errD_fake+errD_real, errG))

    except KeyboardInterrupt:
        print()
    finally:
        pass
        #writer.close()
        #if last_saved_step and step > last_saved_step:
        #    save(saver, sess, logdir, step)
        #    model.save(config.checkpoint_dir, counter)

if __name__ == '__main__':
    main()
