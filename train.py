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
    print("sample_size: {}".format(sample_size))
    num_steps = scrygan_params["num_steps"]
    with tf.name_scope('create_inputs'):
        reader = AudioReader(
            args.data_dir,
            batch_size=batch_size,
            sample_size=sample_size)
    model = ScryGanModel(
        batch_size=batch_size,
        sample_size=sample_size,
        **scrygan_params["model"])

    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    #d_optim = tf.train.AdamOptimizer(scrygan_params["d_learning_rate"], beta1=config.beta1).minimize(model.d_loss, var_list=model.d_vars)
    #g_optim = tf.train.AdamOptimizer(scrygan_params["g_learning_rate"], beta1=config.beta1).minimize(model.g_loss, var_list=model.g_vars)
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
    #writer.add_graph(tf.get_default_graph())
    #saver = tf.train.Saver(var_list=tf.trainable_variables())
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
            for idx, audio in enumerate(batch):
                f, t, Sxx = signal.spectrogram(audio, 16000, nperseg=256, nfft=256)
                Sxx = misc.imresize(Sxx, (64, 64))
                spectrograms.append(Sxx)
                #dBS = 10 * np.log10(Sxx)  # convert to dB
                #plt.pcolormesh(t, f, dBS)


            #last_print = time.time()
            #for idx, audio in enumerate(batch):
                #state = model.zero_state()
                #feed_dict = {raw_audio_input: audio_feed}
                #model.load_placeholders(feed_dict, state)
                #if mini_batch_counter == -1:
                #    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                #    run_metadata = tf.RunMetadata()
                #    summary, loss_value, _, state = sess.run([summaries, loss, optim, state_out],
                #        feed_dict=feed_dict,
                #        options=options,
                #        run_metadata=run_metadata)
                #    profiler.add_step(0, run_metadata)
                #    opts = (tf.profiler.ProfileOptionBuilder(
                #            tf.profiler.ProfileOptionBuilder.time_and_memory())
                #            .with_step(0)
                #            .with_timeline_output('timeline.json').build())
                #    profiler.profile_graph(options=opts)
                #else:
                #    summary, loss_value, _, state = sess.run([summaries, loss, optim, state_out], feed_dict=feed_dict)
                #if time.time() - last_print > 10:
                #    last_print = time.time()
                #    duration = time.time() - start_time
                #    print('loss = {:.7f}'.format(loss_value)

            sample_z = np.random.uniform(-1, 1, size=(model.batch_size, model.z_dim))
            batch_z = np.random.uniform(-1, 1, [model.batch_size, model.z_dim]) \
                    .astype(np.float32)
            raw_audio_batch = np.array(spectrograms)
            raw_audio_batch = np.expand_dims(raw_audio_batch, axis=-1)
            if step == 0:
                save_images(raw_audio_batch, image_manifold_size(raw_audio_batch.shape[0]),
                    os.path.join(logdir, 'sample.png'.format(step)))

            # Update D network
            _, summary_str = sess.run([d_optim, model.d_sum],
            feed_dict={ model.inputs: raw_audio_batch, model.z: batch_z })
            writer.add_summary(summary_str, step)

            # Update G network
            _, summary_str = sess.run([g_optim, model.g_sum],
            #_ = sess.run([g_optim],
            feed_dict={ model.z: batch_z })
            writer.add_summary(summary_str, step)

            errD_fake = sess.run(model.d_loss_fake, feed_dict={model.z: batch_z})
            errD_real = sess.run(model.d_loss_real, feed_dict={ model.inputs: raw_audio_batch })
            errG = sess.run(model.g_loss, feed_dict={model.z: batch_z})
            if np.mod(step, 100) == 1:
                #try:
                samples, d_loss, g_loss = sess.run(
                    [model.sampler, model.d_loss, model.g_loss],
                        feed_dict={
                            model.z: sample_z,
                            model.inputs: raw_audio_batch,
                        },
                )
                save_images(samples, image_manifold_size(samples.shape[0]),
                    os.path.join(logdir, 'train_{:04d}.png'.format(step)))
                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
                #except Exception as e:
                #    print("Problem saving image: [{}]".format(e))

            print("Epoch: [%03d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                % (step, time.time() - start_time, errD_fake+errD_real, errG))
            #duration = time.time() - start_time
            #print('step {:d}/{:d}- loss = {:.7f}, ({:.3f} sec/step)'.format(step, num_steps, loss_value, duration))

            #save(saver, sess, logdir, step)
            last_saved_step = step

    except KeyboardInterrupt:
        print()
    finally:
        pass
        #writer.close()
        #if last_saved_step and step > last_saved_step:
        #    save(saver, sess, logdir, step)


def train(config):
    pass

        #if np.mod(counter, 500) == 2:
        #    model.save(config.checkpoint_dir, counter)

if __name__ == '__main__':
    main()
