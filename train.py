"""Training script for the ScryNet network.

This script trains a network with ScryNet using data from a wav file corpus,
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
from tensorflow.python.client import timeline


from model import ScryNetModel
from audio_reader import AudioReader
from ops import optimizer_factory

random.seed(0)

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

    parser = argparse.ArgumentParser(description='ScryNet training script')
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

    if not os.path.exists(logdir):
        os.makedirs(logdir)

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
            scrynet_params = yaml.load(f)
    except IOError:
        print("no restore")
        with open('default_params.yaml', 'r') as f:
            scrynet_params = yaml.load(f)
        try:
            if args.params:
                with open(args.params, 'r') as f:
                    scrynet_params.update(yaml.load(f))
        except IOError:
            print("No params file found, using defaults.")
    print("Loaded params: {}".format(yaml.dump(scrynet_params)))

    # Load raw waveform from wav file corpus.
    sample_rate = scrynet_params['sample_rate']
    with tf.name_scope('create_inputs'):
        reader = AudioReader(
            args.data_dir,
            sample_rate=scrynet_params['sample_rate'],
            batch_size=scrynet_params['batch_size'],
            num_mini_batches=scrynet_params['num_mini_batches'],
            mini_batch_size=scrynet_params['mini_batch_size'],
            window_size=scrynet_params['model']['window_size'])

    # Create network.
    batch_size = scrynet_params["batch_size"]
    model = ScryNetModel(
        batch_size=batch_size,
        mini_batch_size=scrynet_params["mini_batch_size"],
        **scrynet_params["model"])

    loss, raw_audio_input, outputs, state_out = model.loss()
    optimizer = optimizer_factory[scrynet_params["optimizer"]](
                    learning_rate=scrynet_params["learning_rate"],
                    momentum=scrynet_params["momentum"])
    optim = optimizer.minimize(loss, var_list=tf.trainable_variables())

    # Set up logging for TensorBoard.
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()

    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.trainable_variables())

    text_file = open(os.path.join(logdir, "config.yaml"), "w")
    text_file.write(yaml.dump(scrynet_params))
    text_file.close()

    try:
        saved_global_step = load(saver, sess, restore_from)
        if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1

    except:
        print("Something went wrong while restoring checkpoint. "
            "We will terminate training to avoid accidentally overwriting "
            "the previous model.")
        raise

    step = None
    last_saved_step = saved_global_step
    profiler = tf.profiler.Profiler(sess.graph)
    mini_batch_size = model.mini_batch_size
    num_mini_batches = scrynet_params["num_mini_batches"]
    print("Seconds scanned per audio file: {:.1f}".format(num_mini_batches * mini_batch_size / sample_rate))
    try:
        audio_feed = np.zeros((model.batch_size, (mini_batch_size + 1), model.window_size), dtype=np.float32)
        for step in range(saved_global_step + 1, scrynet_params["num_steps"]):
            for batch in reader.get_batches():
                if step == 0:
                    sample_wav = sf.SoundFile(os.path.join(logdir, 'sample.wav'), channels=1, samplerate=16000, mode='w')
                    for audio in batch:
                        sample_wav.write(audio)
                    sample_wav.close()
                start_time = time.time()
                last_print = time.time()
                state = model.zero_state()
                for mini_batch_counter in range(num_mini_batches):
                    for batch_index, audio in enumerate(batch):
                        offset = mini_batch_counter * mini_batch_size
                        if model.window_size == 1:
                            chunk = audio[offset:offset + mini_batch_size + 1]
                            chunk = chunk.reshape((mini_batch_size + 1, 1))
                        else:
                            chunk = np.array([audio[offset + i:offset + i + model.window_size] for i in range(model.mini_batch_size + 1)])
                        audio_feed[batch_index] = chunk
                    audio_feed = audio_feed
                    feed_dict = {raw_audio_input: audio_feed}
                    model.load_placeholders(feed_dict, state)
                    if mini_batch_counter == -1:
                        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        summary, loss_value, _, state = sess.run([summaries, loss, optim, state_out],
                            feed_dict=feed_dict,
                            options=options,
                            run_metadata=run_metadata)
                        profiler.add_step(0, run_metadata)
                        opts = (tf.profiler.ProfileOptionBuilder(
                                tf.profiler.ProfileOptionBuilder.time_and_memory())
                                .with_step(0)
                                .with_timeline_output('timeline.json').build())
                        profiler.profile_graph(options=opts)
                    else:
                        summary, loss_value, _, state = sess.run([summaries, loss, optim, state_out], feed_dict=feed_dict)
                    mini_batch_counter += 1
                    if time.time() - last_print > 10:
                        last_print = time.time()
                        duration = time.time() - start_time
                        print('progress {:d}/{:d} - loss = {:.7f} - rate = {:.1f}'
                            .format(mini_batch_counter, num_mini_batches, loss_value, (batch_size*mini_batch_counter*mini_batch_size)/duration))
                        #predictions = sess.run([outputs], feed_dict={raw_audio_input: audio_feed})
                        #first_predictions = predictions[0][0]
                        #first_actual = audio_feed[0][1:].reshape((mini_batch_size))
                        #print(np.stack([first_predictions, first_actual], -1)[:4].T)
            #writer.add_summary(summary, step)
            duration = time.time() - start_time
            print('step {:d} - loss = {:.7f}, ({:.3f} sec/step)'
                .format(step, loss_value, duration))

            save(saver, sess, logdir, step)
            last_saved_step = step

    except KeyboardInterrupt:
        print()
    finally:
        writer.close()
        #if last_saved_step and step > last_saved_step:
        #    save(saver, sess, logdir, step)


if __name__ == '__main__':
    main()
