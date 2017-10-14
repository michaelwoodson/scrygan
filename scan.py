"""
Scans an audio file with a trained model and stores the outputs in a raw
numpy file for further processing.  The file is hardcoded.
"""

import argparse
import yaml
import os
import sys
import time

import soundfile as sf
import tensorflow as tf
import numpy as np

from model import ScryNetModel

DATA_DIRECTORY = 'orchive'
LOGDIR_ROOT = 'logdir'
PARAMS = 'params.yaml'

def get_arguments():
    parser = argparse.ArgumentParser(description='ScryNet training script')
    parser.add_argument('--data-dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the wav file corpus.')
    parser.add_argument('--restore-from', required=True, type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in LOGDIR_ROOT. ')
    parser.add_argument('--params', type=str, default=PARAMS,
                        help='YAML file with parameters to override defaults. Default: ' + PARAMS + '.')
    return parser.parse_args()

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

def main():
    args = get_arguments()

    start_minutes = 0
    length_seconds = 60*45
    audio_size = 16000 * length_seconds
    start = 16000 * (start_minutes * 60)
    log_index = 0

    with open(os.path.join(args.restore_from, 'config.yaml'), 'r') as f:
        scrynet_params = yaml.load(f)
    scan_size = scrynet_params['scan_size']
    print("Loaded params: {}".format(scrynet_params))

    # Create network.
    model = ScryNetModel(batch_size = 1, mini_batch_size = scan_size, **scrynet_params["model"])

    raw_audio_input, state_out, outputs = model.scan()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_list=tf.trainable_variables())
    load(saver, sess, args.restore_from)

    try:
        audio_feed = np.zeros((1, scan_size, model.window_size), dtype=np.float32)
        audio, _ = sf.read(os.path.join('orchive', '2006-046B.1c.wav'))
        state_log = np.zeros((audio_size // scan_size, model.n_lstm_hidden))
        last_print = time.time()
        state = model.zero_state()
        for index, offset in enumerate(range(0, audio_size, scan_size)):
            if model.window_size == 1:
                chunk = audio[start + offset:start + offset + scan_size]
                chunk = chunk.reshape(scan_size, 1)
            else:
                chunk = np.array([audio[start+offset+i:start+offset+i+model.window_size] for i in range(scan_size)])
            audio_feed[0] = chunk
            feed_dict = {raw_audio_input: audio_feed}
            model.load_placeholders(feed_dict, state)
            state, out_data = sess.run([state_out, outputs], feed_dict=feed_dict)
            try:
                state_log[index] = out_data[0][-1]
            except:
                pass
            if time.time() - last_print > 10:
                last_print = time.time()
                print("Progress: {}/{}".format(offset, audio_size))
        np.save(os.path.join(args.restore_from, 'scan.npy'), state_log)

    except KeyboardInterrupt:
        print()
    finally:
        pass


if __name__ == '__main__':
    main()
