"""
Creates a wav file for each output of the LSTM.  (Requires lots of disk space.)
"""

import argparse
import matplotlib.pyplot as plt
import yaml
import os
import sys

import numpy as np
import soundfile as sf

DATA_DIRECTORY = 'orchive'
LOGDIR_ROOT = 'logdir'

def get_arguments():
    parser = argparse.ArgumentParser(description='ScryNet training script')
    parser.add_argument('--data-dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the wav file corpus.')
    parser.add_argument('--restore-from', required=True, type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in LOGDIR_ROOT. ')
    return parser.parse_args()

def main():

    args = get_arguments()
    with open(os.path.join(args.restore_from, 'config.yaml'), 'r') as f:
        scrynet_params = yaml.load(f)
    scan_size = scrynet_params['scan_size']
    state_log = np.load(os.path.join(args.restore_from, 'scan.npy'))
    state_log = ((state_log - state_log.min(0)) / state_log.ptp(0) - 0.5) * 2
    num_states = state_log.shape[1]
    num_t = state_log.shape[0]
    print(state_log.T)
    print("states[{}] t[{}]".format(num_states, num_t))
    wave_files = []
    for state_index in range(num_states):
        wave_files.append(sf.SoundFile(os.path.join(args.restore_from, "tsunami{}.wav".format(state_index)), channels=1, samplerate=16000, mode='w'))
    for state_index in range(num_states):
        for t in range(0, num_t):
            wave_files[state_index].write(np.full((scan_size), state_log[t, state_index]))
    for state_index in range(num_states):
        wave_files[state_index].close()

if __name__ == '__main__':
    main()
