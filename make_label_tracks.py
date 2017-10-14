"""Create labels for when activations change in outputs from the RNN.

The labels can be read in via Audacity as label tracks and viewed alongside a spectrogram.
Note that before running the scan.py script must be run to create the scan.npy
record of outputs from a trained model. This script didn't prove very useful
in practice because it takes some analysis to figure out what level to trigger
meaningful state changes and it varies for each output.  Might be useful
with some modifications to make labels for a particular output.

See tsunami.py for an alternative that generates wav files for each output.
"""

import argparse
import yaml
import os
import sys

import numpy as np

DATA_DIRECTORY = 'orchive'
LOGDIR_ROOT = 'logdir'

def get_arguments():
    parser = argparse.ArgumentParser(description='ScryNet label track creation script')
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
    state_log = (state_log - state_log.min(0)) / state_log.ptp(0)
    num_states = state_log.shape[1]
    num_t = state_log.shape[0]
    print(state_log.T)
    print("states[{}] t[{}]".format(num_states, num_t))
    def time(t):
        return t * scan_size / 16000
    for state_index in range(num_states):
        flag1 = False
        flag2 = False
        f1 = open(os.path.join(args.restore_from, "labels.{}.1.txt".format(state_index)), "w")
        f2 = open(os.path.join(args.restore_from, "labels.{}.2.txt".format(state_index)), "w")
        # Hardcoded thresholds that need to be tuned for meaningful labels.
        for t in range(0, num_t):
            value = state_log[t, state_index]
            if not flag1 and value > 0.7:
                start1 = time(t)
                flag1 = True
            if flag1 and value < 0.7:
                f1.write("{:.7f}\t{:.7f}\tPew\n".format(start1, time(t)))
                flag1 = False
            if not flag2 and value < 0.3:
                start2 = time(t)
                flag2 = True
            if flag2 and value > 0.3:
                f2.write("{:.7f}\t{:.7f}\Pew\n".format(start2, time(t)))
                flag2 = False
        f1.close()
        f2.close()

if __name__ == '__main__':
    main()
