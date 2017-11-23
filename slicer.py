"""Convert audio files into spectrograms.
"""

import argparse
import yaml
import os
import sys
import time
import random

import numpy as np
import soundfile as sf
import fnmatch

from scipy import signal
from scipy import misc

from audio_reader import AudioReader

random.seed(3)

DATA_DIRECTORY = 'orchive'
LOGDIR_ROOT = 'logdir'
PARAMS = 'params.yaml'

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
    parser.add_argument('--params', type=str, default=PARAMS,
                        help='YAML file with parameters to override defaults. Default: ' + PARAMS + '.')
    return parser.parse_args()

def main():
    args = get_arguments()

    with open('default_params.yaml', 'r') as f:
        scrygan_params = yaml.load(f)
    try:
        if args.params:
            with open(args.params, 'r') as f2:
                scrygan_params.update(yaml.load(f2))
    except IOError:
        print("No params file found, using defaults.")
    print("Loaded params: {}".format(yaml.dump(scrygan_params)))

    sample_rate = 16000
    sample_size = scrygan_params["sample_size"]
    file_name = "2006-001A.1c.wav"
    for root, dirnames, filenames in os.walk(args.data_dir):
        for file_name in fnmatch.filter(filenames, "*.wav"):
            wav_file = os.path.join(args.data_dir, file_name)
            print("processing: {}".format(wav_file))
            for idx, offset in enumerate(range(0, 46000000, sample_size)):
                audio, _ = sf.read(wav_file, start=offset, stop=offset + sample_size)
                f, t, Sxx = signal.spectrogram(audio, sample_rate, nperseg=256, nfft=256)
                Sxx = misc.imresize(Sxx, (64, 64))
                np.save(os.path.join(args.data_dir, "{}.{}.npy".format(file_name, idx)), Sxx)


if __name__ == '__main__':
    main()
