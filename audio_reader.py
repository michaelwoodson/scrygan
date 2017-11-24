import fnmatch
import os
import random

import soundfile as sf
import numpy as np

random.seed(0)

def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


class AudioReader(object):
    '''Randomly load files from a directory and load samples into a batch.
       Note that all the files are assumed to be at least 48 minutes and 16000
       sample rate.'''

    def __init__(self,
                 audio_dir,
                 batch_size,
                 sample_size,
                 num_t):
        self.audio_dir = audio_dir
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.num_t = num_t
        self.files = find_files(audio_dir)
        print("files length: {}".format(len(self.files)))
        print("batches: {}".format(batch_size))
        if not self.files:
            raise ValueError("No audio files found in '{}'.".format(audio_dir))

    def get_batch(self):
        batch = []
        for filename in randomize_files(self.files):
            size = int(self.sample_size * (self.num_t / 2)) + int(self.sample_size/2)
            #print('size: {}'.format(size))
            start = random.randint(0, 46000000 - size)
            audio, _ = sf.read(filename, start=start, stop = start + size)
            batch.append(audio)
            if len(batch) == self.batch_size:
                break
        return batch
