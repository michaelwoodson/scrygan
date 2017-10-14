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
                 sample_rate,
                 batch_size,
                 num_mini_batches,
                 mini_batch_size,
                 window_size):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.num_mini_batches = num_mini_batches
        self.mini_batch_size = mini_batch_size
        self.window_size = window_size
        files = find_files(audio_dir)
        if not files:
            raise ValueError("No audio files found in '{}'.".format(audio_dir))

    def get_batches(self):
        batch = []
        for audio in self.load_generic_audio():
            batch.append(audio)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

    def load_generic_audio(self):
        '''Generator that yields audio waveforms from the directory.'''
        files = find_files(self.audio_dir)
        print("files length: {}".format(len(files)))
        randomized_files = randomize_files(files)
        for filename in randomized_files:
            size = (self.num_mini_batches * self.mini_batch_size + 1 + (self.window_size - 1))
            start = random.randint(0, 46000000 - size)
            audio, _ = sf.read(filename, start=start, stop = start + size)
            yield audio
