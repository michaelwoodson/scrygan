"""
Outputs a script to convert 2 channel wav files to 1 channel wave files.
"""

import fnmatch
import os

for root, dirnames, filenames in os.walk("orchive"):
    for filename in fnmatch.filter(filenames, "*.wav"):
        new_filename = filename[:-4] + ".1c.wav"
        print("sox {} {} remix 1,2".format(filename, new_filename))
