import os
import sys
import subprocess
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
print("All libs succesfully imported!")
import mxnet as mx

im2rec_path = mx.test_utils.get_im2rec_path()

data_path = '/home/sofus/deep/data'
prefix_path = '/home/sofus/deep/data/concrete_data'


with open(os.devnull, 'wb') as devnull:
    subprocess.check_call(['python', im2rec_path, '--list', '--recursive', '--test-ratio=0.1', '--train-ratio=0.525', prefix_path, data_path],
                          stdout=devnull)


with open(os.devnull, 'wb') as devnull:
    subprocess.check_call(['python', im2rec_path, '--num-thread=0', '--pass-through', '--test-ratio=0.1', '--train-ratio=0.525', prefix_path, data_path],
                          stdout=devnull)

