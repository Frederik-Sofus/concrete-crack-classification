#!/bin/python
import glob
import os
import shutil
import re

# This script takes a path and copies all the images to 
# a training, test and validation folder split into binary categories

# Rename paths to match where to find the images
path = '/home/sofus/deep/data/Negative/'
path = '/home/sofus/deep/data/Positive/'

# TODO make the path string into variable in loops
for filename in glob.glob(os.path.join(path, '*.jpg')):
    digits =  re.findall(r'\d+', filename)
    if int(digits[0]) < 10501:
        shutil.copy(filename, '/home/sofus/deep/data/train/Negative')
    if int(digits[0]) > 10501 and int(digits[0]) < 14501:
        shutil.copy(filename, '/home/sofus/deep/data/test/Negative')
    if int(digits[0]) > 14501 and int(digits[0]) < 18501:
        shutil.copy(filename, '/home/sofus/deep/data/val/Negative')


for filename in glob.glob(os.path.join(path, '*.jpg')):
    digits =  re.findall(r'\d+', filename)
    if int(digits[0]) < 10501:
        shutil.copy(filename, '/home/sofus/deep/data/train/Positive')
    if int(digits[0]) > 10501 and int(digits[0]) < 14501:
        shutil.copy(filename, '/home/sofus/deep/data/test/Positive')
    if int(digits[0]) > 14501 and int(digits[0]) < 18501:
        shutil.copy(filename, '/home/sofus/deep/data/val/Positive')
