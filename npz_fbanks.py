import numpy as np
import sys
import os

a = {}

folder = sys.argv[1].rstrip('/')

for fname in os.listdir(folder):
    fullfname = folder + '/' + fname
    fbanks = np.load(fullfname)
    word = fname.split('/')[-1]
    a[word] = fbanks

np.savez('all_words.npz', **a)

