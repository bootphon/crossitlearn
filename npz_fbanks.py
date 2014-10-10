import numpy as np
import sys

a = {}

for fname in sys.argv[1:]:
    fbanks = np.load(fname)
    word = fname.split('_')[0]
    a[word] = fbanks

np.savez('all_words.npz', a)

