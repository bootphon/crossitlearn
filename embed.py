"""python embed_fbanks.py pascal1k_crossnet_adadelta_emb_200.pickle"""

import cPickle, sys
import numpy as np
from nnet_archs import CrossNet

with open(sys.argv[1], 'rb') as f:
    nnet = cPickle.load(f)

transform_imgs = nnet.transform_img()
transform_snds = nnet.transform_snd()
transform_both = nnet.transform_img_snd()

# TODO if needed, normalize data if it was not

for batch_of_img in images_features:
    # batch_of_img has to be a (n_images, 4096) ndarray of float32
    # look at dataset_iterators.py:47-56 to get the images features
    embedded_images = transform_imgs(batch_of_img)

for batch_of_snd in words_features:
    # batch_of_snd has to be a (n_words, 40*71) ndarray of float32
    # look at dataset_iterators.py:61-67 to get the speech features
    embedded_words = transform_snds(batch_of_snd)

# TODO note: you could even reuse from dataset_iterators import CrossLearnIterator

for batch_of_img, batch_of_snd in zip(images_features, words_features):
    embedded_images, embedded_words = transform_both(batch_of_img,
            batch_of_snd)
