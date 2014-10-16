"""python compute_test_set_embeddings.py pascal1k_crossnet_adadelta_emb_200.pickle"""

import cPickle
import sys

import numpy as np
from nnet_archs import CrossNet

with open(sys.argv[1], 'rb') as f:
    nnet = cPickle.load(f)
transform_imgs = nnet.transform_img()
transform_snds = nnet.transform_snd()

dataset_path = "/fhgfs/bootphon/scratch/gsynnaeve/learning_semantics2014/pascal_full/"
corpus_pkl = "corpus_drop.pkl"
lucid_folder = "LUCID_stack_tokens_drop"
test_images = "split_test_img.mat"

from scipy.io import loadmat as loadmat
imgs_emb = {}  # image_name: [20 embeddings]
img_mat_path = dataset_path + test_images
m = loadmat(img_mat_path)
for i in xrange(m['Img'].shape[0]):
    c = m['Img'][i][0]
    fname = c['fname'][0][0][0]
    codes = c['codes'][0][0][:,:-1]  # discard the last feature, it's always 1.
    imgs_emb[fname.split('/')[-1].split('.')[0]] = transform_imgs(np.asarray(codes, dtype='float32'))

import pandas
imgs_tokens_gold = {}  # image_name: [K tokens]
tokens_emb = {}
lucid_path = dataset_path + lucid_folder + '/'
test_imgs = set(imgs_emb.keys())
df = pandas.read_pickle(dataset_path + corpus_pkl)
for i, img_name in enumerate(df['picture']):
    if img_name in test_imgs:
        imgs_tokens_gold[img_name] = []
        for tokens_list_by_same_speaker in df['tokens'][i]:
            imgs_tokens_gold[img_name].append(tokens_list_by_same_speaker)
            for token in tokens_list_by_same_speaker:
                if token not in tokens_emb:
                    tmp = np.asarray(np.load(lucid_path + token + '.npy'), dtype='float32')
                    tokens_emb[token] = transform_snds(np.reshape(tmp, (1, tmp.shape[0])))

with open("imgs_tokens_gold_triplet.pkl", 'wb') as wf:
    cPickle.dump((imgs_emb, tokens_emb, imgs_tokens_gold), wf, -1)

print imgs_emb.keys()
print tokens_emb.keys()
