"""python embed_fbanks.py pascal1k_crossnet_adadelta_emb_200.pickle"""
from __future__ import division

import cPickle as pickle
import sys
import os.path as path
import glob
import operator
import os
from collections import defaultdict

from scipy.spatial.distance import pdist, cdist
from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

from nnet_archs import CrossNet
from dataset_iterators import CrossLearnIterator

CACHE_DIR = '/fhgfs/bootphon/scratch/mwv/speech2image/cache/'
try:
    os.makedirs(CACHE_DIR)
except OSError:
    pass

def tridist_idx(d, i, j, n):
    if j > i:
        return tridist_idx(d, j, i, n)
    if j == i:
        return 0.
    else:
        return d[n*j - j*(j+1)/2 + i - j - 1]

def load_images(img_mat_fname, corpus_df, normalize=True, force_rebuild=False):
    _memo_fname = path.join(CACHE_DIR, 'IMAGES_' + path.splitext(path.basename(img_mat_fname))[0]) + '.pkl'
    if not path.exists(_memo_fname) or force_rebuild:
        used_pics = set(corpus_df['picture'])
        m = loadmat(img_mat_fname)
        fnames = []
        X = np.empty((m['Img'].shape[0] * 20, 4096), dtype=np.float32)
        idx_x = 0
        for idx_m in xrange(m['Img'].shape[0]):
            c = m['Img'][idx_m][0]
            fname = path.splitext(path.basename(c['fname'][0][0][0]))[0]
            if not fname in used_pics:
                continue
            codes = c['codes'][0][0][:, :-1]
            for img_segment in codes:
                fnames.append(fname)
                X[idx_x] = img_segment
                idx_x += 1
        X = X[:idx_x, :]
        assert(len(fnames) == X.shape[0])
        with open(_memo_fname, 'wb') as fid:
            pickle.dump((fnames, X), fid, -1)
    else:
        with open(_memo_fname, 'rb') as fid:
            fnames, X = pickle.load(fid)
    if normalize:
        X = StandardScaler().fit_transform(X)
    return fnames, X


def load_sounds(img_fnames, corpus_df, stackdir, normalize=True, force_rebuild=False):
    _memo_fname = path.join(CACHE_DIR, 'SOUNDS_' + '-'.join(stackdir.split('/')[1:-1])) + '.pkl'
    if not path.exists(_memo_fname) or force_rebuild:
        images = set(img_fnames)
        tokens = set()
        for _, row in corpus_df.iterrows():
            if not row['picture'] in images:
                continue
            for tokenlist in row['tokens']:
                tokens |= set(tokenlist)
        tokens = sorted(list(tokens))
        nfeatures = np.load(path.join(stackdir, tokens[0] + '.npy')).shape[0]
        X = np.empty((len(tokens), nfeatures), dtype=np.float32)
        for i, token in enumerate(tokens):
            if i % 500 == 0:
                print '  loading:', i, token
            X[i] = np.load(path.join(stackdir, token + '.npy'))
        with open(_memo_fname, 'wb') as fid:
            pickle.dump((tokens, X), fid, -1)
    else:
        with open(_memo_fname, 'rb') as fid:
            tokens, X = pickle.load(fid)
    if normalize:
        X = StandardScaler().fit_transform(X)
    return tokens, X

def calc_snd_dists(snd_X, stackdir, nnet_file, force_rebuild=False):
    _memo_fname = path.join(CACHE_DIR,
                            'SOUND_DISTANCES_' +
                            '-'.join(stackdir.split('/')[1:-1]) +
                            '_' +
                            path.splitext(path.basename(nnet_file))[0] +
                            '.npy')
    if not path.exists(_memo_fname) or force_rebuild:
        dists = pdist(snd_X, 'cosine')
        np.save(_memo_fname, dists)
    else:
        dists = np.load(_memo_fname)
    return dists

def calc_img_dists(img_X, img_mat_fname, nnet_file, force_rebuild=False):
    _memo_fname = path.join(CACHE_DIR, 'IMAGE_DISTANCES_' +
                            path.splitext(path.basename(img_mat_fname))[0] +
                            '_' +
                            path.splitext(path.basename(nnet_file))[0] +
                            '.npy')
    if not path.exists(_memo_fname) or force_rebuild:
        dists = pdist(img_X, 'cosine')
        np.save(_memo_fname, dists)
    else:
        dists = np.load(_memo_fname)
    return dists

def calc_multi_dists(img_X, snd_X, stackdir, img_mat_fname, nnet_file, force_rebuild=False):
    _memo_fname = path.join(CACHE_DIR,
                            'MULTI_DISTANCES_' +
                            path.splitext(path.basename(img_mat_fname))[0] +
                            '_' +
                            '-'.join(stackdir.split('/')[1:-1]) +
                            '_' +
                            path.splitext(path.basename(nnet_file))[0] +
                            '.npy')
    if not path.exists(_memo_fname) or force_rebuild:
        dists = cdist(img_X, snd_X, 'cosine')
        np.save(_memo_fname, dists)
    else:
        dists = np.load(_memo_fname)
    return dists

def kneighbors(d, i, k, n):
    """Return the closest k neighbors to element at index 0<=i<n
    according to the triangular distance vector d"""
    return np.argsort(np.vectorize(lambda x: tridist_idx(d, i, x, n))(np.arange(n)))[1:k+1]


def word_similarity_for_image(query_snd, distances, n, snd_fnames, id2pic):
    """find the neighbors of word in the embedded space.
    sort these and find rank of word that describes same image"""
    neighbors = kneighbors(distances, query_snd, n-1, n)
    query_id = snd_fnames[query_snd]
    query_pic = id2pic[query_id]
    rank = n
    for i in xrange(neighbors.shape[0]):
        if len(id2pic[snd_fnames[neighbors[i]]] & query_pic) > 0:
            rank = i+1
            break
    return rank

def image_search_by_word_query(query_snd, multi_distances, snd_fnames, img_fnames, pic2id):
    """map a word query into the embedding space and find images in the same space.
    return rank of first image that has the query in its tokenlist

    multi_distances: n_images x n_sounds
    """
    image_neighbors = np.argsort(multi_distances[:, query_snd])
    query_id = snd_fnames[query_snd]
    n_images, n_sounds = multi_distances.shape
    rank = n_images
    for i in xrange(image_neighbors.shape[0]):
        if query_id in pic2id[img_fnames[image_neighbors[i]]]:
            rank = i+1
            break
    return rank

def image_search_recall_at_k(k, query_snd, multi_distances, snd_fnames, img_fnames, pic2id):
    """is there a hit among the closest k image neighbors for the query sound?"""
    image_neighbors = np.argsort(multi_distances[:, query_snd])[:k]
    query_id = snd_fnames[query_snd]
    for i in xrange(k):
        if query_id in pic2id[img_fnames[image_neighbors[i]]]:
            return True
    return False


def word_search_by_image_query(query_img, multi_distances, snd_fnames, img_fnames, id2pic):
    """map an image query into the embedding space and find words in the same space.
    return rank of first word that is in the tokenlist of the image"""
    word_neighbors = np.argsort(multi_distances[query_img, :])
    query_pic = img_fnames[query_img]
    n_images, n_sounds = multi_distances.shape
    rank = n_images
    for i in xrange(word_neighbors.shape[0]):
        if query_pic in id2pic[snd_fnames[word_neighbors[i]]]:
            rank = i+1
            break
    return rank

def word_search_recall_at_k(k, query_img, multi_distances, snd_fnames, img_fnames, id2pic):
    """is there a hit among the closest k sound neighbors for query image?"""
    word_neighbors = np.argsort(multi_distances[query_img, :])[:k]
    query_pic = img_fnames[query_img]
    for i in xrange(k):
        if query_pic in id2pic[snd_fnames[word_neighbors[i]]]:
            return True
    return False


def mapping_dicts(lucid_tokens_df, corpus_df, lucid2pascal):
    """return id2word, mapping LUCID id's to words, and id2pic, mapping LUCID id's to
    sets of picture names and pic2id, mapping picture names to tokens"""
    id2word = dict(zip(lucid_tokens_df['id'], lucid_tokens_df['word']))
    word2id = defaultdict(set)
    for _, row in lucid_tokens_df.iterrows():
        word2id[lucid2pascal.get(row['word'], row['word'])].add(row['id'])
    word2id.default_factory = None
    word2id = dict(word2id)
    id2pic = defaultdict(set)
    pic2id = defaultdict(set)
    for _, row in corpus_df.iterrows():
        picname = row['picture']
        for token in reduce(operator.add, row['tokens']):
            id2pic[token].add(picname)
            pic2id[picname].add(token)
    id2pic.default_factory = None
    id2pic = dict(id2pic)
    pic2id.default_factory = None
    pic2id = dict(pic2id)

    return id2word, word2id, id2pic, pic2id

def load_net(nnet_file):
    with open(nnet_file, 'rb') as fid:
        nnet = pickle.load(fid)
    return nnet


if __name__ == '__main__':
    dataset_path = "/fhgfs/bootphon/scratch/gsynnaeve/learning_semantics2014/pascal_full/"

    if len(sys.argv) != 6:
        print 'usage: embed_eval.py NNET_FILE CORPUS_FILE TOKEN_FILE STACKTOKENSDIR REPLACEMENTFILE'
        exit()
    print 'loading net...',
    nnet_file = sys.argv[1]
    nnet = load_net(nnet_file)

    transform_imgs = nnet.transform_img()
    transform_snds = nnet.transform_snd()
    transform_both = nnet.transform_img_snd()
    print 'done.'

    print 'loading corpus...',
    corpus_file = sys.argv[2]
    corpus_df = pd.read_pickle(corpus_file)
    print 'done.'

    print 'loading tokens file...',
    tokens_file = sys.argv[3]
    tokens_df = pd.read_pickle(tokens_file)
    print 'done.'

    print 'loading mappings...',
    replacement_file = sys.argv[5]
    lucid2pascal = dict(x.strip().split('\t')[::-1]
                        for x in open(replacement_file))
    id2word, word2id, id2pic, pic2id = mapping_dicts(tokens_df, corpus_df, lucid2pascal)
    print 'done.'

    print 'loading images...',
    img_mat_fname = path.join(dataset_path, 'split_test_img.mat')
    img_fnames, img_X = load_images(img_mat_fname, corpus_df, force_rebuild=True)
    pic2ix = {v:k for k, v in enumerate(img_fnames)}
    print 'done.'

    print 'loading sounds...',
    stack_tokens_dir = sys.argv[4]
    snd_fnames, snd_X = load_sounds(img_fnames, corpus_df, stack_tokens_dir, force_rebuild=True)
    sndID2ix = {v:k for k, v in enumerate(snd_fnames)}
    print 'done.'

    snd_fnames_set = set(snd_fnames)
    id2pic_keys_set = set(id2pic.keys())
    assert(all(f in id2pic_keys_set for f in snd_fnames_set))

    print 'embedding images...',
    img_X_emb = transform_imgs(img_X)
    print 'done.'

    print 'embedding sounds...',
    snd_X_emb = transform_snds(snd_X)
    print 'done.'

    print 'calculating sound distances...',
    snd_dists = calc_snd_dists(snd_X_emb, stack_tokens_dir, nnet_file, force_rebuild=True)
    print 'done.'

    print 'calculating image distances...',
    img_dists = calc_img_dists(img_X_emb, img_mat_fname, nnet_file, force_rebuild=True)
    print 'done.'

    print 'calculating sound-image distances...',
    multi_dists = calc_multi_dists(img_X_emb, snd_X_emb, stack_tokens_dir, img_mat_fname, nnet_file, force_rebuild=True)
    print 'done.'

    print

    # SCORES
    print 'SCORES:'

    # 1. word similarity for image
    word_similarity_ranks = np.array([word_similarity_for_image(i,
                                                                snd_dists,
                                                                snd_X.shape[0],
                                                                snd_fnames,
                                                                id2pic)
                                      for i in xrange(len(snd_fnames))])

    print 'word similarity for image (median, mean rank): {0:.3f} {1:.3f}'.format(
        np.median(word_similarity_ranks), word_similarity_ranks.mean())

    # 2. image search with query words
    image_search_ranks = np.array([image_search_by_word_query(i,
                                                              multi_dists,
                                                              snd_fnames,
                                                              img_fnames,
                                                              pic2id)
                                   for i in xrange(len(snd_fnames))])
    print 'image search by word query (median, mean rank): {0:.3f} {1:3f}'.format(
        np.median(image_search_ranks), image_search_ranks.mean())

    # 3. word search with query image
    word_search_ranks = np.array([word_search_by_image_query(i,
                                                             multi_dists,
                                                             snd_fnames,
                                                             img_fnames,
                                                             id2pic)
                                  for i in xrange(len(img_fnames))])
    print 'word search by image query (median, mean rank): {0:.3f} {1:.3f}'.format(
        np.median(word_search_ranks), word_search_ranks.mean())

    # 4. recall at k

    for k in [1, 5, 10, 25, 50]:
        image_search_recall = image_search_ranks <= k
        # image_search_recall = np.array([image_search_recall_at_k(k,
        #                                                          i,
        #                                                          multi_dists,
        #                                                          snd_fnames,
        #                                                          img_fnames,
        #                                                          pic2id)
        #                                 for i in xrange(len(snd_fnames))])
        print 'image search recall at {0}: {1:.3f}'.format(
            k, np.sum(image_search_recall) * 100 / len(image_search_recall))
    for k in [1, 5, 10, 25, 50]:
        word_search_recall = word_search_ranks <= k
        # word_search_recall = np.array([word_search_recall_at_k(k,
        #                                                        i,
        #                                                        multi_dists,
        #                                                        snd_fnames,
        #                                                        img_fnames,
        #                                                        pic2id)
        #                                for i in xrange(len(img_fnames))])
        print 'word search recall at {0}: {1:.3f}'.format(
            k, np.sum(word_search_recall) * 100 / len(word_search_recall))
