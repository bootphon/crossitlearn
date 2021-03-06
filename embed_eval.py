"""python embed_fbanks.py pascal1k_crossnet_adadelta_emb_200.pickle"""
from __future__ import division

import cPickle as pickle
import sys
import os.path as path
import glob
import operator
import os
from collections import defaultdict

import joblib

from scipy.spatial.distance import pdist, cdist
from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

from nnet_archs import CrossNet
from dataset_iterators import CrossLearnIterator

CACHE_DIR = os.getcwd()
# try:
#     os.makedirs(CACHE_DIR)
# except OSError:
#     pass

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

def image_TOKEN_search_by_word_query_TOKEN(query_snd_ix, multi_distances,
                                           snd_fnames, img_fnames,
                                           id2pic):
    """map a word token query into the embedding space and find images in the same space
    return rank of first neighbor whose TOKEN is in the picture list of the id"""
    n_images, n_sounds = multi_distances.shape
    query_id = snd_fnames[query_snd_ix]

    img_neighbors = np.argsort(multi_distances[:, query_snd_ix])

    pictures_for_query = id2pic[query_id]

    rank = img_neighbors.shape[0]
    for i in xrange(img_neighbors.shape[0]):
        if img_fnames[img_neighbors[i]] in pictures_for_query:
            rank = i + 1
            break
    return rank


def image_TOKEN_search_by_word_query_TYPE_at_k(k, word_TYPE, multi_distances,
                                               snd_fnames, img_fnames,
                                               word2id, id2pic):
    """map all word tokens corresponding to word_TYPE into the embedding space and take
    the k nearest neighbors for each token. concatenate the neighbors. return rank of the
    first neighbor whose token is in the concatenated picture lists"""
    n_images, n_sounds = multi_distances.shape
    word_tokens = word2id[word_TYPE]
    word_tokens &= set(snd_fnames)

    if len(word_tokens) == 0:
        raise ValueError('not enough word tokens for {0}'.format(word_TYPE))

    pictures_for_word = set()
    for token in word_tokens:
        pictures_for_word.update(id2pic[token])

    id2ix = {v: ix for ix, v in enumerate(snd_fnames)}

    image_token_ix = None
    image_token_dists = None

    for token in word_tokens:
        # find k neighbors, keep their ix and distance
        nn = multi_distances[:, id2ix[token]]
        nn_ix = np.argsort(nn)[:k]
        nn_dists = nn[nn_ix]
        if image_token_ix is None:
            image_token_ix = nn_ix
            image_token_dists = nn_dists
        else:
            image_token_ix = np.hstack((image_token_ix, nn_ix))
            image_token_dists = np.hstack((image_token_dists, nn_dists))

    sort_ix = np.argsort(image_token_dists)
    rank = sort_ix.shape[0]
    for ix in sort_ix:
        if img_fnames[image_token_ix[ix]] in pictures_for_word:
            rank = ix + 1
            break
    return rank


def word_TOKEN_search_by_image_query_TOKEN(query_img_ix, multi_distances,
                                           snd_fnames, img_fnames,
                                           pic2id):
    """map an image query into the embedding space and find words in the same space.
    return rank of first neighbor whose TOKEN is in the tokenlist of the image"""
    n_images, n_sounds = multi_distances.shape

    word_neighbors = np.argsort(multi_distances[query_img_ix, :])
    query_pic = img_fnames[query_img_ix]
    tokens_for_query = pic2id[query_pic]

    rank = n_sounds
    for i in xrange(word_neighbors.shape[0]):
        if snd_fnames[word_neighbors[i]] in tokens_for_query:
            rank = i + 1
            break
    return rank

def word_TYPE_search_by_image_query_TOKEN(query_img_ix, multi_distances,
                                          snd_fnames, img_fnames,
                                          pic2id, id2word):
    """map an image query into the embedding space and find words in the same space.
    return rank of first neighbor whose TYPE is in the tokenlist of the image"""
    n_images, n_sounds = multi_distances.shape

    word_neighbors = np.argsort(multi_distances[query_img_ix, :])
    query_pic = img_fnames[query_img_ix]
    tokens_for_query = pic2id[query_pic]
    words_for_query = set([id2word[x] for x in tokens_for_query])

    rank = n_sounds
    for i in xrange(word_neighbors.shape[0]):
        if id2word[snd_fnames[word_neighbors[i]]] in words_for_query:
            rank = i + 1
            break
    return rank





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
    if path.splitext(nnet_file)[1] == 'joblib':
        nnet = joblib.load(nnet_file)
    else:
        with open(nnet_file, 'rb') as fid:
            nnet = pickle.load(fid)
    return nnet


if __name__ == '__main__':
    datadir = '/mnt/data/pascal1k_prepared_dataset/'
    stackdir = path.join(datadir, 'LUCID_stack_tokens_drop')
    # if len(sys.argv) != 6:
    #     print 'usage: embed_eval.py NNET_FILE CORPUS_FILE TOKEN_FILE STACKTOKENSDIR REPLACEMENTFILE'
    #     exit()
    print 'loading net...',
    nnet_file = 'full_pascal1k_crossnet_adadelta_emb_200.pickle'
    nnet = load_net(nnet_file)

    transform_imgs = nnet.transform_img()
    transform_snds = nnet.transform_snd()
    transform_both = nnet.transform_img_snd()
    print 'done.'

    print 'loading corpus...',
    corpus_file = path.join(datadir, 'corpus_drop.pkl')
    corpus_df = pd.read_pickle(corpus_file)
    print 'done.'

    print 'loading tokens file...',
    tokens_file = path.join(datadir, 'lucid_tokens_drop.pkl')
    tokens_df = pd.read_pickle(tokens_file)
    print 'done.'

    print 'loading mappings...',
    replacement_file = path.join(datadir, 'replacements.txt')
    lucid2pascal = dict(x.strip().split('\t')[::-1]
                        for x in open(replacement_file))
    pascal2lucid = dict(x.strip().split('\t')
                        for x in open(replacement_file))
    id2word, word2id, id2pic, pic2id = mapping_dicts(tokens_df, corpus_df, lucid2pascal)
    print 'done.'

    print 'loading images...',
    img_mat_fname = '/mnt/data/pascal1k_prepared_dataset/split_test_img.mat'
    img_fnames, img_X = load_images(img_mat_fname, corpus_df, force_rebuild=True)
    pic2ix = {v:k for k, v in enumerate(img_fnames)}
    print 'done.'

    print 'loading sounds...',
    stack_tokens_dir = '/mnt/data/pascal1k_prepared_dataset/LUCID_stack_tokens_drop/'
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


    # 2. word TOKEN search by image TOKEN
    word_token_image_token_ranks = np.array(
        [word_TOKEN_search_by_image_query_TOKEN(i, multi_dists, snd_fnames, img_fnames, pic2id)
         for i in xrange(len(img_fnames))])

    print 'word TOKEN search by image TOKEN (median, mean rank): {0:.3f} {1:.3f}'.format(
        np.median(word_token_image_token_ranks), word_token_image_token_ranks.mean())

    # 3. word TYPE search by image TOKEN
    word_type_image_token_ranks = np.array(
        [word_TYPE_search_by_image_query_TOKEN(i, multi_dists, snd_fnames, img_fnames, pic2id, id2word)
         for i in xrange(len(img_fnames))])

    print 'word TYPE search by image TOKEN (median, mean rank): {0:.3f} {1:.3f}'.format(
        np.median(word_type_image_token_ranks), word_type_image_token_ranks.mean())

    # 4. image TOKEN search by word TOKEN
    image_token_word_token_ranks = np.array(
        [image_TOKEN_search_by_word_query_TOKEN(i, multi_dists, snd_fnames, img_fnames, id2pic)
         for i in xrange(len(snd_fnames))])

    print 'image TOKEN search by word TOKEN (median, mean rank): {0:.3f} {1:.3f}'.format(
        np.median(image_token_word_token_ranks), image_token_word_token_ranks.mean())

    # 5. image TYPE search by word TOKEN

    # RECALL AT K
    for k in [1, 5, 10, 25, 50]:
        word_token_image_token_recall = \
            100 * np.sum(word_token_image_token_ranks <= k) / len(word_token_image_token_ranks)
        print 'word token search by image token recall at {0:d}: {1:.3f}'.format(
            k, word_token_image_token_recall)
    for k in [1, 5, 10, 25, 50]:
        word_type_image_token_recall = \
            100 * np.sum(word_type_image_token_ranks <= k) / len(word_type_image_token_ranks)
        print 'word type search by image token recall at {0:d}: {1:.3f}'.format(
            k, word_type_image_token_recall)
    for k in [1, 5, 10, 25, 50]:
        image_token_word_token_recall = \
            100 * np.sum(image_token_word_token_ranks <= k) / len(image_token_word_token_ranks)
        print 'image token search by word token recall at {0:d}: {1:.3f}'.format(
            k, image_token_word_token_recall)
    # for k in [1, 5, 10, 25, 50]:
    image_token_word_type_ranks = \
        np.array([image_TOKEN_search_by_word_query_TYPE_at_k(200, word, multi_dists,
                                                             snd_fnames, img_fnames,
                                                             word2id, id2pic)
                  for word in [lucid2pascal.get(id2word[ID], id2word[ID])
                               for ID in set(snd_fnames)]])
    for k in [1, 5, 10, 25, 50]:
        image_token_word_type_recall = \
            100 * np.sum(image_token_word_type_ranks <= k) / len(image_token_word_type_ranks)
        print 'image token search by word type recall at {0:d}: (median, mean) {1:.3f}'.format(
            k, image_token_word_type_recall)

    # 2. image search with query words
    # image_search_ranks = np.array([image_search_by_word_query(i,
    #                                                           multi_dists,
    #                                                           snd_fnames,
    #                                                           img_fnames,
    #                                                           pic2id)
    #                                for i in xrange(len(snd_fnames))])
    # print 'image search by word query (median, mean rank): {0:.3f} {1:3f}'.format(
    #     np.median(image_search_ranks), image_search_ranks.mean())

    # # 3. word search with query image
    # word_search_ranks = np.array([word_search_by_image_query_token(i,
    #                                                                multi_dists,
    #                                                                snd_fnames,
    #                                                                img_fnames,
    #                                                                id2pic)
    #                               for i in xrange(len(img_fnames))])
    # print 'word search by image query (median, mean rank): {0:.3f} {1:.3f}'.format(
    #     np.median(word_search_ranks), word_search_ranks.mean())

    # # 4. recall at k

    # for k in [1, 5, 10, 25, 50]:
    #     image_search_recall = image_search_ranks <= k
    #     # image_search_recall = np.array([image_search_recall_at_k(k,
    #     #                                                          i,
    #     #                                                          multi_dists,
    #     #                                                          snd_fnames,
    #     #                                                          img_fnames,
    #     #                                                          pic2id)
    #     #                                 for i in xrange(len(snd_fnames))])
    #     print 'image search recall at {0}: {1:.3f}'.format(
    #         k, np.sum(image_search_recall) * 100 / len(image_search_recall))
    # for k in [1, 5, 10, 25, 50]:
    #     word_search_recall = word_search_ranks <= k
    #     # word_search_recall = np.array([word_search_recall_at_k(k,
    #     #                                                        i,
    #     #                                                        multi_dists,
    #     #                                                        snd_fnames,
    #     #                                                        img_fnames,
    #     #                                                        pic2id)
    #     #                                for i in xrange(len(img_fnames))])
    #     print 'word search recall at {0}: {1:.3f}'.format(
    #         k, np.sum(word_search_recall) * 100 / len(word_search_recall))
