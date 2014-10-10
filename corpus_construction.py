#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: corpus_construction.py
# date: Wed October 08 19:34 2014
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""corpus_construction:

"""

from __future__ import division

import os
import os.path as path
import string
from collections import defaultdict, Counter
import operator
import glob
import re

import nltk
import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()
singularize = wnl.lemmatize

LUCID_DIR = path.join(os.environ['HOME'], 'data', 'LUCID')
PASCAL1K_DIR = path.join(os.environ['HOME'], 'data', 'pascal1k')


GOOD_FNAMES = []
for line in open('fnames.txt'):
    GOOD_FNAMES.append(path.splitext(line.strip())[0])
GOOD_FNAMES = set(GOOD_FNAMES)


# only content tags
GOOD_TAGS = set(['JJ', 'NN', 'RB', 'VB'])

# a bunch of words that are very common in LUCID, but that I don't want
BAD_WORDS  = ['got', 'have', 'has', 'was', 'were', 'yes', 'yeah', 'okay', 'um',
              'er', 'then', 'oh', 'yep', 'my', 'mine', 'you', 'your', 'he',
              'she', 'his', 'her', 'it', 'we', 'our', 'they', 'their', 'ah',
              'uh', 'err', 't', 'am', 'are', 'is', 'eh', 'ag', 'ay', 'be',
              'da'] + list(string.ascii_lowercase)

def load_lucid_df():
    """load lucid tokens file and clean it up
    """
    token_file = path.join(LUCID_DIR, 'lucid_native.tokens.txt')
    df = pd.read_table(token_file, header=0,
                       names=['filename',
                              'onset',
                              'offset',
                              'word',
                              'talker',
                              'task',
                              'condition',
                              'word_length'])
    # # for some reason, column 1 and column 7 don't load to the right types
    # df['onset'] = df['onset'].apply(float)
    # df['word_length'] = df['word_length'].apply(int)

    # clean up and lowercase the word column
    df['word'] = df['word'].map(str).map(str.lower)

    df = df[df['filename'].isin(GOOD_FNAMES)]

    # drop words with !, <, >, -, '
    bad_symbols_p = re.compile(r"^[^'!<\->]*$")
    df = df[df['word'].str.match(bad_symbols_p)]

    # drop words not in LUCID dictionary
    dict_df = pd.read_table(path.join(LUCID_DIR, 'lucid_native.words.txt'),
                            names=['word', 'phon_trs', 'frequency', 'length'],
                            header=0)
    words = set(dict_df['word'].unique())
    df = df[df['word'].isin(words)]

    # drop bad words
    df = df[~df['word'].isin(BAD_WORDS)]

    # drop words that don't have a tag in GOOD_TAGS
    vocab = set(df['word'].unique())
    tags = dict(reduce(operator.add, (nltk.tag.pos_tag([w]) for w in vocab)))
    tags = {k: v[:2] for k, v in tags.iteritems()}
    df['tag'] = df['word'].apply(tags.__getitem__)
    df = df[df['tag'].isin(GOOD_TAGS)]

    # singularize
    df.loc[df['tag'] == 'NN', 'word'] = df[df['tag'] == 'NN']['word'].apply(singularize)

    # add ID column
    df['id'] = ['LUCID{0:06d}'.format(i+1) for i in range(len(df))]
    return df.reset_index(drop=True)


def load_pascal_df():
    """Load pascal as a dataframe with columns:
    1. filename
    2. original sentence
    3. stripped sentence
    4. original tags
    5. stripped tags

    """
    # processing fragments as tuple
    # fragment[0] = filename
    # fragment[1] = original sentence, tokenized and lowered
    # fragment[2] = processed sentence (stripped, etc)
    fragments = []
    for fname in glob.iglob(path.join(PASCAL1K_DIR, '*.txt')):
        bname = path.splitext(path.basename(fname))[0]
        for line in open(fname):
            fragments.append((bname,
                              line.strip()
                              .translate(None,
                                         string.punctuation)))
    # tokenize sentences and lower
    fragments = [(f[0],
                  tuple([word.lower()
                         for word in nltk.word_tokenize(f[1])]))
                 for f in fragments]

    # filter out bad tags
    vocab = set(w for f in fragments for w in f[1])
    tags = dict(reduce(operator.add, (nltk.tag.pos_tag([w]) for w in vocab)))
    tags = {k: v[:2] for k, v in tags.iteritems()}
    fragments = [(f[0],
                  f[1],
                  filter(lambda x: tags[x] in GOOD_TAGS,
                         f[1]),
                  [tags[x] for x in f[1]])
                 for f in fragments]

    # filter out bad words
    fragments = [(f[0],
                  f[1],
                  filter(lambda x: not x in BAD_WORDS,
                         f[2]),
                  f[3]
                  )
                 for f in fragments]

    # remove words with article attached and true hapaxes
    vocab = set(w for f in fragments for w in f[2])
    allwordcounts = Counter(reduce(operator.add, zip(*fragments)[2]))
    def check_word(word):
        if allwordcounts[word] > 2:
            return word
        if word.startswith('an'):
            if word[2:] in vocab:
                return word[2:]
            else:
                return ''
        if word.startswith('a'):
            if word[1:] in vocab:
                return word[1:]
            else:
                return ''
        if word.startswith('the'):
            if word[3:] in vocab:
                return word[3:]
            else:
                return ''
        return ''
    fragments = [(f[0],
                  f[1],
                  map(check_word,
                      f[2]),
                  f[3]) for f in fragments]

    # remove '' words
    fragments = [(f[0],
                  f[1],
                  filter(lambda x: x != '',
                         f[2]),
                  f[3]) for f in fragments]

    # remake tags dict
    vocab = set(w for f in fragments for w in f[2])
    tags = {k: v for k, v in tags.iteritems() if k in vocab}

    df = pd.DataFrame(fragments,
                      columns=['filename', 'orig_sent', 'stripped_sent', 'orig_tags'])
    df['stripped_tags'] = df['stripped_sent'].apply(lambda x: map(lambda y: tags[y], x))

    rlist = []
    for wordlist, taglist in zip(df['stripped_sent'], df['stripped_tags']):
        wlist = []
        for word, tag in zip(wordlist, taglist):
            if tag == 'NN':
                wlist.append(singularize(word))
            else:
                wlist.append(word)
        rlist.append(wlist)
    df['stripped_sent'] = rlist

    return df


def find_replacements(lucid_df, pascal_df, matchup=False):
    """Find replacements for words that occur in pascal1k, but not in lucid.
    Replacement words are matched for part-of-speech, frequency and length.

    Arguments:
    :param lucid_df:
    :param pascal_df:
    :param pascal_tags:
    """
    # get vocab and counts
    lucid_counts = lucid_df['word'].value_counts().to_dict()
    pascal_words = [word for sentence in pascal_df['stripped_sent']
                    for word in sentence]
    pascal_counts = Counter(pascal_words)
    lucid_vocab = set(lucid_df['word'].unique())
    pascal1k_vocab = set(pascal_words)
    lucid_tags = dict(zip(lucid_df['word'], lucid_df['tag']))
    pascal_tags = dict(zip(reduce(operator.add, pascal_df['stripped_sent']),
                           reduce(operator.add, pascal_df['stripped_tags'])))

    # collect total counts for POS tags
    pascal_tag_totals = defaultdict(int)
    for tag in GOOD_TAGS:
        pascal_tag_totals[tag] += sum([v for k, v in pascal_counts.iteritems()
                                         if pascal_tags[k] == tag])
    lucid_tag_totals = lucid_df['tag'].value_counts().to_dict()

    # collect missing and extra words
    bins = [1, 3, 5, 7, 9, 11, 30]
    missing_words = pd.DataFrame([(word,
                                   pascal_counts[word]/pascal_tag_totals[pascal_tags[word]],
                                   pascal_counts[word]/len(pascal_words),
                                   pascal_tags[word],
                                   len(word))
                                  for word in pascal1k_vocab - lucid_vocab],
                                 columns=['word', 'frequency_per_tag', 'frequency', 'tag', 'length'])
    missing_words['bin'] = np.digitize(missing_words['length'],
                                       bins=bins)
    extra_words = pd.DataFrame([(word,
                                 lucid_counts[word]/lucid_tag_totals[lucid_tags[word]],
                                 lucid_counts[word]/len(lucid_df),
                                 lucid_tags[word],
                                 len(word))
                                for word in lucid_vocab - pascal1k_vocab],
                               columns=['word', 'frequency_per_tag', 'frequency', 'tag', 'length'])
    extra_words['bin'] = np.digitize(extra_words['length'],
                                     bins=bins)

    if matchup:
        # map missing words onto extra words matched by tag and bin,
        # ranked by frequency within tag

        replacements  = {}
        for tag in missing_words['tag'].unique():
            for bin in missing_words['bin'].unique():
                missing = missing_words[(missing_words['tag'] == tag) &
                                        (missing_words['bin'] == bin)] \
                    .sort('frequency_per_tag', ascending=False)['word']
                extra = extra_words[(extra_words['tag'] == tag) &
                                    (extra_words['bin'] == bin)] \
                    .sort('frequency_per_tag', ascending=False)['word']
                replacements.update(dict(zip(missing, extra)))
    else:
        # replace however
        replacements = dict(zip(missing_words.sort('frequency', ascending=False)['word'],
                                extra_words.sort('frequency', ascending=False)['word']))
    return replacements, extra_words[~extra_words['word'].isin(replacements.values())]


def match_corpora(lucid_df, pascal_df):
    """return 2 dataframes:

    1. lucid_tokens, fields: ID, filename, onset, offset, word, talker, tag
    2. corpus, fields: picture, orig_sent, mapped_sent, [[lucid_tokens]]
    """
    speakers = set(lucid_df['talker'].unique())
    words_per_speaker = {s: set(lucid_df[lucid_df['talker'] == s]['word'].unique())
                         for s in speakers}

    lucid_tokens = pd.DataFrame(lucid_df.loc[:, ['id', 'filename',
                                                 'onset', 'offset',
                                                 'word', 'talker', 'tag']])
    lucid_tokens.reset_index(drop=True, inplace=True)

    r = []
    missed = 0
    for idx, row in pascal_df.iterrows():
        if idx % 25 == 0:
            print '  processing {0}/{1} - missed: {2}'.format(
                idx, len(pascal_df), missed
            )
        pic_id = row['filename']
        words = tuple(row['replaced_sent'])
        token_lists = []
        for speaker in sorted(speakers,
                              key=lambda x: len(words_per_speaker[x]),
                              reverse=True):
            if not all(word in words_per_speaker[speaker]
                       for word in words):
                continue
            tokens = []
            for word in words:
                # pick a random instance of this word for this speaker
                inds = lucid_df[(lucid_df['talker'] == speaker) &
                                (lucid_df['word'] == word)].index
                tokens.append(lucid_df.loc[np.random.choice(inds)].id)
            token_lists.append(tokens)
        if len(token_lists) == 0:
            missed += 1
            continue
        r.append([pic_id, words, token_lists])
    r = pd.DataFrame(r, columns=['picture', 'sentence', 'tokens'])
    return r, lucid_tokens, missed


if __name__ == '__main__':
    print 'loading lucid_df'
    df_dir = 'datafiles'
    try:
        os.makedirs(df_dir)
    except OSError:
        pass
    _lucid_df_pkl = path.join(df_dir, 'lucid_df.pkl')
    if not path.exists(_lucid_df_pkl):
        lucid_df = load_lucid_df()
        lucid_df.to_pickle(_lucid_df_pkl)
    else:
        lucid_df = pd.read_pickle(_lucid_df_pkl)

    print 'loading pascal1k annotations'
    _pascal_df_pkl = path.join(df_dir, 'pascal_df.pkl')
    if not path.exists(_pascal_df_pkl):
        pascal_df = load_pascal_df()
        pascal_df.to_pickle(_pascal_df_pkl)

    else:
        pascal_df = pd.read_pickle(_pascal_df_pkl)

    replacements, _ = find_replacements(lucid_df, pascal_df, matchup=True)
    pascal_df['replaced_sent'] = pascal_df['stripped_sent']\
        .apply(lambda x: tuple(map(lambda y: replacements.get(y, y), x)))
    with open(path.join(df_dir, 'pascal1k_sentences.txt'), 'w') as fid:
        fid.write('#filename\toriginal_sentence\tstripped_sentence'
                  '\treplaced_sentence\n')
        for idx, row in pascal_df.iterrows():
            fid.write('{0}\t{1}\t{2}\t{3}\n'.format(
                row['filename'],
                ','.join(row['orig_sent']),
                ','.join(row['stripped_sent']),
                ','.join(row['replaced_sent'])))

    with open(path.join(df_dir, 'replacements.txt'), 'w') as fid:
        fid.write('\n'.join('\t'.join(i) for i in replacements.iteritems()))

    print 'matching up corpora'
    corpus, lucid_tokens, missed = match_corpora(lucid_df, pascal_df)
    print 'missed sentences:', missed

    corpus.to_pickle(path.join(df_dir, 'corpus.pkl'))
    lucid_tokens.to_csv(path.join(df_dir, 'lucid_tokens.csv'))
    lucid_tokens.to_pickle(path.join(df_dir, 'lucid_tokens.pkl'))
