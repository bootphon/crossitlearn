MIN_FRAMES_PER_SENTENCE = 26
BATCH_SIZE = 100
import numpy, theano, joblib
from collections import defaultdict
import random, joblib, math, sys, os


class DatasetMiniBatchIterator(object):
    """ Basic mini-batch iterator """
    def __init__(self, x, y, batch_size=BATCH_SIZE, randomize=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.randomize = randomize
        from sklearn.utils import check_random_state
        self.rng = check_random_state(42)

    def __iter__(self):
        n_samples = self.x.shape[0]
        if self.randomize:
            for _ in xrange(n_samples / BATCH_SIZE):
                if BATCH_SIZE > 1:
                    i = int(self.rng.rand(1) * ((n_samples+BATCH_SIZE-1) / BATCH_SIZE))
                else:
                    i = int(math.floor(self.rng.rand(1) * n_samples))
                yield (i, self.x[i*self.batch_size:(i+1)*self.batch_size],
                       self.y[i*self.batch_size:(i+1)*self.batch_size])
        else:
            for i in xrange((n_samples + self.batch_size - 1)
                            / self.batch_size):
                yield (self.x[i*self.batch_size:(i+1)*self.batch_size],
                       self.y[i*self.batch_size:(i+1)*self.batch_size])



class CrossLearnIterator(object):
    """ Iterates over same/different pairs of images and sounds features """
    def __init__(self, img_mat_path, corpus_pkl_path, batch_size=BATCH_SIZE):
        self.batch_size = batch_size
#        >>> a['Img'][0][0]['codes'][0][0].shape
#        (20, 4097)
#        >>> a['Img'][1][0]['codes'][0][0].shape
#        (20, 4097)
#        >>> a['Img'][1][0]['fname']
#        array([[array([u'imgs/2008_003703.jpg'],
#                  dtype='<U20')]], dtype=object)
        print >> sys.stderr, "loading img features"
        from scipy.io import loadmat as loadmat
        self.imgs = {}
        m = loadmat(img_mat_path)
        for i in xrange(m['Img'].shape[0]):
            c = m['Img'][i][0]
            fname = c['fname'][0][0][0]
            codes = c['codes'][0][0][:,:-1]  # discard the last feature, it's always 1.
            self.imgs[fname.split('/')[-1]] = codes
        joblib.dump(self.imgs, 'imgs.joblib')

        USE_DUMPED_FBANKS = False  # TODO MAKE THIS True FOR SPEED!!!
        if not USE_DUMPED_FBANKS:
            print >> sys.stderr, "loading fbanks"
            self.fbanks = {}
            STACKS_FOLDER = '/fhgfs/bootphon/scratch/mwv/LUCID/stack_tokens/'
            for fname in os.listdir(STACKS_FOLDER):
                if "npy" in fname:
                    self.fbanks[fname.split('.')[0]] = numpy.load(STACKS_FOLDER + fname)
            joblib.dump(self.fbanks, 'fbanks.joblib')
        else:
            self.fbanks = joblib.load('fbanks.joblib')

        print >> sys.stderr, "loading corpus"
        import pandas
        self.snds = defaultdict(lambda: [])
        df = pandas.read_pickle(corpus_pkl_path)
        for i, fname in enumerate(df['picture']):
            #self.snds[fname].extend([numpy.load(STACKS_FOLDER + x)
            #    for x in df['tokens'][i][0]])
            self.snds[fname].extend([x for x in df['tokens'][i][0]])
            # TODO TODO TODO df['tokens'][i][j] with j iterating over speakers
        #joblib.dump(self.snds, 'snds.joblib')

    def __iter__(self):
        for img_name, img_feats in self.imgs.iteritems():
            good_pairs = []
            bad_pairs = []
            n_img_features = img_feats.shape[1]
            n_fbanks_features = 0
            img_name = img_name.split('.')[0]
            for img_feat in img_feats:
                #print img_feat
                for fbanksname in self.snds[img_name]:
                    tmpfb = self.fbanks[fbanksname]
                    n_fbanks_features = tmpfb.shape[0]
                    good_pairs.append((img_feat, tmpfb))
                    lkeys = len(self.fbanks.keys())
                    bad_fbanksname = self.fbanks.keys()[random.randint(0, lkeys-1)]
                    while bad_fbanksname in self.snds[img_name]:
                        bad_fbanksname = self.fbanks.keys()[random.randint(0, lkeys-1)]
                    bad_pairs.append((img_feat, self.fbanks[bad_fbanksname]))
            gp = len(good_pairs)
            if not gp:
                continue
            bp = len(bad_pairs)
            assert(gp == bp)
            img = numpy.ndarray((gp+bp, n_img_features), dtype='float32')
            snd = numpy.ndarray((gp+bp, n_fbanks_features), dtype='float32')
            y = numpy.ones((gp+bp,), dtype='int32')
            #print good_pairs
            #print bad_pairs
            #print zip(*good_pairs)[0]
            img[::2] = zip(*good_pairs)[0]
            img[1::2] = zip(*bad_pairs)[0]
            snd[::2] = zip(*good_pairs)[1]
            snd[1::2] = zip(*bad_pairs)[1]
            y[1::2] = 0
            yield (img, snd, y)


