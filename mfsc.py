"""
python mfsc.py *.wav
will transform all the $name.wav files passed as arguments into
$name_fbanks.npy files of log-compressed Mel filterbanks.
"""
from spectral import Spectral
import sys
import wave, struct
import numpy as np

def readwav(fname):
    fid = wave.open(fname, 'r')
    _, _, fs, nframes, _, _ = fid.getparams()
    sig = np.array(struct.unpack_from("%dh" % nframes,
    fid.readframes(nframes)))
    fid.close()
    return sig, fs

FBANKS_WINDOW = 0.025 # 25ms
FBANKS_RATE = 100 # 10ms
N_FBANKS = 40

for wavfname in sys.argv[1:]:
    sound, srate = readwav(wavfname)
    fbanks = Spectral(nfilt=N_FBANKS,      # nb of filters in mel bank
            alpha=0.97,               # pre-emphasis
            do_dct=False,             # we do not want MFCCs
            compression='log',
            fs=srate,                 # sampling rate
            #lowerf=50,                # lower frequency
            frate=FBANKS_RATE,        # frame rate
            wlen=FBANKS_WINDOW,       # window length
            nfft=1024,                # length of dft
            do_deltas=False,          # speed
            do_deltasdeltas=False     # acceleration
            )
    fbank = fbanks.transform(sound)
    fbanksfname = wavfname[:-4]+'_fbanks.npy'
    with open(fbanksfname, 'w') as o_f:
        np.save(o_f, fbank)

