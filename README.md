### Datasets

[Pascal1K](http://nlp.cs.illinois.edu/HockenmaierGroup/pascal-sentences/index.html)  
[Flickr8K](http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html) ([ImagesData](https://illinois.edu/fb/sec/1713398))

### Code

TODO: one line on the code to transform LUCID into our dataset.

TODO: one line on the code to extract R-CNN features from images.

To train the multi-modal embedding net, do:  
```
THEANO_FLAGS="device=gpu1" python run_exp_AB.py --dataset-path=/fhgfs/bootphon/scratch/gsynnaeve/learning_semantics2014/pascal_full/ --prefix-output-fname="maxnorm" --debug-print=1 --debug-time
```

### Results

TODO: scores && plots

### "Say"-based corpus

(On Mac OS X only)
Use `bash say_words.sh` with a words.txt file in the same directory. Then you
can `sox` the produced `*.aif` to `*.wav` files (you may need to `for` loop),
then transform them to filterbanks with `python mfsc.py *.wav` (`for` too) and
finally put them in one big `*.npz` formatted dictionary with
`python npz_fbanks.py FOLDER`.  
Finally use `python simple_dnn.py` to train a DNN if wanted (look into it).

