# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import cPickle

def pickle(data, file_path):
    with open(file_path, 'wb') as f:
        cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)

def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = cPickle.load(f)
    return data