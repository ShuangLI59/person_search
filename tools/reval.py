#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Reval = re-eval. Re-evaluate saved results."""

import _init_paths
from fast_rcnn.test import apply_nms
from fast_rcnn.config import cfg
from datasets.factory import get_imdb
import cPickle
import sys
import os.path as osp
import numpy as np
from argparse import ArgumentParser
from scipy.io import loadmat

if osp.dirname(__file__) not in sys.path:
    sys.path.insert(0, osp.dirname(__file__))

from test_net import evaluate


def load_protoc(root_dir, gallery_size):
    fname = 'TestG{}'.format(gallery_size if gallery_size > 0 else 100)
    protoc = loadmat(osp.join(root_dir, 'annotation/test/train_test',
                              fname + '.mat'))[fname].squeeze()
    return protoc


def main(args):
    imdb = get_imdb(args.imdb_name)
    protoc = load_protoc(imdb._root_dir, args.gallery_size)
    evaluate(protoc, imdb.image_index, args.result_dir, args.gallery_size == 0)


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Re-evaluate the person search result"
    )
    parser.add_argument(
        'result_dir',
        help="Result directory containing *.pkl"
    )
    parser.add_argument(
        '--imdb', dest='imdb_name',
        default='psdb_test',
        help="Dataset to test"
    )
    parser.add_argument(
        '--gallery_size',
        type=int,
        choices=[0, 50, 100, 500, 1000, 2000, 4000],
        default=0,
        help="Gallery size. 0 means using full gallery set"
    )
    args = parser.parse_args()
    main(args)