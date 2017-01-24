import _init_paths

import argparse
import pprint
import time, os, sys
import os.path as osp

import numpy as np
import caffe
from mpi4py import MPI

from fast_rcnn.test_gallery import detect_and_exfeat, usegt_and_exfeat
from fast_rcnn.test_probe import exfeat
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from utils import pickle, unpickle
from eval_utils import mpi_dispatch, mpi_collect


# mpi setup
mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()
if mpi_rank > 0:
    # disable print on other mpi processes
    sys.stdout = open(os.devnull, 'w')


def main(args):
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # parse gpus
    gpus = map(int, args.gpus.split(','))
    assert len(gpus) >= mpi_size, "Number of GPUs must be >= MPI size"
    cfg.GPU_ID = gpus[mpi_rank]

    # parse feature blob names
    blob_names = args.blob_names.split(',')

    print('Using config:')
    pprint.pprint(cfg)

    while not osp.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    # load imdb
    imdb = get_imdb(args.imdb_name)
    root_dir = imdb._root_dir
    images_dir = imdb._data_path
    output_dir = get_output_dir(imdb.name,
                                osp.splitext(osp.basename(args.caffemodel))[0])

    if args.eval_only:
        def _load(fname):
            fpath = osp.join(output_dir, fname)
            assert osp.isfile(fpath), "Must have extracted detections and " \
                                      "features first before evaluation"
            return unpickle(fpath)
        if mpi_rank == 0:
            gboxes = _load('gallery_detections.pkl')
            gfeatures = _load('gallery_features.pkl')
            pfeatures = _load('probe_features.pkl')
    else:
        # setup caffe
        caffe.mpi_init()
        caffe.set_mode_gpu()
        caffe.set_device(cfg.GPU_ID)

        # 1. Detect and extract features from all the gallery images in the imdb
        start, end = mpi_dispatch(len(imdb.image_index), mpi_size, mpi_rank)
        if args.use_gt:
            net = caffe.Net(args.probe_def, args.caffemodel, caffe.TEST)
            gboxes, gfeatures = usegt_and_exfeat(net, imdb,
                start=start, end=end, blob_names=blob_names)
        else:
            net = caffe.Net(args.gallery_def, args.caffemodel, caffe.TEST)
            gboxes, gfeatures = detect_and_exfeat(net, imdb,
                start=start, end=end, blob_names=blob_names)
        gboxes = mpi_collect(mpi_comm, mpi_rank, gboxes)
        gfeatures = mpi_collect(mpi_comm, mpi_rank, gfeatures)
        del net # to release the cudnn conv static workspace

        # 2. Only extract features from given probe rois
        start, end = mpi_dispatch(len(imdb.probes), mpi_size, mpi_rank)
        net = caffe.Net(args.probe_def, args.caffemodel, caffe.TEST)
        pfeatures = exfeat(net, imdb.probes,
            start=start, end=end, blob_names=blob_names)
        pfeatures = mpi_collect(mpi_comm, mpi_rank, pfeatures)
        del net

        # Save
        if mpi_rank == 0:
            pickle(gboxes, osp.join(output_dir, 'gallery_detections.pkl'))
            pickle(gfeatures, osp.join(output_dir, 'gallery_features.pkl'))
            pickle(pfeatures, osp.join(output_dir, 'probe_features.pkl'))

    # Evaluate
    if mpi_rank == 0:
        imdb.evaluate_detections(gboxes, det_thresh=args.det_thresh)
        imdb.evaluate_detections(gboxes, det_thresh=args.det_thresh,
                                 labeled_only=True)
        imdb.evaluate_search(gboxes, gfeatures['feat'], pfeatures['feat'],
             det_thresh=args.det_thresh,
             gallery_size=args.gallery_size,
             dump_json=osp.join(output_dir, 'results.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evalute on test set, including search ranking accuracy')
    parser.add_argument('--gpus',
                        help='comma separated GPU device ids',
                        default='0')
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='psdb_test')
    parser.add_argument('--gallery_def',
                        help='prototxt file defining the gallery network')
    parser.add_argument('--probe_def',
                        help='prototxt file defining the probe network')
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test')
    parser.add_argument('--blob_names',
                        help='comma separated names of the feature blobs ' \
                             'to be extracted',
                        default='feat')
    parser.add_argument('--det_thresh',
                        help="detection score threshold to be evaluated",
                        type=float, default=0.5)
    parser.add_argument('--gallery_size',
                        help='gallery size for evaluation, -1 for full set',
                        type=int, default=100,
                        choices=[-1, 50, 100, 500, 1000, 2000, 4000])
    parser.add_argument('--eval_only',
                        help='skip the feature extraction and only do eval',
                        action='store_true')
    parser.add_argument('--use_gt',
                        help='use ground truth boxes as proposals',
                        action='store_true')
    parser.add_argument('--wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    print('Called with args:')
    print(args)

    main(args)
