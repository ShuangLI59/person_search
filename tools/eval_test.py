import _init_paths

import itertools
import argparse
import pprint
import time, os, sys
import os.path as osp

import numpy as np
import caffe
from mpi4py import MPI

from fast_rcnn.test_gallery import detect_and_exfeat
from fast_rcnn.test_probe import exfeat
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from utils import pickle, unpickle


# mpi setup
mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()
if mpi_rank > 0:
    # disable print on other mpi processes
    sys.stdout = open(os.devnull, 'w')


# functions for mpi
def _dispatch_job(num_jobs, num_workers=mpi_size, worker_id=mpi_rank):
    jobs_per_worker = num_jobs // num_workers
    start = worker_id * jobs_per_worker
    end = num_jobs if worker_id == num_workers-1 else start + jobs_per_worker
    return start, end

def _collect_result(data):
    if isinstance(data, list):
        data = mpi_comm.gather(data, root=0)
        if mpi_rank == 0:
            data = list(itertools.chain.from_iterable(data))
    elif isinstance(data, dict):
        for k, v in data.iteritems():
            data[k] = _collect_result(v)
    else:
        raise ValueError("Cannot collect result of " + type(data))
    return data


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

    # setup caffe
    caffe.mpi_init()
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)

    # 1. Detect and extract features from all the gallery images in the imdb
    start, end = _dispatch_job(len(imdb.image_index))
    net = caffe.Net(args.gallery_def, args.caffemodel, caffe.TEST)
    net.name = osp.splitext(osp.basename(args.caffemodel))[0]
    gboxes, gfeatures = detect_and_exfeat(net, imdb,
        start=start, end=end, blob_names=blob_names, vis=args.vis)
    gboxes = _collect_result(gboxes)
    gfeatures = _collect_result(gfeatures)
    del net # to release the cudnn conv static workspace

    # 2. Only extract features from given probe rois
    start, end = _dispatch_job(len(imdb.probes))
    net = caffe.Net(args.probe_def, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(osp.basename(args.caffemodel))[0]
    pfeatures = exfeat(net, imdb.probes,
        start=start, end=end, blob_names=blob_names)
    pfeatures = _collect_result(pfeatures)
    del net

    # Evaluate
    if mpi_rank == 0:
        output_dir = get_output_dir(imdb, net)
        pickle(gboxes, osp.join(output_dir, 'gallery_detections.pkl'))
        pickle(gfeatures, osp.join(output_dir, 'gallery_features.pkl'))
        pickle(pfeatures, osp.join(output_dir, 'probe_features.pkl'))
        imdb.evaluate_search(gboxes, gfeatures['feat'], pfeatures['feat'],
                             args.det_thresh, args.gallery_size)

    caffe.mpi_finalize()


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
    parser.add_argument('--wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--vis',
                        help='visualize detections',
                        action='store_true')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    print('Called with args:')
    print(args)

    main(args)
