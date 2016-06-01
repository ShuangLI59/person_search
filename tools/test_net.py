#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test_gallery import test_net_on_gallery_set
from fast_rcnn.test_probe import test_net_on_probe_set
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from utils import unpickle
import caffe
import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import average_precision_score

def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union

def load_probe(root_dir, images_dir, gallery_size):
    fname = 'TestG{}'.format(gallery_size if gallery_size > 0 else 100)
    protoc = loadmat(osp.join(root_dir, 'annotation/test/train_test',
                              fname + '.mat'))[fname].squeeze()
    images, rois = [], []
    for item in protoc['Query']:
        im_name = str(item['imname'][0,0][0])
        box = item['idlocate'][0,0][0].astype(np.int32)
        box[2:] += box[:2]
        images.append(osp.join(images_dir, im_name))
        rois.append(box)
    return protoc, images, rois

def evaluate(protoc, images, result_dir, use_full_set=False):
    gallery_det = unpickle(osp.join(result_dir, 'gallery_detections.pkl'))
    gallery_feat = unpickle(osp.join(result_dir, 'gallery_features.pkl'))
    gallery_det = gallery_det[1]
    gallery_feat = gallery_feat[1]
    probe_feat = unpickle(osp.join(result_dir, 'probe_features.pkl'))

    assert len(images) == len(gallery_det)
    assert len(images) == len(gallery_feat)
    name_to_det_feat = {}
    for name, det, feat in zip(images, gallery_det, gallery_feat):
        scores = det[:, 4].ravel()
        inds = np.where(scores >= 0.5)[0]
        if len(inds) > 0:
            det = det[inds]
            feat = feat[inds]
            name_to_det_feat[name] = (det, feat)

    num_probe = len(protoc)
    assert len(probe_feat) == num_probe
    aps = []
    top1_acc = []
    for i in xrange(num_probe):
        y_true, y_score = [], []
        feat_p = probe_feat[i][np.newaxis, :]
        count_gt = 0
        count_tp = 0
        tested = set([str(protoc['Query'][i]['imname'][0,0][0])])
        for item in protoc['Gallery'][i].squeeze():
            gallery_imname = str(item[0][0])
            tested.add(gallery_imname)
            gt = item[1][0].astype(np.int32)
            count_gt += (gt.size > 0)
            if gallery_imname not in name_to_det_feat: continue
            det, feat_g = name_to_det_feat[gallery_imname]
            dis = np.sum((feat_p - feat_g) ** 2, axis=1)
            label = np.zeros(len(dis), dtype=np.int32)
            if gt.size > 0:
                w, h = gt[2], gt[3]
                gt[2:] += gt[:2]
                thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                inds = np.argsort(dis)
                dis = dis[inds]
                # set the label of the first box matched to gt to 1
                for j, roi in enumerate(det[inds, :4]):
                    if _compute_iou(roi, gt) >= thresh:
                        label[j] = 1
                        count_tp += 1
                        break
            y_true.extend(list(label))
            y_score.extend(list(-dis))
        if use_full_set:
            for gallery_imname in images:
                if gallery_imname in tested: continue
                if gallery_imname not in name_to_det_feat: continue
                det, feat_g = name_to_det_feat[gallery_imname]
                dis = np.sum((feat_p - feat_g) ** 2, axis=1)
                label = np.zeros(len(dis), dtype=np.int32)
                y_true.extend(list(label))
                y_score.extend(list(-dis))
        assert count_tp <= count_gt
        recall_rate = count_tp * 1.0 / count_gt
        ap = average_precision_score(y_true, y_score) * recall_rate
        if not np.isnan(ap):
            aps.append(ap)
        else:
            aps.append(0)
        maxind = np.argmax(y_score)
        top1_acc.append(y_true[maxind])

    print 'mAP: {:.2%}'.format(np.mean(aps))
    print 'top-1: {:.2%}'.format(np.mean(top1_acc))


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--gallery_def',
                        help='prototxt file defining the gallery network',
                        default=None, type=str)
    parser.add_argument('--probe_def',
                        help='prototxt file defining the probe network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='psdb_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--feat_blob',
                        help='name of the feature blob to be extracted',
                        default='feat')
    parser.add_argument('--gallery_size',
                        help='gallery size for evaluation',
                        type=int, default=100,
                        choices=[0, 50, 100, 500, 1000, 2000, 4000])

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    while not osp.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    # Detect and store re-id features for all the images in the test images pool
    net = caffe.Net(args.gallery_def, args.caffemodel, caffe.TEST)
    net.name = osp.splitext(osp.basename(args.caffemodel))[0]
    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
    test_net_on_gallery_set(net, imdb, args.feat_blob,
                            max_per_image=args.max_per_image, vis=args.vis)

    root_dir = imdb._root_dir
    images_dir = imdb._data_path
    output_dir = get_output_dir(imdb, net)

    # Extract features for probe people
    net = caffe.Net(args.probe_def, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(osp.basename(args.caffemodel))[0]
    protoc, probe_images, probe_rois = load_probe(
        root_dir, images_dir, args.gallery_size)
    test_net_on_probe_set(net, probe_images, probe_rois, args.feat_blob,
                          output_dir)

    # Evaluate
    evaluate(protoc, imdb.image_index, output_dir, args.gallery_size == 0)