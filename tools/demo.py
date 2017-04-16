import _init_paths

import argparse
import time
import os
import sys
import os.path as osp
from glob import glob

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import caffe
from mpi4py import MPI

from fast_rcnn.test_probe import demo_exfeat
from fast_rcnn.test_gallery import demo_detect
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list



def main(args):
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # Setup caffe
    if args.gpu >= 0:
        caffe.mpi_init()
        caffe.set_mode_gpu()
        caffe.set_device(cfg.GPU_ID)
    else:
        caffe.mpi_init()
        caffe.set_mode_cpu()

    # Get query image and roi
    query_img = 'demo/query.jpg'
    query_roi = [0, 0, 466, 943]  # [x1, y1, x2, y2]

    # Extract feature of the query person
    net = caffe.Net(args.probe_def, args.caffemodel, caffe.TEST)
    query_feat = demo_exfeat(net, query_img, query_roi)
    del net  # Necessary to release cuDNN conv static workspace

    # Get gallery images
    gallery_imgs = sorted(glob('demo/gallery*.jpg'))

    # Detect and extract feature of persons in each gallery image
    net = caffe.Net(args.gallery_def, args.caffemodel, caffe.TEST)

    # Necessary to warm-up the net, otherwise the first image results are wrong
    # Don't know why. Possibly a bug in caffe's memory optimization.
    # Nevertheless, the results are correct after this warm-up.
    demo_detect(net, query_img)

    for gallery_img in gallery_imgs:
        print gallery_img, '...'
        boxes, features = demo_detect(net, gallery_img,
                                      threshold=args.det_thresh)
        if boxes is None:
            print gallery_img, 'no detections'
            continue
        # Compute pairwise cosine similarities,
        #   equals to inner-products, as features are already L2-normed
        similarities = features.dot(query_feat)

        # Visualize the results
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.imshow(plt.imread(gallery_img))
        plt.axis('off')
        for box, sim in zip(boxes, similarities):
            x1, y1, x2, y2, _ = box
            ax.add_patch(
                plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                              fill=False, edgecolor='#4CAF50', linewidth=3.5))
            ax.add_patch(
                plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                              fill=False, edgecolor='white', linewidth=1))
            ax.text(x1 + 5, y1 - 18, '{:.2f}'.format(sim),
                    bbox=dict(facecolor='#4CAF50', linewidth=0),
                    fontsize=20, color='white')
        plt.tight_layout()
        fig.savefig(gallery_img.replace('gallery', 'result'))
        plt.show()
        plt.close(fig)
    del net


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Person Search Demo')
    parser.add_argument('--gpu',
                        help='GPU id to be used, -1 for CPU. Default: 0',
                        type=int, default=0)
    parser.add_argument('--gallery_def',
                        help='prototxt file defining the gallery network',
                        default='models/psdb/resnet50/eval_gallery.prototxt')
    parser.add_argument('--probe_def',
                        help='prototxt file defining the probe network',
                        default='models/psdb/resnet50/eval_probe.prototxt')
    parser.add_argument('--net', dest='caffemodel',
                        help='path to trained caffemodel',
                        default='output/psdb_train/resnet50/resnet50_iter_50000.caffemodel')
    parser.add_argument('--det_thresh',
                        help="detection score threshold to be evaluated",
                        type=float, default=0.75)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='experiments/cfgs/resnet50.yml')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    main(args)
