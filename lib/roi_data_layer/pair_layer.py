# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

PairRoIDataLayer implements a Caffe Python layer.
"""

import caffe
from fast_rcnn.config import cfg
from roi_data_layer.pair_minibatch import get_pair_minibatch
import numpy as np
import yaml
from collections import defaultdict

import pdb

class PairRoIDataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def _shuffle(self):
        self._shuffled_pids = self._pos_roidb_index.keys()
        np.random.shuffle(self._shuffled_pids)
        self._cur = 0

    def _get_next_minibatch(self):
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._shuffled_pids):
            self._shuffle()
        pids = self._shuffled_pids[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return get_pair_minibatch(self._roidb, self._pos_roidb_index, pids)

    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._pos_roidb_index = defaultdict(list)
        for index, item in enumerate(roidb):
            pids = item['gt_pids']
            for pid in pids:
                if pid > 0:
                    self._pos_roidb_index[pid].append(index)
        self._shuffle()

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}

        # gallery image
        idx = 0
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,
            max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)
        self._name_to_top_map['data_g'] = idx
        idx += 1

        # probe image
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3, 160, 64)
        self._name_to_top_map['data_p'] = idx
        idx += 1

        if cfg.TRAIN.HAS_RPN:
            top[idx].reshape(1, 3)
            self._name_to_top_map['im_info'] = idx
            idx += 1

            top[idx].reshape(1, 5)
            self._name_to_top_map['gt_boxes'] = idx
            idx += 1

            top[idx].reshape(1, 5)
            self._name_to_top_map['rois_p'] = idx
            idx += 1
        else: # not using RPN
            raise NotImplementedError(
                "Currently only supports RPN for proposal generation")

        print 'PairRoiDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
