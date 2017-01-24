import numpy as np
import cv2

from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.test_utils import get_image_blob, get_rois_blob
from utils.timer import Timer


def _im_exfeat(net, im, roi, blob_names):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        roi (ndarray): 1 x 4 array of the target roi
        blob_names (list of str): list of feature blob names to be extracted

    Returns:
        features (dict of ndarray): {blob name: R x D array of features}
    """
    im_blob, im_scales = get_image_blob(im)

    blobs = {
        'data': im_blob,
        'im_info': np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32),
        'rois': get_rois_blob(roi, im_scales),
    }

    # reshape network inputs
    for k, v in blobs.iteritems():
        net.blobs[k].reshape(*(v.shape))

    # do forward
    forward_kwargs = {k: v.astype(np.float32, copy=False)
                      for k, v in blobs.iteritems()}
    net.forward(**forward_kwargs)

    features = {blob: net.blobs[blob].data.copy() for blob in blob_names} \
               if blob_names is not None else {}

    return features


def exfeat(net, probes,
           start=None, end=None, blob_names=None):
    start = start or 0
    end = end or len(probes)
    num_images = end - start

    # all_features[blob] = num_images x D array of features
    all_features = {} if blob_names is None else \
                   {blob: [0 for _ in xrange(num_images)]
                    for blob in blob_names}

    # timers
    _t = {'im_exfeat' : Timer(), 'misc' : Timer()}

    for i in xrange(num_images):
        im_name, roi = probes[start + i]
        im = cv2.imread(im_name)
        roi = roi.reshape(1, 4)

        _t['im_exfeat'].tic()
        feat_dic = _im_exfeat(net, im, roi, blob_names)
        _t['im_exfeat'].toc()

        for blob, feat in feat_dic.iteritems():
            assert feat.shape[0] == 1
            all_features[blob][i] = feat[0]

        print 'im_exfeat: {:d}/{:d} {:.3f}s'.format(i + 1, num_images,
            _t['im_exfeat'].average_time)

    return all_features
