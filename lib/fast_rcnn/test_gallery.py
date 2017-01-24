import numpy as np
import cv2

from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.test_utils import get_image_blob, get_gt_boxes_blob
from fast_rcnn.test_probe import _im_exfeat
from utils.timer import Timer


def _im_detect(net, im, roidb, blob_names=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        roidb (an roidb item): to provide gt_boxes if necessary
        blob_names (list of str): list of feature blob names to be extracted

    Returns:
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        features (dict of ndarray): {blob name: R x D array of features}
    """
    im_blob, im_scales = get_image_blob(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"

    blobs = {
        'data': im_blob,
        'im_info': np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32),
    }

    if 'gt_boxes' in net.blobs:
        # Supply gt_boxes as input. Used to get pid_labels for proposals.
        blobs['gt_boxes'] = get_gt_boxes_blob(
            roidb['boxes'], roidb['gt_classes'], roidb['gt_pids'], im_scales)

    # reshape network inputs
    for k, v in blobs.iteritems():
        net.blobs[k].reshape(*(v.shape))

    # do forward
    forward_kwargs = {k: v.astype(np.float32, copy=False)
                      for k, v in blobs.iteritems()}
    blobs_out = net.forward(**forward_kwargs)

    # unscale rois back to raw image space
    rois = net.blobs['rois'].data.copy()
    boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # the first column of the pid_prob is the non-person box score
        scores = blobs_out['pid_prob'][:, 0]
        scores = scores[:, np.newaxis]
        scores = np.hstack([scores, 1. - scores])

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        # As we no longer scale and shift the bbox_pred weights when snapshot,
        # we need to manually do this during test.
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS and \
                cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            num_classes = box_deltas.shape[1] // 4
            stds = np.tile(cfg.TRAIN.BBOX_NORMALIZE_STDS, num_classes)
            means = np.tile(cfg.TRAIN.BBOX_NORMALIZE_MEANS, num_classes)
            box_deltas = box_deltas * stds + means
        boxes = bbox_transform_inv(boxes, box_deltas)
        boxes = clip_boxes(boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        boxes = np.tile(boxes, (1, scores.shape[1]))

    features = {blob: net.blobs[blob].data.copy() for blob in blob_names} \
               if blob_names is not None else {}

    return boxes, scores, features


def _vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()


def detect_and_exfeat(net, imdb,
                      start=None, end=None,
                      blob_names=None,
                      thresh=0.05, vis=False):
    assert imdb.num_classes == 2, "Only support two-class detection"
    assert cfg.TEST.HAS_RPN, "Only support RPN as proposal"

    start = start or 0
    end = end or imdb.num_images
    num_images = end - start

    # all detections are collected into:
    #    all_boxes[image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    #    all_features[blob][image] = N x D array of features
    all_boxes = [0 for _ in xrange(num_images)]
    all_features = {} if blob_names is None else \
                   {blob: [0 for _ in xrange(num_images)]
                    for blob in blob_names}

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(start + i))
        roidb = imdb.roidb[start + i]

        _t['im_detect'].tic()
        boxes, scores, feat_dic = _im_detect(net, im, roidb, blob_names)
        _t['im_detect'].toc()

        _t['misc'].tic()
        j = 1  # only consider j = 1 (foreground class)
        inds = np.where(scores[:, j] > thresh)[0]
        cls_scores = scores[inds, j]
        cls_boxes = boxes[inds, j*4:(j+1)*4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
        keep = nms(cls_dets, cfg.TEST.NMS)
        all_boxes[i] = cls_dets[keep]
        for blob, feat in feat_dic.iteritems():
            all_features[blob][i] = feat[inds][keep]
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(i + 1, num_images,
            _t['im_detect'].average_time, _t['misc'].average_time)

        if vis:
            _vis_detections(im, imdb.classes[j], all_boxes[i])

    return all_boxes, all_features


def usegt_and_exfeat(net, imdb,
                     start=None, end=None, blob_names=None):
    start = start or 0
    end = end or imdb.num_images
    num_images = end - start

    # all detections are collected into:
    #    all_boxes[image] = N x 5 array of detections (gt) in
    #    (x1, y1, x2, y2, score)
    #    all_features[blob][image] = N x D array of features
    all_boxes = [0 for _ in xrange(num_images)]
    all_features = {} if blob_names is None else \
                   {blob: [0 for _ in xrange(num_images)]
                    for blob in blob_names}

    # timers
    _t = {'gt_exfeat' : Timer(), 'misc' : Timer()}

    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(start + i))
        gt = imdb.roidb[start + i]['boxes']

        _t['gt_exfeat'].tic()
        feat_dic = _im_exfeat(net, im, gt, blob_names)
        _t['gt_exfeat'].toc()

        all_boxes[i] = np.hstack((gt, np.ones((gt.shape[0], 1)))) \
            .astype(np.float32)
        for blob, feat in feat_dic.iteritems():
            all_features[blob][i] = feat

        print 'gt_exfeat: {:d}/{:d} {:.3f}s'.format(i + 1, num_images,
            _t['gt_exfeat'].average_time)

    return all_boxes, all_features
