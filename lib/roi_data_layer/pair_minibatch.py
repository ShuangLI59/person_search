import numpy as np
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

def get_pair_minibatch(roidb, pos_roidb_index, pids):
    if not cfg.TRAIN.HAS_RPN:
        raise NotImplementedError(
            "Currently only supports RPN for proposal generation")
    assert len(pids) == 1
    pid = pids[0]
    coin = np.random.rand()
    if coin < 0.5:
        # positive pair
        probe_index, gallery_index = np.random.choice(
            pos_roidb_index[pid], 2, replace=False)
    else:
        # negative pair
        probe_index = np.random.choice(pos_roidb_index[pid])
        while True:
            gallery_index = np.random.randint(len(roidb))
            if gallery_index not in pos_roidb_index[pid]: break

    probe_blob = _get_roi_blob(roidb[probe_index], pid)
    gallery_blob, gallery_im_scale = _get_image_blob(roidb[gallery_index])
    im_info = np.asarray(
        [[gallery_blob.shape[2], gallery_blob.shape[3], gallery_im_scale]],
        dtype=np.float32)

    # gt_boxes: (x1, y1, x2, y2, is_person, is_target_person)
    gt_inds = np.where(roidb[gallery_index]['gt_classes'] != 0)[0]
    gt_boxes = np.zeros((len(gt_inds), 6), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[gallery_index]['boxes'][gt_inds, :] * gallery_im_scale
    gt_boxes[:, 4] = roidb[gallery_index]['gt_classes'][gt_inds]
    inds = np.where(roidb[gallery_index]['gt_pids'] == pid)[0]
    if len(inds) != 0:
        gt_boxes[inds[0], 5] = 1.

    # rois_p: (0, 0, 0, 64, 160)
    rois_p = np.asarray([0, 0, 0, 64, 160]).astype(np.float32)

    blobs = {
        'data_g': gallery_blob,
        'data_p': probe_blob,
        'gt_boxes': gt_boxes,
        'im_info': im_info,
        'rois_p': rois_p
    }

    # _vis_minibatch(gallery_blob, probe_blob, gt_boxes)

    return blobs

def _get_image_blob(roidb):
    im = cv2.imread(roidb['image'])
    if roidb['flipped']:
        im = im[:, ::-1, :]
    target_size = np.random.choice(cfg.TRAIN.SCALES)
    im, im_scale = prep_im_for_blob(
        im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
    blob = im_list_to_blob([im])
    return blob, im_scale

def _get_roi_blob(roidb, pid):
    im = cv2.imread(roidb['image'])
    if roidb['flipped']:
        im = im[:, ::-1, :]
    im = im.astype(np.float32, copy=False)
    k = np.where(roidb['gt_pids'] == pid)[0][0]
    x1, y1, x2, y2 = roidb['boxes'][k]
    im = im[y1:y2+1, x1:x2+1, :]
    im -= cfg.PIXEL_MEANS
    im = cv2.resize(im, (64, 160), interpolation=cv2.INTER_LINEAR)
    blob = im_list_to_blob([im])
    return blob

def _vis_minibatch(gallery_blob, probe_blob, gt_boxes):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    assert gallery_blob.shape[0] == 1
    assert probe_blob.shape[0] == 1

    gallery_im = gallery_blob[0].transpose(1, 2, 0).copy()
    gallery_im += cfg.PIXEL_MEANS
    gallery_im = gallery_im[:, :, ::-1].astype(np.uint8)

    probe_im = probe_blob[0].transpose(1, 2, 0).copy()
    probe_im += cfg.PIXEL_MEANS
    probe_im = probe_im[:, :, ::-1].astype(np.uint8)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(gallery_im)
    ax2.imshow(probe_im)

    for x1, y1, x2, y2, is_person, is_target_person in gt_boxes:
        assert is_person == 1
        ec = 'g' if is_target_person == 1 else 'r'
        ax1.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    fill=False, edgecolor=ec, linewidth=3))
    plt.show()
