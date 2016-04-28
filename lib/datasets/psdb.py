import os.path as osp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import loadmat

import datasets
from datasets.imdb import imdb
from fast_rcnn.config import cfg
from utils import cython_bbox, pickle, unpickle


def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union


class psdb(imdb):
    def __init__(self, image_set, root_dir=None):
        super(psdb, self).__init__('psdb_' + image_set)
        self._image_set = image_set
        self._root_dir = self._get_default_path() if root_dir is None \
                         else root_dir
        self._data_path = osp.join(self._root_dir, 'Image', 'SSM')
        self._classes = ('__background__', 'person')
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb
        assert osp.isdir(self._root_dir), \
                "PSDB does not exist: {}".format(self._root_dir)
        assert osp.isdir(self._data_path), \
                "Path does not exist: {}".format(self._data_path)

    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        image_path = osp.join(self._data_path, index)
        assert osp.isfile(image_path), \
                "Path does not exist: {}".format(image_path)
        return image_path

    def gt_roidb(self):
        cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if osp.isfile(cache_file):
            roidb = unpickle(cache_file)
            print "{} gt roidb loaded from {}".format(self.name, cache_file)
            return roidb

        # Load all images and build a dict from image to boxes
        all_imgs = loadmat(osp.join(self._root_dir, 'annotation', 'Images.mat'))
        all_imgs = all_imgs['Img'].squeeze()
        name_to_boxes = {}
        name_to_pids = {}
        for im_name, __, boxes in all_imgs:
            im_name = str(im_name[0])
            boxes = np.asarray([b[0] for b in boxes[0]])
            boxes = boxes.reshape(boxes.shape[0], 4)
            valid_index = np.where((boxes[:, 2] > 0) & (boxes[:, 3] > 0))[0]
            assert valid_index.size > 0, \
                'Warning: {} has no valid boxes.'.format(im_name)
            boxes = boxes[valid_index]
            name_to_boxes[im_name] = boxes.astype(np.int32)
            name_to_pids[im_name] = -1 * np.ones(boxes.shape[0], dtype=np.int32)

        def _set_box_pid(boxes, box, pids, pid):
            for i in xrange(boxes.shape[0]):
                if np.all(boxes[i] == box):
                    pids[i] = pid
                    return
            print 'Warning: person {} box {} cannot find in Images'.format(pid, box)

        # Load all the train / test persons and number their pids from 1 to N-1
        # Background people have pid == 0
        if self._image_set == 'train':
            train = loadmat(osp.join(self._root_dir,
                                     'annotation/test/train_test/Train.mat'))
            train = train['Train'].squeeze()
            for index, item in enumerate(train):
                scenes = item[0, 0][2].squeeze()
                for im_name, box, __ in scenes:
                    im_name = str(im_name[0])
                    box = box.squeeze().astype(np.int32)
                    _set_box_pid(name_to_boxes[im_name], box,
                                 name_to_pids[im_name], index)
        else:
            test = loadmat(osp.join(self._root_dir,
                                    'annotation/test/train_test/TestG50.mat'))
            test = test['TestG50'].squeeze()
            for index, item in enumerate(test):
                # query
                im_name = str(item['Query'][0,0][0][0])
                box = item['Query'][0,0][1].squeeze().astype(np.int32)
                _set_box_pid(name_to_boxes[im_name], box,
                             name_to_pids[im_name], index)
                # gallery
                gallery = item['Gallery'].squeeze()
                for im_name, box, __ in gallery:
                    im_name = str(im_name[0])
                    if box.size == 0: break
                    box = box.squeeze().astype(np.int32)
                    _set_box_pid(name_to_boxes[im_name], box,
                                 name_to_pids[im_name], index)

        # Construct the gt_roidb
        gt_roidb = []
        for index in self.image_index:
            boxes = name_to_boxes[index]
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            pids = name_to_pids[index]
            num_objs = len(boxes)
            gt_classes = np.ones((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
            overlaps[:, 1] = 1.0
            overlaps = csr_matrix(overlaps)
            gt_roidb.append({
                'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'gt_pids': pids,
                'flipped': False})

        pickle(gt_roidb, cache_file)
        print "wrote gt roidb to {}".format(cache_file)

        return gt_roidb

    def evaluate_detections(self, all_boxes, output_dir):
        # all_boxes[cls][image] = N x 5 (x1, y1, x2, y2, score)
        gt_roidb = self.gt_roidb()
        assert len(all_boxes) == 2
        assert len(all_boxes[1]) == len(gt_roidb)
        y_true = []
        y_score = []
        count_gt = 0
        count_tp = 0
        for gt, det in zip(gt_roidb, all_boxes[1]):
            gt_boxes = gt['boxes']
            det = np.asarray(det)
            num_gt = gt_boxes.shape[0]
            num_det = det.shape[0]
            if num_det == 0:
                count_gt += num_gt
                continue
            ious = np.zeros((num_gt, num_det), dtype=np.float32)
            for i in xrange(num_gt):
                for j in xrange(num_det):
                    ious[i, j] = _compute_iou(gt_boxes[i], det[j, :4])
            tfmat = (ious >= 0.5)
            # for each det, keep only the largest iou of all the gt
            for j in xrange(num_det):
                largest_ind = np.argmax(ious[:, j])
                for i in xrange(num_gt):
                    if i != largest_ind:
                        tfmat[i, j] = False
            # for each gt, keep only the largest iou of all the det
            for i in xrange(num_gt):
                largest_ind = np.argmax(ious[i, :])
                for j in xrange(num_det):
                    if j != largest_ind:
                        tfmat[i, j] = False
            for j in xrange(num_det):
                y_score.append(det[j, -1])
                if tfmat[:, j].any():
                    y_true.append(True)
                else:
                    y_true.append(False)
            count_tp += tfmat.sum()
            count_gt += num_gt

        from sklearn.metrics import average_precision_score, precision_recall_curve
        det_rate = count_tp * 1.0 / count_gt
        ap = average_precision_score(y_true, y_score) * det_rate
        precision, recall, __ = precision_recall_curve(y_true, y_score)
        recall *= det_rate

        import matplotlib.pyplot as plt
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title('Det Rate: {:.2%}  AP: {:.2%}'.format(det_rate, ap))
        plt.show()

    def _get_default_path(self):
        return osp.join(cfg.DATA_DIR, 'psdb', 'dataset')

    def _load_image_set_index(self):
        """
        Load the indexes for the specific subset (train / test).
        For PSDB, the index is just the image file name.
        """
        # test pool
        test = loadmat(osp.join(self._root_dir, 'annotation', 'pool.mat'))
        test = test['pool'].squeeze()
        test = [str(a[0]) for a in test]
        if self._image_set == 'test': return test
        # all images
        all_imgs = loadmat(osp.join(self._root_dir, 'annotation', 'Images.mat'))
        all_imgs = all_imgs['Img'].squeeze()
        all_imgs = [str(a[0][0]) for a in all_imgs]
        # training
        return list(set(all_imgs) - set(test))


if __name__ == '__main__':
    from datasets.psdb import psdb
    d = psdb('train')
    res = d.roidb
    from IPython import embed; embed()
