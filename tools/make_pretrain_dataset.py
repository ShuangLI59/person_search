import numpy as np
import os
import os.path as osp
from scipy.io import loadmat
from scipy.misc import imread, imsave
from argparse import ArgumentParser

def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter * 1.0 / union

def write_meta(meta, filename):
    content = ['{} {}'.format(a, b) for a, b in meta]
    with open(filename, 'w') as f:
        f.write('\n'.join(content))

def main(args):
    if not osp.isdir(osp.join(args.output_dir, 'images')):
        os.makedirs(osp.join(args.output_dir, 'images'))

    meta = []

    # positive samples (all the training people)
    train = loadmat(osp.join(args.root_dir,
                             'annotation/test/train_test/Train.mat'))
    train = train['Train'].squeeze()
    for pid, item in enumerate(train):
        scenes = item[0, 0][2].squeeze()
        for imid, (im_name, box, __) in enumerate(scenes):
            im_name = str(im_name[0])
            box = box.squeeze().astype(np.int32)
            im = imread(osp.join(args.root_dir, 'Image/SSM', im_name))
            x, y, w, h = box
            cropped = im[y:y+h+1, x:x+w+1, :]
            filename = osp.join('images', '{:05d}_{:02d}.jpg'.format(pid, imid))
            imsave(osp.join(args.output_dir, filename), cropped)
            meta.append((filename, pid))

    # negative samples (background rois with little overlap with people)
    all_imgs = loadmat(osp.join(args.root_dir, 'annotation', 'Images.mat'))
    all_imgs = all_imgs['Img'].squeeze()
    name_to_boxes = {}
    all_imnames = []
    for im_name, __, boxes in all_imgs:
        im_name = str(im_name[0])
        all_imnames.append(im_name)
        boxes = np.asarray([b[0] for b in boxes[0]])
        boxes = boxes.reshape(boxes.shape[0], 4)
        valid_index = np.where((boxes[:, 2] > 0) & (boxes[:, 3] > 0))[0]
        assert valid_index.size > 0, \
            'Warning: {} has no valid boxes.'.format(im_name)
        name_to_boxes[im_name] = boxes.astype(np.int32)

    num_neg = len(meta)
    neg_id = len(train)
    for item_id in xrange(num_neg):
        imname = np.random.choice(all_imnames)
        im = imread(osp.join(args.root_dir, 'Image/SSM', imname))
        height, width = im.shape[:2]
        gt_boxes = name_to_boxes[imname]
        while True:
            h = int(160 * (np.random.rand() * 1.5 + 0.5))
            w = int(64 * (np.random.rand() * 1.5 + 0.5))
            x = np.random.randint(0, width - w)
            y = np.random.randint(0, height - h)
            is_bg = True
            for gt in gt_boxes:
                if _compute_iou([x, y, w, h], gt) >= 0.3:
                    is_bg = False
                    break
            if is_bg: break
        cropped = im[y:y+h+1, x:x+w+1, :]
        filename = osp.join('images', '{:05d}_{:02d}.jpg'.format(neg_id, item_id))
        imsave(osp.join(args.output_dir, filename), cropped)
        meta.append((filename, neg_id))

    # train / val
    np.random.shuffle(meta)
    num_train = int(len(meta) * args.train_ratio)
    write_meta(meta[:num_train], osp.join(args.output_dir, 'train.txt'))
    write_meta(meta[num_train:], osp.join(args.output_dir, 'val.txt'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root_dir', default='data/psdb/dataset')
    parser.add_argument('--output_dir', default='data/psdb/pretrain_db')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    args = parser.parse_args()
    main(args)