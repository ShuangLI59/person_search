# Person Search Project

This repository hosts the code for our paper [End-to-End Deep Learning for Person Search](https://arxiv.org/abs/1604.01850). The code is modified from the py-faster-rcnn written by Ross Girshick.

## Requirements

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```

  You can download my [Makefile.config](http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/Makefile.config) for reference.
2. Python packages you might not have: `cython`, `python-opencv`, `easydict (>=1.6)`
3. MATLAB is required for processing the raw data.

## Installation

1. Build the Cython modules
    ```Shell
    cd lib
    make
    ```

2. Build Caffe and pycaffe
    ```Shell
    cd caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```

## Experiments

1. Request the dataset from sli [at] ee.cuhk.edu.hk (academic only)

2. Convert the raw dataset into formatted database
    ```Shell
    scripts/make_db.sh /path/to/the/downloaded/dataset.zip
    ```

3. Pretraining
    ```Shell
    data/scripts/fetch_imagenet_models.sh
    experiments/scripts/pretrain.sh
    ```
    Or you may directly download a **pretrained model** from [here](https://drive.google.com/file/d/0B67_d0rLRTQYQTJJSTNOX3pEVlE/view?usp=sharing) to `output/psdb_pretrain/`.

4. Training and evaluation
    ```Shell
    experiments/scripts/train.sh 0
    ```
    Or you may directly download a **trained model** from [here](https://drive.google.com/file/d/0B67_d0rLRTQYeDlXMlc2OGUzeG8/view?usp=sharing) to `output/psdb_train/`, and test the performance with
    ```Shell
    python2 tools/test_net.py --gpu 0 \
      --gallery_def models/psdb/VGG16/test_gallery.prototxt \
      --probe_def models/psdb/VGG16/test_probe.prototxt \
      --net output/psdb_train/VGG16_iter_100000.caffemodel \
      --cfg experiments/cfgs/train.yml \
      --imdb psdb_test
    ```

## Citation

    @article{xiao2016end,
      title={End-to-End Deep Learning for Person Search},
      author={Xiao, Tong and Li, Shuang and Wang, Bochao and Lin, Liang and Wang, Xiaogang},
      journal={arXiv:1604.01850},
      year={2016}
    }