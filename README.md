# Person Search Project

This repository hosts the code for our paper [End-to-End Deep Learning for Person Search](https://arxiv.org/abs/1604.01850). The code is modified from the py-faster-rcnn written by Ross Girshick.


## Installation

1. Clone this repo **recursively** and switch to the `oim` branch

  ```Shell
  git clone --recursive https://github.com/ShuangLI59/person_search.git -b oim
  ```

2. Build Caffe with python layers and interface

  We modified caffe based on [Yuanjun's fork](https://github.com/yjxiong/caffe/tree/mem), which provides multi-gpu parallelism and memory optimization.

  Apart from the official installation [prerequisites](http://caffe.berkeleyvision.org/installation.html), we have several other dependencies:

  - [cudnn-v5.1](https://developer.nvidia.com/cudnn)
  - 1.7.4 < [openmpi](https://www.open-mpi.org/) < 2.0.0
  - boost >= 1.55 (A tip for Ubuntu 14.04: `sudo apt-get autoremove libboost1.54*` then `sudo apt-get install libboost1.55-all-dev`)

  Then compile and install the caffe with
  ```Shell
  cd caffe
  mkdir build && cd build
  cmake .. -DUSE_MPI=ON -DCUDNN_INCLUDE=/path/to/cudnn/include -DCUDNN_LIBRARY=/path/to/cudnn/lib64/libcudnn.so
  make -j8 && make install
  cd ../..
  ```

  Please refer to [this page](https://github.com/yjxiong/caffe/tree/mem#usage) for detailed installation instructions and troubleshooting.

3. Build the Cython modules

  Install some Python packages you might not have: `Cython`, `python-opencv`, `easydict (>=1.6)`, `PyYAML`, `protobuf`, `mpi4py`. Then
  ```Shell
  cd lib && make && cd ..
  ```

## Experiments

1. Request the dataset from sli [at] ee.cuhk.edu.hk or xiaotong [at] ee.cuhk.edu.hk (academic only). Then

  ```Shell
  experiments/scripts/prepare_data.sh /path/to/the/downloaded/dataset.zip
  ```

2. Download an ImageNet [pretrained **ResNet-50 model**](https://drive.google.com/open?id=0B67_d0rLRTQYUHFjU0pPSExhS1U) to `data/imagenet_models`.

3. Training with GPU=0

  ```Shell
  experiments/scripts/train.sh 0 --set EXP_DIR resnet50
  ```

  It will finish in around 18 hours, or you may directly download a [**trained model**](https://drive.google.com/open?id=0B67_d0rLRTQYbVFENlVjdXRSWVE) to `output/psdb_train/resnet50/`

4. Evaluation

    By default we use 8 GPUs for faster evaluation. Please adjust the `experiments/scripts/eval_test.sh` with your hardware settings. For example, to use only one GPU, remove the `mpirun -n 8` in L14 and change L16 to `--gpu 0`.

    ```Shell
    experiments/scripts/eval_test.sh resnet50 50000 resnet50
    ```

    The result should be around

    ```Shell
    search ranking:
      mAP = 75.47%
      top- 1 = 78.62%
      top- 5 = 90.24%
      top-10 = 92.38%
    ```

## Citation

    @article{xiaoli2016end,
      title={End-to-End Deep Learning for Person Search},
      author={Xiao, Tong and Li, Shuang and Wang, Bochao and Lin, Liang and Wang, Xiaogang},
      journal={arXiv:1604.01850},
      year={2016}
    }
