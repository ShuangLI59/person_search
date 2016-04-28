#!/usr/bin/env bash

DATA=data/psdb
CAFFE=caffe-fast-rcnn

cd $(dirname ${BASH_SOURCE[0]})/../

# Parse arguments.
if [[ $# -ne 1 ]]; then
  echo "Usage: $(basename $0) dataset"
  echo "    dataset    Path to the downloaded dataset.zip"
  exit
fi

# Unzip the raw data
echo "Unzip the raw data ..."
mkdir -p $DATA
unzip -qq $1 -d $DATA/
rm -rf $DATA/__MACOSX

# Create the pretrain database
echo "Create pretrain database ..."
mkdir -p $DATA/pretrain_db
python2 tools/make_pretrain_dataset.py
for subset in train val; do
  $CAFFE/build/tools/convert_imageset \
    -encoded -resize_height 256 -resize_width 256 \
    $DATA/pretrain_db/ $DATA/pretrain_db/${subset}.txt \
    $DATA/pretrain_db/${subset}_lmdb
done
