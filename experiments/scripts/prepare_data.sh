#!/usr/bin/env bash

DATA=data/psdb

cd $(dirname ${BASH_SOURCE[0]})/../../

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

# Create folder for pretrained imagenet models
mkdir -p data/imagenet_models
