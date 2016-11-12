#! /bin/sh

BUILD_DIR=build
CUDNN_PATH=/home/shidong/cudnn
if [ ! -d ${BUILD_DIR} ]; then
  mkdir ${BUILD_DIR}
fi
cd ${BUILD_DIR}
cmake -DCUDNN_ROOT=${CUDNN_PATH} ..
