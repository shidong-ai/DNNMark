#! /bin/sh

if [ $# -ne 1 ]
then
  echo "[Error] The setup script requires one additional parameter specifying whether CUDA or HCC is used"
  echo "Options: [CUDA, HIP]"
  exit
fi

OPTION=$1

BUILD_DIR=build
if [ ! -d ${BUILD_DIR} ]; then
  mkdir ${BUILD_DIR}
fi
cd ${BUILD_DIR}

if [ ${OPTION} = "CUDA" ]
then
  CUDNN_PATH=${HOME}/cudnn
  cmake -DCUDNN_ROOT=${CUDNN_PATH} ..
elif [ ${OPTION} = "HIP" ]
then
  MIOPEN_PATH=/opt/rocm/miopen
  CXX=/opt/rocm/hcc/bin/hcc cmake -DMIOPEN_ROOT=${MIOPEN_PATH} ..
fi
