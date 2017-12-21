#! /bin/bash

WORK_DIR="$(pwd)"
BENCHMARK_DIR=${WORK_DIR}/../build/benchmarks/test_alexnet/
CONFIG_DIR=${WORK_DIR}/../config_example/

EXE="$(find ${BENCHMARK_DIR} -executable -type f)"

KF_CONFIG=${CONFIG_DIR}alexnet_no_relu_128.dnnmark
NON_KF_CONFIG=${CONFIG_DIR}alexnet_128.dnnmark

${EXE} -config ${KF_CONFIG} -debuginfo 1 -warmup 1
${EXE} -config ${NON_KF_CONFIG} -debuginfo 1 -warmup 1
