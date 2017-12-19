#! /bin/bash

if [ $# -eq 1 ]; then
  echo "Archtecture:" $1
else
  echo "Usage: ./<script> <arch>"
  exit
fi

# Setup directories
WORK_DIR="$(pwd)"
ARCH=$1
if [ $1 = "pascal" ]; then
  TRACE_DIR=results_alexnet_gputrace_pascal
fi
if [ $1 = "kepler" ]; then
  TRACE_DIR=results_alexnet_gputrace_kepler
fi
FIGURE_DIR=${WORK_DIR}/${TRACE_DIR}/figures/
if [ ! -d "${FIGURE_DIR}" ]; then
  mkdir ${FIGURE_DIR}
fi

BATCH_SIZE_LIST=( 16,64,128 )

SCRIPT=${WORK_DIR}/nvprof_gpu_trace_plot_runtime.py
${SCRIPT} -a ${ARCH} -d ${TRACE_DIR} -n ${BATCH_SIZE_LIST}

