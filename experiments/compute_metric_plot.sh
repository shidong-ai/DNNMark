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
  TRACE_DIR=results_alexnet_metrics_pascal
fi
if [ $1 = "kepler" ]; then
  TRACE_DIR=results_alexnet_metrics_kepler
fi
FIGURE_DIR=${WORK_DIR}/${TRACE_DIR}/figures/
if [ ! -d "${FIGURE_DIR}" ]; then
  mkdir ${FIGURE_DIR}
fi

BATCH_SIZE_LIST=( 128 )
PROFILER=nvprof
LAYERS1=conv2,relu2,lrn2,pool2,fc6,softmax
LAYERS2=conv1,relu1,lrn1,pool1,conv2,relu2,lrn2,pool2,conv3,relu3,conv4,relu4,conv5,relu5,pool5,fc6,relu6,fc7,relu7,fc8,softmax
if [ "${ARCH}" == "pascal" ]; then
  IPC_METRICS=executed_ipc
  FU_UTIL_METRICS=single_precision_fu_utilization,special_fu_utilization
else
  IPC_METRICS=ipc
  FU_UTIL_METRICS=alu_fu_utilization

fi

for bs in ${BATCH_SIZE_LIST[@]}
do
  SCRIPT=${WORK_DIR}/nvprof_kernel_metric_trace_search_tool.py
  ${SCRIPT} -a ${ARCH} -d ${TRACE_DIR} -t CuUtilization --layer1 ${LAYERS1} --layer2 ${LAYERS2} -n ${bs} -m ${FU_UTIL_METRICS},ldst_fu_utilization,cf_fu_utilization,tex_fu_utilization
  #${SCRIPT} -a ${ARCH} -d ${TRACE_DIR} -t IPC  -l ${LAYERS} -n ${bs} -m ${IPC_METRICS}
  #${SCRIPT} -a ${ARCH} -d ${TRACE_DIR} -t ReplayRate  -l ${LAYERS} -n ${bs} -m inst_replay_overhead

  #${SCRIPT} -a ${ARCH} -d ${TRACE_DIR} -t Efficiency -l ${LAYERS} -n ${bs} -m issue_slot_utilization,flop_sp_efficiency

  ${SCRIPT} -a ${ARCH} -d ${TRACE_DIR} -t StallReason --layer1 ${LAYERS2} --layer2 ${LAYERS2} -n ${bs} -m stall_inst_fetch,stall_exec_dependency,stall_memory_dependency,stall_texture,stall_sync,stall_other,stall_pipe_busy,stall_constant_memory_dependency,stall_memory_throttle,stall_not_selected

  ${SCRIPT} -a ${ARCH} -d ${TRACE_DIR} -t FlopCountSP --layer1 ${LAYERS1} --layer2 ${LAYERS2} -n ${bs} -m flop_count_sp

done

#mv *.pdf ${FIGURE_DIR}
