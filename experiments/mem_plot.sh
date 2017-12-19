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
  #TRACE_DIR=alexnet_results_pascal
fi
if [ $1 = "kepler" ]; then
  TRACE_DIR=results_alexnet_metrics_kepler
fi
FIGURE_DIR=${WORK_DIR}/${TRACE_DIR}/figures/
if [ ! -d "${FIGURE_DIR}" ]; then
  mkdir ${FIGURE_DIR}
fi

#BATCH_SIZE_LIST=( 16 64 128 )
BATCH_SIZE_LIST=( 128 )
PROFILER=nvprof
#LAYERS=conv1,relu1,lrn1,pool1,conv2,relu2,lrn2,pool2,conv3,relu3,conv4,relu4,conv5,relu5,pool5,fc6,relu6,fc7,relu7,fc8,softmax
LAYERS=conv2,relu2,lrn2,pool2,fc6,softmax
if [ "${TRACE_DIR}" == "alexnet_results_pascal_l1_enable" ]; then
  CACHE_HIT_RATE_METRICS=tex_cache_hit_rate,l2_tex_read_hit_rate,l2_tex_write_hit_rate
  MEM_TRAN1_METRICS=tex_cache_transactions,shared_load_transactions,shared_store_transactions,l2_tex_read_transactions,l2_tex_write_transactions
  MEM_THROUGH1_METRICS=tex_cache_throughput,shared_load_throughput,shared_store_throughput,l2_tex_read_throughput,l2_tex_write_throughput
  MEM_UTIL_METRICS=shared_utilization,tex_utilization,l2_utilization,dram_utilization
else
  CACHE_HIT_RATE_METRICS=l1_cache_global_hit_rate,l1_cache_local_hit_rate,tex_cache_hit_rate,l2_l1_read_hit_rate,l2_texture_read_hit_rate
  MEM_TRAN1_METRICS=tex_cache_transactions,shared_load_transactions,shared_store_transactions,l2_l1_read_transactions,l2_l1_write_transactions,l2_tex_read_transactions
  MEM_THROUGH1_METRICS=tex_cache_throughput,shared_load_throughput,shared_store_throughput,l2_l1_read_throughput,l2_l1_write_throughput,l2_texture_read_throughput
  MEM_UTIL_METRICS=l1_shared_utilization,tex_utilization,l2_utilization,dram_utilization
fi

for bs in ${BATCH_SIZE_LIST[@]}
do
  SCRIPT=${WORK_DIR}/nvprof_kernel_metric_trace_search_tool.py
  ${SCRIPT} -a ${ARCH} -d ${TRACE_DIR} -t CacheHitRate -l ${LAYERS} -n ${bs} -m ${CACHE_HIT_RATE_METRICS}

  ${SCRIPT} -a ${ARCH} -d ${TRACE_DIR} -t MemTransaction1 -l ${LAYERS} -n ${bs} -m ${MEM_TRAN1_METRICS}

  ${SCRIPT} -a ${ARCH} -d ${TRACE_DIR} -t MemThroughput1 -l ${LAYERS} -n ${bs} -m ${MEM_THROUGH1_METRICS}

  ${SCRIPT} -a ${ARCH} -d ${TRACE_DIR} -t MemTransaction2 -l ${LAYERS} -n ${bs} -m l2_read_transactions,l2_write_transactions,dram_read_transactions,dram_write_transactions
  ${SCRIPT} -a ${ARCH} -d ${TRACE_DIR} -t MemThroughput2 -l ${LAYERS} -n ${bs} -m l2_read_throughput,l2_write_throughput,dram_read_throughput,dram_write_throughput

  ${SCRIPT} -a ${ARCH} -d ${TRACE_DIR} -t MemUtilization -l ${LAYERS} -n ${bs} -m ${MEM_UTIL_METRICS}

done

