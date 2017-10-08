#! /bin/bash

if [ $# -eq 1 ]; then
  echo "Archtecture:" $1
else
  echo "Usage: ./<script> <arch>"
  exit
fi

# Setup directories
WORK_DIR="$(pwd)"
CONFIG_DIR=${WORK_DIR}/config/
BENCHMARK_DIR=${WORK_DIR}/../build/benchmarks/
ARCH=$1

if [ $1 = "pascal" ]; then
  TRACE_DIR=conv_results_pascal
fi
if [ $1 = "kepler" ]; then
  TRACE_DIR=conv_results_kepler
fi

RESULT_DIR=${WORK_DIR}/${TRACE_DIR}_gputrace/
if [ ! -d "${RESULT_DIR}" ]; then
  mkdir ${RESULT_DIR}
fi

BENCHMARK_LIST=( test_fwd_conv test_bwd_conv )
PROFILER=nvprof

for bm in ${BENCHMARK_LIST[@]}
do
  EXE="$(find ${BENCHMARK_DIR}${bm} -executable -type f)"
  CONFIG_SUBDIR=${CONFIG_DIR}"$(echo ${bm} | cut -d "_" -f3)" 
  cd ${CONFIG_SUBDIR}
  CONFIG_LIST="$(ls *.dnnmark)"
  for config in ${CONFIG_LIST[@]}
  do
    echo $config
    batch_size="$(echo $config | cut -d "." -f1 | cut -d "_" -f3)"
    algo="$(echo $config | cut -d "." -f1 | cut -d "_" -f4)"
    prop="$(echo ${bm} | cut -d "_" -f2)"
    bm_name="$(echo ${bm} | cut -d "_" -f3)"-${prop}-${batch_size}-${algo}-${ARCH}
    echo ${bm_name} ':'
    echo 'Generating gpu trace file'
    ${PROFILER} --profile-from-start off --csv --print-gpu-trace ${EXE} -config ${config} 2>&1 | tee ${RESULT_DIR}${bm_name}-gputrace.csv
  done
  cd ..
done


