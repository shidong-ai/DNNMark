#! /bin/bash

# Setup directories
WORK_DIR="$(pwd)"
CONFIG_DIR=${WORK_DIR}/config/
BENCHMARK_DIR=${WORK_DIR}/../build/benchmarks/
RESULT_DIR=${WORK_DIR}/alexnet_results_kepler/
ARCH=kepler
if [ ! -d "${RESULT_DIR}" ]; then
  mkdir ${RESULT_DIR}
fi

BENCHMARK_LIST=( test_alexnet )
PROFILER=nvprof

for bm in ${BENCHMARK_LIST[@]}
do
  EXE="$(find ${BENCHMARK_DIR}${bm} -executable -type f)"
  CONFIG_SUBDIR=${CONFIG_DIR}"$(echo ${bm} | cut -d "_" -f2)" 
  cd ${CONFIG_SUBDIR}
  CONFIG_LIST="$(ls *.dnnmark)"
  for config in ${CONFIG_LIST[@]}
  do
    echo $config
    batch_size="$(echo $config | cut -d "." -f1 | cut -d "_" -f3)"
    bm_name="$(echo ${bm} | cut -d "_" -f2)"-${batch_size}-${ARCH}
    echo ${bm_name} ':'
    echo 'Generating gpu trace file'
    ${PROFILER} --profile-from-start off --csv --print-gpu-trace ${EXE} -config ${config} 2>&1 | tee ${RESULT_DIR}${bm_name}-gputrace.csv
  done
  cd ..
done


