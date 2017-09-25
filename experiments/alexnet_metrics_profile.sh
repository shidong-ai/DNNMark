#! /bin/bash

# Setup directories
WORK_DIR="$(pwd)"
CONFIG_DIR=${WORK_DIR}/config/
BENCHMARK_DIR=${WORK_DIR}/../build/benchmarks/
RESULT_DIR=${WORK_DIR}/alexnet_results_kepler/
if [ ! -d "${RESULT_DIR}" ]; then
  mkdir ${RESULT_DIR}
fi

BENCHMARK_LIST=( test_alexnet )
PROFILER=nvprof
ARCH=kepler

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
    for pair in $(python ${WORK_DIR}/queryKernelDB.py ${batch_size} ${ARCH})
    do
      kernel_name="$(echo ${pair} | cut -d "," -f1)"
      invocation_num="$(echo ${pair} | cut -d "," -f2)"
      for (( i=1; i<${invocation_num=}+1; i++ ))
      do
        bm_name=${kernel_name}-${batch_size}-${i}
        echo ${bm_name} ':'
        echo 'Generating metrics file'
        ${PROFILER} --profile-from-start off  --csv --kernels ::${kernel_name}:${i} --metrics all ${EXE} -config ${config} 2>&1 | tee ${RESULT_DIR}${bm_name}-metrics.csv
      done
    done
  done
  cd ..
done


