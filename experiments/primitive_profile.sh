#! /bin/bash

# Setup directories
WORK_DIR="$(pwd)"
CONFIG_DIR=${WORK_DIR}/config/
BENCHMARK_DIR=${WORK_DIR}/../build/benchmarks/
RESULT_DIR=${WORK_DIR}/primtives_results/
if [ ! -d "${RESULT_DIR}" ]; then
  mkdir ${RESULT_DIR}
fi

BENCHMARK_LIST=( test_fwd_conv test_fwd_pool test_fwd_lrn test_fwd_activation\
                 test_fwd_fc test_fwd_softmax test_bwd_conv test_bwd_pool\
                 test_bwd_lrn test_bwd_activation test_bwd_softmax )
#BENCHMARK_LIST=( test_alexnet )
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
    bm_name=$(echo $config | cut -d "." -f1)
    propagation=$(echo ${bm} | cut -d "_" -f2)
    bm_name=${propagation}_${bm_name}
    echo ${bm_name} ':'
    echo 'Exporting profile file'
    ${PROFILER} --profile-from-start off --analysis-metrics --export-profile ${RESULT_DIR}${bm_name}.prof ${EXE} -config ${config}
    echo 'Generating gpu trace file'
    ${PROFILER} --profile-from-start off  --csv --print-gpu-trace ${EXE} -config ${config} 2>&1 | tee ${RESULT_DIR}${bm_name}_gpu_trace.csv
    echo 'Generating api trace file'
    ${PROFILER} --profile-from-start off  --csv --print-api-trace ${EXE} -config ${config} 2>&1 | tee ${RESULT_DIR}${bm_name}_api_trace.csv
    echo 'Generating events file'
    ${PROFILER} --profile-from-start off  --csv --events all ${EXE} -config ${config} 2>&1 | tee ${RESULT_DIR}${bm_name}_events.csv
    echo 'Generating metrics file'
    ${PROFILER} --profile-from-start off  --csv --metrics all ${EXE} -config ${config} 2>&1 | tee ${RESULT_DIR}${bm_name}_metrics.csv
  done
  cd ..
done


