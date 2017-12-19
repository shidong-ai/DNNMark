#! /bin/bash

MACHINE="$(nvidia-smi --query-gpu=name --format=csv | head -2 | tail -1)"
echo "The machine you are using is:" ${MACHINE}
echo "Please select the correct micro-architecture code"
echo "1. kepler 2. pascal"

read ARCH_CODE

if [ ${ARCH_CODE} -eq 1 ]; then
  ARCH="kepler"
else
  ARCH="pascal"
fi

WORK_DIR="$(pwd)"
GPU_TRACE_PROFILER=alexnet_gputrace_profile.sh
METRIC_PROFILER=alexnet_metrics_profile.sh
PERF_PLOT_SCRIPT=exe_param_plot.sh
MEM_PLOT_SCRIPT=mem_plot.sh
COMPUTE_METRICS_PLOT_SCRIPT=compute_metric_plot.sh

echo "Generating GPU trace in " $ARCH
${WORK_DIR}/${GPU_TRACE_PROFILER} ${ARCH}
echo "DONE"
echo "Generating Metrics in " $ARCH
${WORK_DIR}/${METRIC_PROFILER} ${ARCH}
echo "DONE"
echo "Generating performance related plots for " ${ARCH}
${WORK_DIR}/${PERF_PLOT_SCRIPT} ${ARCH}
echo "DONE"
echo "Generating MEM related plots for " ${ARCH}
${WORK_DIR}/${MEM_PLOT_SCRIPT} ${ARCH}
echo "DONE"
echo "Generating Computing related plots for " ${ARCH}
${WORK_DIR}/${COMPUTE_METRICS_PLOT_SCRIPT} ${ARCH}
echo "DONE"
