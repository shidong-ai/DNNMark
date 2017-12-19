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

#echo "Generating GPU trace in " $ARCH
#${WORK_DIR}/${GPU_TRACE_PROFILER} ${ARCH}
#echo "DONE"
#echo "Generating Metrics in " $ARCH
#${WORK_DIR}/${METRIC_PROFILER} ${ARCH}
#echo "DONE"
#echo "Generating performance related plots for " ${ARCH}
#${WORK_DIR}/${PERF_PLOT_SCRIPT} ${ARCH}
#echo "DONE"
#echo "Generating MEM related plots for " ${ARCH}
#${WORK_DIR}/${MEM_PLOT_SCRIPT} ${ARCH}
#echo "DONE"
#echo "Generating Computing related plots for " ${ARCH}
#${WORK_DIR}/${COMPUTE_METRICS_PLOT_SCRIPT} ${ARCH}
#echo "DONE"

COMPARE_TOOLS_DIR=${WORK_DIR}/comparison_tools
if [ ! -d "${COMPARE_TOOLS_DIR=}" ]; then
  echo "No comparison tools"
  exit
fi
RESULTS_DIR="$(ls | grep results)"
for results in ${RESULTS_DIR[@]}
do
  cd ${results}
  cp ${ARCH}*.csv ${COMPARE_TOOLS_DIR}
  cd ..
done

echo "You should run the same thing on the other machine (K40 or GTX1080) in order to have performance comparison charts"
echo "Copy the metric trace files starting with microarchitecture code in experiments/comparison_tools from one machine to another"
echo "(To the same folder - experiments/comparison_tools)"
echo "e.g. From G1080 to K40"
echo "Have you finished copy?"
echo "1. Yes 2. No"
read ANSWER_CODE

if [ ${ANSWER_CODE} -eq 1 ]; then
  echo "Great! Now go to experiments/comparison_tools folder and run runme.sh"
else
  echo "You should copy the results from other machine in comparison_tools"
  echo "And then go to experiments/comparison_tools folder in current machine and run runme.sh"
fi
