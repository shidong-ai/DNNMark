#! /bin/bash

WORK_DIR="$(pwd)"

COMBINED_TRACE_SCRIPT=combine_metrics.py
echo "Generate combined trace files"
${WORK_DIR}/${COMBINED_TRACE_SCRIPT}
echo "DONE"

FIGURE_DIR=${WORK_DIR}/figures/
if [ ! -d "${FIGURE_DIR}" ]; then
  mkdir ${FIGURE_DIR}
fi

PLOT_SPEEDUP=plot_with_normalization.py
echo "Generate speedup chart"
${WORK_DIR}/${PLOT_SPEEDUP} comparison-exetime.csv Speedup
echo "DONE"

PLOT_OTHER=plot_with_only_backward.py
echo "Generate floating-point instruction counts chart"
${WORK_DIR}/${PLOT_OTHER} comparison-flop_count_sp.csv FP\ Inst\ Counts
echo "DONE"

echo "Generate compute unit utilization chart"
${WORK_DIR}/${PLOT_OTHER} comparison-alu_fu_utilization.csv ALU\ Utilization
echo "DONE"
