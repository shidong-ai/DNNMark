#! /usr/bin/env python

import glob, os
import csv
import re
import sys
import collections

kepler_trace_list = []
pascal_trace_list = []

work_path = os.getcwd()
for f in glob.glob("kepler*.csv"):
  kepler_trace_list.append(f)
for f in glob.glob("pascal*.csv"):
  pascal_trace_list.append(f)

for kf in kepler_trace_list:
  k_reader = csv.reader(open(kf, 'rb'))
  k_metric_name = kf[0:-4].split('-')[1]
  for pf in pascal_trace_list:
    p_metric_name = pf[0:-4].split('-')[1]
    if k_metric_name in p_metric_name or k_metric_name == p_metric_name or p_metric_name in k_metric_name or (k_metric_name == "alu_fu_utilization" and p_metric_name == "single_precision_fu_utilization"):
      p_reader = csv.reader(open(pf, 'rb'))
      output_name = "comparison-"+k_metric_name+".csv"
      o_writer = csv.writer(open(output_name, 'wb'))
      o_writer.writerow(["layer_name", "kepler", "pascal"])
      for kepler_metrics in k_reader:
        row = []
        row.append(kepler_metrics[0])
        row.append(kepler_metrics[1])
        pascal_metrics = next(p_reader)
        row.append(pascal_metrics[1])
        o_writer.writerow(row)

