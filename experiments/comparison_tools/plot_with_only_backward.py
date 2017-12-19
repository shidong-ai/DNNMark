#! /usr/bin/env python

import glob, os
import csv
import re
import sys
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib

label_list = []
kepler_metrics_list = []
pascal_metrics_list = []

usage = "usage: %prog input_file y_axis_label"
def main():
  if len(sys.argv) < 3:
    print usage
    exit()

  input_file_name = sys.argv[1]  
  y_axis_label = sys.argv[2]
  title = input_file_name[0:-4].split('-')[1]
  reader = csv.reader(open(input_file_name, 'rb'))
  header = next(reader)
  for row in reader:
    if "bwd" in row[0]:
      if "_w" in row[0] and "conv1" not in row[0]:
        label_list.insert(0, "")
        label_list.insert(0, row[0][0:-4])
        kepler_metrics_list.insert(0, 0.0)
        pascal_metrics_list.insert(0, 0.0)
        kepler_metrics_list.insert(0, float(row[1]))
        pascal_metrics_list.insert(0, float(row[2]))       
      elif "_d" in row[0]:
        label_list[1] = row[0][0:-4]
        kepler_metrics_list[1] = float(row[1])
        pascal_metrics_list[1] = float(row[2])
      else:
        label_list.insert(0, row[0][0:-4])
        kepler_metrics_list.insert(0, float(row[1]))
        pascal_metrics_list.insert(0, float(row[2]))
  metrics_list = []
  metrics_list.append(kepler_metrics_list)
  metrics_list.append(pascal_metrics_list)

  # Plot parameter
  opacity = 1.0
  bar_width = 1.
  #color_map = ['r', 'c', 'y', 'g', 'grey', 'b', 'm']
  #color_map = ['0.5', '1.0', '0.95', '0.95', '0.3', '0.35', '0.65', '0.75', '0.45', '0.5', '0.8']
  color_map = [
    '#E8EAF6',
    '#C5CAE9',
    '#9FA8DA',
    '#7986CB',
    '#5C6BC0',
    '#3F51B5',
    '#3949AB',
    '#303F9F',
    '#283593',
    '#1A237E',
  ]
  hatch_map = ['', '/', '', '/', '', '\\', '', '\\', '-', '+']
  n_groups = len(label_list)
  num_arch = 2

  index = np.arange(n_groups) * 1.3 * num_arch * bar_width

  matplotlib.rc('font', size=40)
  fig = plt.figure(figsize=(30, 10))
  ax = fig.add_subplot(111)
  arch_list = ["K40", "GTX1080"]
  for i in range(num_arch):
    ax.bar(index + i * bar_width, metrics_list[i], bar_width,
       align='center',
       alpha=opacity,
       color=color_map[i],
       hatch=hatch_map[i],
       label=arch_list[i])
 
  ax.set_xlabel('Layers')
  ax.set_ylabel(y_axis_label)

  ax.yaxis.grid()
  ax.set_xticks(index + bar_width * num_arch / 2)
  ax.set_xticklabels(label_list, rotation='45', ha='right')
  ax.legend(bbox_to_anchor=(0.65, 1.02, 0.35, .102), loc=3,
             ncol=2, mode="expand", borderaxespad=0.)
  ax.set_xlim(min(index)-bar_width, max(index)+bar_width*num_arch)
  if "util" in input_file_name:
    ax.set_ylim(0, 100)

  fig.tight_layout()
  fig.savefig(os.getcwd()+'/figures/'+title+'.pdf', format='pdf', bbox_inches='tight')

if __name__ == '__main__':
  main()

