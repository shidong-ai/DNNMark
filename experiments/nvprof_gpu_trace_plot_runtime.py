#! /usr/bin/env python

import glob, os
import csv
import re
import sys
import collections
from optparse import OptionParser
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
import time

gpu_trace_list = []

regex_kernel = re.compile(r"(?:void)?\s?(?:\w+(?:::))*(\w+)(<[^\(]+>)?\(?.*\)?\s?\[\d+\]")
regex_int = re.compile(r"^(?:[a-zA-z]+)*\s*\(*(\d+)\)*$")
regex_float = re.compile(r"(\d+\.\d+(e\+)*\d*)(?:\w+\/\w+|%)*")

class KernelParam:
  invocation_order = 0
  def __init__(self, full_name, kernel_name, duration, grid_size, block_size,\
               num_register, shared_mem_usage, invocation_order):
    self.full_name = full_name
    self.kernel_name = kernel_name
    self.duration = duration
    self.grid_size = grid_size
    self.block_size = block_size
    self.num_register = num_register
    self.shared_mem_usage = shared_mem_usage
    self.invocation_order = invocation_order

duration_idx = 1
grid_idx = [2, 3, 4]
block_idx = [5, 6, 7]
register_idx = 8
shared_mem_idx = [9, 10]
kernel_idx = 16

db_path = os.getcwd()
conn = sqlite3.connect(db_path+'/alexnet_kernel.db')
table_name = 'alexnet_kernel_dict'
c = conn.cursor()

def ObtainGPUTraceFiles(trace_dir):
  current_dir = os.getcwd()+'/'+trace_dir
  os.chdir(current_dir)
  for f in glob.glob("*-gputrace.csv"):
    gpu_trace_list.append(f)

  if len(gpu_trace_list) == 0:
    print "No GPU trace file in current directory"
    exit()

#--arch
#----batch size
#------kernel param
gpu_trace_dict = collections.OrderedDict()
def GenerateGPUDict():
  # Number of useless lines
  num_useless_lines = 5
  for f in gpu_trace_list:
    invocation_count_dict = collections.OrderedDict()
    print "Process gpu trace file: ", f
    reader = csv.reader(open(f, 'rb'))
    arch = f.split('.')[0].split('-')[2]
    batch_size = f.split('.')[0].split('-')[1]

    query_kernel_str = "SELECT DISTINCT kernelname FROM "+table_name
    c.execute(query_kernel_str)
    kernel_from_sql = c.fetchall()
    kernel_list = [k[0] for k in kernel_from_sql]

    for i in range(num_useless_lines):
      next(reader)
    for row in reader:
      content = row[kernel_idx]
      if regex_kernel.match(content):
        content = row[kernel_idx]
        kernel_name = regex_kernel.match(content).group(1)
        if kernel_name not in kernel_list:
          continue
        template_arguments = regex_kernel.match(content).group(2) if\
          regex_kernel.match(content).group(2) != None else ""
        full_name = kernel_name+template_arguments
        duration = float(row[duration_idx])
        grid_size = 1
        for idx in grid_idx:
          grid_size *= int(row[idx])
        block_size = 1
        for idx in block_idx:
          block_size *= int(row[idx])
        num_register = row[register_idx]
        shared_mem_usage = 0
        for idx in shared_mem_idx:
          shared_mem_usage += float(row[idx])

        if full_name not in invocation_count_dict:
          invocation_count_dict[full_name] = 1
        else:
          invocation_count_dict[full_name] += 1

        kernel_param = KernelParam(full_name, kernel_name, duration, grid_size, block_size,\
                                   num_register, shared_mem_usage, invocation_count_dict[full_name])
        if arch not in gpu_trace_dict:
          gpu_trace_dict[arch] = collections.OrderedDict()

        if batch_size not in gpu_trace_dict[arch]:
          gpu_trace_dict[arch][batch_size] = []

        gpu_trace_dict[arch][batch_size].append(kernel_param)

def Plot(batch_size_list, arch, figure_dir):
  runtime_linear_list = []
  runtime_conv_list = []
  runtime_fc_list = []
  runtime_nonlinear_list = []
  runtime_other_list = []

  # Query the database
  for batch_size in batch_size_list:
    runtime_linear = 0.0
    runtime_conv = 0.0
    runtime_fc = 0.0
    runtime_nonlinear = 0.0
    runtime_other = 0.0
    for kernel_param in gpu_trace_dict[arch][batch_size]:
      full_name = kernel_param.full_name
      invocation_order = kernel_param.invocation_order
      kernel = kernel_param.kernel_name
      query_layer_name_str = "SELECT layername FROM "+table_name+" WHERE batchsize = "+batch_size+" AND fullname = '"+full_name+"' AND invocationorder = "+str(invocation_order)+" AND arch = '"+arch+"'"
      c.execute(query_layer_name_str)
      layer_name_from_sql = c.fetchall()
      if len(layer_name_from_sql) == 0:
        if "dgrad" not in kernel:
          print "Illegal layers!!!"
          print kernel
          print batch_size
          print invocation_order
          print layer_name_from_sql
          exit()
        else:
          continue
      layer_name = layer_name_from_sql[0][0].encode("utf-8")

      if "conv" in layer_name:
        runtime_conv += kernel_param.duration
        runtime_linear += kernel_param.duration
      elif "fc" in layer_name:
        runtime_fc += kernel_param.duration
        runtime_linear += kernel_param.duration
      elif "relu" in layer_name:
        runtime_nonlinear += kernel_param.duration
      else:
        runtime_other += kernel_param.duration
    runtime_linear_list.append(runtime_linear)
    runtime_conv_list.append(runtime_conv)
    runtime_fc_list.append(runtime_fc)
    runtime_nonlinear_list.append(runtime_nonlinear)
    runtime_other_list.append(runtime_other)
  for i in range(3):
    print "Batch size:", batch_size_list[i]
    print "Conv:", runtime_conv_list[i]/ (runtime_linear_list[i]+runtime_nonlinear_list[i]+runtime_other_list[i])
    print "FC:", runtime_fc_list[i]/ (runtime_linear_list[i]+runtime_nonlinear_list[i]+runtime_other_list[i])
    print "Activation", runtime_nonlinear_list[i]/ (runtime_linear_list[i]+runtime_nonlinear_list[i]+runtime_other_list[i])
    print "Other", runtime_other_list[i]/ (runtime_linear_list[i]+runtime_nonlinear_list[i]+runtime_other_list[i])

  # Plot parameter
  opacity = 1.0
  bar_width = 1.
  #color_map = ['g', 'b', 'y', 'r', 'grey', 'b', 'm']
  #color_map = ['0.95', '0.75', '0.55', '0.35', '0.3', '0.35', '0.65', '0.75', '0.45', '0.5', '0.8']
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
  hatch_map = ['/', '//', '\\', '/', '', '\\', '', '\\', '-', '+']

  xtick_name_list = batch_size_list
  n_groups = len(xtick_name_list)

  matplotlib.rc('font', size=40)
  fig = plt.figure(figsize=(25, 10))
  ax = fig.add_subplot(111)

  index = np.arange(n_groups) * 3 * bar_width
  bottom_list = [0] * n_groups
  ax.bar(index, runtime_conv_list, bar_width,
         align='center',
         alpha=opacity,
         color=color_map[0],
         hatch=hatch_map[0],
         linewidth=3,
         label="Run time of Covolution Layers")
  bottom_list = [sum(x) for x in zip(bottom_list, runtime_conv_list)]
  ax.bar(index, runtime_fc_list, bar_width,
         bottom=bottom_list,
         align='center',
         alpha=opacity,
         color=color_map[1],
         hatch=hatch_map[1],
         linewidth=3,
         label="Run time of Fully-Connected Layers")
  bottom_list = [sum(x) for x in zip(bottom_list, runtime_fc_list)]
  ax.bar(index, runtime_nonlinear_list, bar_width,
         bottom=bottom_list,
         align='center',
         alpha=opacity,
         color=color_map[2],
         hatch=hatch_map[2],
         linewidth=3,
         label="Run time of Nonlinearity")
  bottom_list = [sum(x) for x in zip(bottom_list, runtime_nonlinear_list)]
  ax.bar(index, runtime_nonlinear_list, bar_width,
         bottom=bottom_list,
         align='center',
         alpha=opacity,
         color=color_map[3],
         hatch=hatch_map[3],
         linewidth=3,
         label="Run time of Other Techniques")
 
  ax.set_xlabel('Batch Size')
  ax.set_ylabel('Running Time(ms)')
  
  #ax.grid(True)
  ax.yaxis.grid()
  ax.set_xticks(index)
  #ax.set_xticklabels(layer_list[len(layer_list)-len(gpu_dict[kernel][metric]):])
  ax.set_xticklabels(xtick_name_list)
  #ax.legend(bbox_to_anchor=(1.05, 1), loc=2,
  #           ncol=1, borderaxespad=0.)
  ax.legend(bbox_to_anchor=(0.35, 1.02, 0.65, .102), loc=3,
            ncol=1, mode="expand", borderaxespad=0.)
  #ax.set_xlim(min(index)-bar_width, max(index)+bar_width)

  #ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
  fig.tight_layout()
  fig.savefig(os.getcwd()+'/figures/'+arch+'_runtime.pdf', format='pdf', bbox_inches='tight')

usage = "usage: %prog [command] [option1] arg1,arg2 [option2] arg1,arg2"
def main():
  if len(sys.argv) == 1:
    print usage
    exit()

  parser = OptionParser(usage)
  parser.add_option("-d", "--trace_dir", type="string", dest="tracedir",
                    help="tracedir", metavar="TRACE_DIR")
  parser.add_option("-a", "--arch", type="string", dest="arch",
                    help="arch", metavar="ARCH")
  parser.add_option("-n", "--batch-size", type="string", dest="batchsize",
                    help="Batch size seperated by coma", metavar="BATCH_SIZE")

  (options, args) = parser.parse_args()
  if len(args) > 0:
    print "Arguments parsing fail. Possible reason: space occurred between arguments"
    exit()

  batchsize_list = []
  trace_dir = ""
  figure_dir = ""
  arch = ""
  if not options.tracedir:
    parser.error("Trace Directory not given")
  else:
    trace_dir = options.tracedir
  figure_dir = trace_dir + "/figures/"

  if not options.arch:
    parser.error("Architecture not given")
  else:
    arch = options.arch

  if options.batchsize != None:
    batchsize_list = options.batchsize.split(",")

  ObtainGPUTraceFiles(trace_dir)
  
  GenerateGPUDict()
  if len(gpu_trace_dict) == 0:
    print "NO trace found!!!"
    exit()

  Plot(batchsize_list, arch, figure_dir)

if __name__ == '__main__':
  main()
