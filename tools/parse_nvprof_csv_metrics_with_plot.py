#! /usr/bin/python                                                              

import numpy as np
import matplotlib.pyplot as plt
import re                                                                       
import sys
import collections
import csv

if len(sys.argv) == 1:
  print "Input file name is empty"
  exit()

num_files = len(sys.argv) - 1

# Obtain all trace files if there are many
filename_list = []
for i in range(num_files):
  # Obtain file name
  filename_list.append(sys.argv[i+1])

regex_kernel = re.compile(r"(?:void)?\s?(?:\w+(?:::))*(\w+)(?:<.*>)*\(?.*\)?$")
regex_int = re.compile(r"^(?:\w+)*\s*\(*(\d+)\)*$")
regex_float = re.compile(r"(\d+\.\d+(e\+)*\d*)(?:\w+\/\w+|%)*")

kernel_name = ""
filename_kernel_dict = collections.OrderedDict()
metric_dict = collections.OrderedDict()
kernel_idx = 1
metric_name_idx = 3
metric_avg_value_idx = 7;

# Extract occupancy of each trace file
for filename in filename_list:
  # Obtain csv object
  log_file = csv.reader(open(filename, "rb"))

  simplified_filename = filename[0:-12]
  if simplified_filename not in filename_kernel_dict:
    filename_kernel_dict[simplified_filename] = []

  # Number of useless lines
  num_useless_lines = 6

  for i in range(num_useless_lines):
    next(log_file)

  for row in log_file:
    if len(row) < 8:
      continue
    if regex_kernel.match(row[kernel_idx]):
      content = row[kernel_idx]
      kernel_name = regex_kernel.match(content).group(1)
      if kernel_name not in filename_kernel_dict[simplified_filename]:
        filename_kernel_dict[simplified_filename].append(kernel_name)
      if kernel_name not in metric_dict:
        metric_dict[kernel_name] = collections.OrderedDict()
    if regex_int.match(row[metric_avg_value_idx]):
      content = row[metric_avg_value_idx]
      value = int(regex_int.match(content).group(1))
    elif regex_float.match(row[metric_avg_value_idx]):
      content = row[metric_avg_value_idx]
      value = float(regex_float.match(content).group(1))
    metric_dict[kernel_name][row[metric_name_idx]] = value

# General information
regex_type = re.compile(r"(fwd|bwd)_(\w+)")
benchmark_list = []
fwd_benchmark_list = []
bwd_benchmark_list = []
for key in filename_kernel_dict:
  if regex_type.match(key):
    if regex_type.match(key).group(2) not in benchmark_list:
      benchmark_list.append(regex_type.match(key).group(2))
    if "fwd" == regex_type.match(key).group(1):
      fwd_benchmark_list.append(key)
    elif "bwd" == regex_type.match(key).group(1):
      bwd_benchmark_list.append(key)

benchmark_num = len(benchmark_list)
n_groups = benchmark_num
opacity = 0.8
index = np.arange(n_groups)

bar_width = 0.35

# Collect data and generate plot for IPC
metric = "ipc"
prefix = metric
fwd_ipc = []
bwd_ipc = []
for i in range(0, benchmark_num):
  fwd_kernel = filename_kernel_dict[fwd_benchmark_list[i]]
  print fwd_kernel
  fwd_ipc.append(metric_dict[fwd_kernel[-1]][metric])
  bwd_kernel = filename_kernel_dict[bwd_benchmark_list[i]]
  print bwd_kernel
  bwd_ipc.append(metric_dict[bwd_kernel[-1]][metric])

plt.figure(1)
fig, ax = plt.subplots()
rects1 = plt.bar(index, fwd_ipc, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Forward propagation')

rects2 = plt.bar(index + bar_width, bwd_ipc, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Backward propagation')
plt.xlabel('Banchmarks')
plt.ylabel('IPC')

plt.grid()
plt.xticks(index + bar_width, benchmark_list)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
plt.tight_layout()
plt.savefig(prefix + '.pdf', format='pdf', bbox_inches='tight')

# Collect data and generate plot for occupancy
metric = "achieved_occupancy"
prefix = metric
fwd_occupancy = []
bwd_occupancy = []
for i in range(0, benchmark_num):
  fwd_kernel = filename_kernel_dict[fwd_benchmark_list[i]]
  fwd_occupancy.append(metric_dict[fwd_kernel[-1]][metric])
  bwd_kernel = filename_kernel_dict[bwd_benchmark_list[i]]
  bwd_occupancy.append(metric_dict[bwd_kernel[-1]][metric])

plt.figure(1)
fig, ax = plt.subplots()
rects1 = plt.bar(index, fwd_occupancy, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Forward propagation')

rects2 = plt.bar(index + bar_width, bwd_occupancy, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Backward propagation')
plt.xlabel('Banchmarks')
plt.ylabel('Achieved Occpancy')
plt.ylim((0, 1))

plt.grid()
plt.xticks(index + bar_width, benchmark_list)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
plt.tight_layout()
plt.savefig(prefix + '.pdf', format='pdf', bbox_inches='tight')

# Collect data and generate plot for L1/shared utilization
metric = "l1_shared_utilization"
prefix = metric
fwd_l1_shared_utilization = []
bwd_l1_shared_utilization = []
for i in range(0, benchmark_num):
  fwd_kernel = filename_kernel_dict[fwd_benchmark_list[i]]
  fwd_l1_shared_utilization.append(metric_dict[fwd_kernel[-1]][metric])
  bwd_kernel = filename_kernel_dict[bwd_benchmark_list[i]]
  bwd_l1_shared_utilization.append(metric_dict[bwd_kernel[-1]][metric])

plt.figure(1)
fig, ax = plt.subplots()
rects1 = plt.bar(index, fwd_l1_shared_utilization, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Forward propagation')

rects2 = plt.bar(index + bar_width, bwd_l1_shared_utilization, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Backward propagation')
plt.xlabel('Banchmarks')
plt.ylabel('L1/Shared Utilization')
plt.ylim((0, 10))

plt.grid()
plt.xticks(index + bar_width, benchmark_list)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0., fontsize='x-large')
plt.tight_layout()
plt.savefig(prefix + '.pdf', format='pdf', bbox_inches='tight')


# Collect data and generate plot for L1/shared utilization
metric = "l2_utilization"
prefix = metric
fwd_l2_utilization = []
bwd_l2_utilization = []
for i in range(0, benchmark_num):
  fwd_kernel = filename_kernel_dict[fwd_benchmark_list[i]]
  fwd_l2_utilization.append(metric_dict[fwd_kernel[-1]][metric])
  bwd_kernel = filename_kernel_dict[bwd_benchmark_list[i]]
  bwd_l2_utilization.append(metric_dict[bwd_kernel[-1]][metric])

plt.figure(1)
fig, ax = plt.subplots()

rects1 = plt.bar(index, fwd_l2_utilization, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Forward propagation')

rects2 = plt.bar(index + bar_width, bwd_l2_utilization, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Backward propagation')
plt.xlabel('Banchmarks')
plt.ylabel('L2 Utilization')
plt.ylim((0, 10))

plt.grid()
plt.xticks(index + bar_width, benchmark_list)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0., fontsize='x-large')
plt.tight_layout()
plt.savefig(prefix + '.pdf', format='pdf', bbox_inches='tight')

# Collect data and generate plot for L1/shared utilization
metric = "dram_utilization"
prefix = metric
fwd_dram_utilization = []
bwd_dram_utilization = []
for i in range(0, benchmark_num):
  fwd_kernel = filename_kernel_dict[fwd_benchmark_list[i]]
  fwd_dram_utilization.append(metric_dict[fwd_kernel[-1]][metric])
  bwd_kernel = filename_kernel_dict[bwd_benchmark_list[i]]
  bwd_dram_utilization.append(metric_dict[bwd_kernel[-1]][metric])

plt.figure(1)
fig, ax = plt.subplots()

rects1 = plt.bar(index, fwd_dram_utilization, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Forward propagation')

rects2 = plt.bar(index + bar_width, bwd_dram_utilization, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Backward propagation')
plt.xlabel('Banchmarks')
plt.ylabel('DRAM Utilization')
plt.ylim((0, 10))

plt.grid()
plt.xticks(index + bar_width, benchmark_list)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0., fontsize='x-large')
plt.tight_layout()
plt.savefig(prefix + '.pdf', format='pdf', bbox_inches='tight')

# Collect data and generate plot for L1/shared utilization
metric = "alu_fu_utilization"
prefix = metric
fwd_alu_fu_utilization = []
bwd_alu_fu_utilization = []
for i in range(0, benchmark_num):
  fwd_kernel = filename_kernel_dict[fwd_benchmark_list[i]]
  fwd_alu_fu_utilization.append(metric_dict[fwd_kernel[-1]][metric])
  bwd_kernel = filename_kernel_dict[bwd_benchmark_list[i]]
  bwd_alu_fu_utilization.append(metric_dict[bwd_kernel[-1]][metric])

plt.figure(1)
fig, ax = plt.subplots()

rects1 = plt.bar(index, fwd_alu_fu_utilization, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Forward propagation')

rects2 = plt.bar(index + bar_width, bwd_alu_fu_utilization, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Backward propagation')
plt.xlabel('Banchmarks')
plt.ylabel('ALU Utilization')
plt.ylim((0, 10))

plt.grid()
plt.xticks(index + bar_width, benchmark_list)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0., fontsize='x-large')
plt.tight_layout()
plt.savefig(prefix + '.pdf', format='pdf', bbox_inches='tight')

# Collect data and generate plot for L1/shared utilization
metric = "ldst_fu_utilization"
prefix = metric
fwd_ldst_fu_utilization = []
bwd_ldst_fu_utilization = []
for i in range(0, benchmark_num):
  fwd_kernel = filename_kernel_dict[fwd_benchmark_list[i]]
  fwd_ldst_fu_utilization.append(metric_dict[fwd_kernel[-1]][metric])
  bwd_kernel = filename_kernel_dict[bwd_benchmark_list[i]]
  bwd_ldst_fu_utilization.append(metric_dict[bwd_kernel[-1]][metric])

plt.figure(1)
fig, ax = plt.subplots()

rects1 = plt.bar(index, fwd_ldst_fu_utilization, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Forward propagation')

rects2 = plt.bar(index + bar_width, bwd_ldst_fu_utilization, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Backward propagation')
plt.xlabel('Banchmarks')
plt.ylabel('Load/Store Unit Utilization')
plt.ylim((0, 10))

plt.grid()
plt.xticks(index + bar_width, benchmark_list)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0., fontsize='x-large')
plt.tight_layout()
plt.savefig(prefix + '.pdf', format='pdf', bbox_inches='tight')


# Collect data and generate plot for stall reason of Forward
prefix = "stall_reason_fwd"
fwd_stall_inst_fetch = []
fwd_stall_exec_dependency = []
fwd_stall_texture = []
fwd_stall_sync = []
fwd_stall_other = []
fwd_stall_memory_dependency = []
fwd_stall_pipe_busy = []
fwd_stall_constant_memory_dependency = []
fwd_stall_memory_throttle = []
fwd_stall_not_selected = []
for i in range(0, benchmark_num):
  fwd_kernel = filename_kernel_dict[fwd_benchmark_list[i]]
  fwd_stall_inst_fetch.append(metric_dict[fwd_kernel[-1]]["stall_inst_fetch"])
  fwd_stall_exec_dependency.append(metric_dict[fwd_kernel[-1]]["stall_exec_dependency"])
  fwd_stall_texture.append(metric_dict[fwd_kernel[-1]]["stall_texture"])
  fwd_stall_sync.append(metric_dict[fwd_kernel[-1]]["stall_sync"])
  fwd_stall_other.append(metric_dict[fwd_kernel[-1]]["stall_other"])
  fwd_stall_memory_dependency.append(metric_dict[fwd_kernel[-1]]["stall_memory_dependency"])
  fwd_stall_pipe_busy.append(metric_dict[fwd_kernel[-1]]["stall_pipe_busy"])
  fwd_stall_constant_memory_dependency.append(metric_dict[fwd_kernel[-1]]["stall_constant_memory_dependency"])
  fwd_stall_memory_throttle.append(metric_dict[fwd_kernel[-1]]["stall_memory_throttle"])
  fwd_stall_not_selected.append(metric_dict[fwd_kernel[-1]]["stall_not_selected"])

plt.figure(2)
rects1 = plt.bar(index, fwd_stall_inst_fetch, bar_width,
                 alpha=opacity,
                 color='b',
                 label='stall_inst_fetch')
rects2 = plt.bar(index, fwd_stall_exec_dependency, bar_width,
                 alpha=opacity,
                 color='r',
                 bottom=fwd_stall_inst_fetch,
                 label='stall_exec_dependency')
rects3 = plt.bar(index, fwd_stall_texture, bar_width,
                 alpha=opacity,
                 color='g',
                 bottom=fwd_stall_exec_dependency,
                 label='stall_texture')
rects4 = plt.bar(index, fwd_stall_sync, bar_width,
                 alpha=opacity,
                 color='c',
                 bottom=fwd_stall_texture,
                 label='stall_sync')
rects5 = plt.bar(index, fwd_stall_other, bar_width,
                 alpha=opacity,
                 color='m',
                 bottom=fwd_stall_sync,
                 label='stall_other')
rects6 = plt.bar(index, fwd_stall_memory_dependency, bar_width,
                 alpha=opacity,
                 color='y',
                 bottom=fwd_stall_other,
                 label='stall_memory_dependency')
rects7 = plt.bar(index, fwd_stall_pipe_busy, bar_width,
                 alpha=opacity,
                 color='k',
                 hatch='/',
                 bottom=fwd_stall_memory_dependency,
                 label='stall_pipe_busy')
rects8 = plt.bar(index, fwd_stall_constant_memory_dependency, bar_width,
                 alpha=opacity,
                 color='0.25',
                 hatch='//',
                 bottom=fwd_stall_pipe_busy,
                 label='stall_constant_memory_dependency')
rects9 = plt.bar(index, fwd_stall_memory_throttle, bar_width,
                 alpha=opacity,
                 color='0.5',
                 hatch='-',
                 bottom=fwd_stall_constant_memory_dependency,
                 label='stall_memory_throttle')
rects9 = plt.bar(index, fwd_stall_not_selected, bar_width,
                 alpha=opacity,
                 color='0.75',
                 hatch='o',
                 bottom=fwd_stall_memory_throttle,
                 label='stall_not_selected')

plt.xlabel('Benchmarks')
plt.ylabel('Percentage of Stall Reason(%)')
plt.ylim((0, 100))
plt.grid()
plt.xticks(index + bar_width/2, benchmark_list)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1, mode="expand", borderaxespad=0.)
plt.tight_layout()
plt.savefig(prefix + '.pdf', format='pdf', bbox_inches='tight')


# Collect data and generate plot for stall reason of Forward
prefix = "stall_reason_bwd"
bwd_stall_inst_fetch = []
bwd_stall_exec_dependency = []
bwd_stall_texture = []
bwd_stall_sync = []
bwd_stall_other = []
bwd_stall_memory_dependency = []
bwd_stall_pipe_busy = []
bwd_stall_constant_memory_dependency = []
bwd_stall_memory_throttle = []
bwd_stall_not_selected = []
for i in range(0, benchmark_num):
  bwd_kernel = filename_kernel_dict[bwd_benchmark_list[i]]
  bwd_stall_inst_fetch.append(metric_dict[bwd_kernel[-1]]["stall_inst_fetch"])
  bwd_stall_exec_dependency.append(metric_dict[bwd_kernel[-1]]["stall_exec_dependency"])
  bwd_stall_texture.append(metric_dict[bwd_kernel[-1]]["stall_texture"])
  bwd_stall_sync.append(metric_dict[bwd_kernel[-1]]["stall_sync"])
  bwd_stall_other.append(metric_dict[bwd_kernel[-1]]["stall_other"])
  bwd_stall_memory_dependency.append(metric_dict[bwd_kernel[-1]]["stall_memory_dependency"])
  bwd_stall_pipe_busy.append(metric_dict[bwd_kernel[-1]]["stall_pipe_busy"])
  bwd_stall_constant_memory_dependency.append(metric_dict[bwd_kernel[-1]]["stall_constant_memory_dependency"])
  bwd_stall_memory_throttle.append(metric_dict[bwd_kernel[-1]]["stall_memory_throttle"])
  bwd_stall_not_selected.append(metric_dict[bwd_kernel[-1]]["stall_not_selected"])

plt.figure(3)
rects1 = plt.bar(index, bwd_stall_inst_fetch, bar_width,
                 alpha=opacity,
                 color='b',
                 label='stall_inst_fetch')
rects2 = plt.bar(index, bwd_stall_exec_dependency, bar_width,
                 alpha=opacity,
                 color='r',
                 bottom=bwd_stall_inst_fetch,
                 label='stall_exec_dependency')
rects3 = plt.bar(index, bwd_stall_texture, bar_width,
                 alpha=opacity,
                 color='g',
                 bottom=bwd_stall_exec_dependency,
                 label='stall_texture')
rects4 = plt.bar(index, bwd_stall_sync, bar_width,
                 alpha=opacity,
                 color='c',
                 bottom=bwd_stall_texture,
                 label='stall_sync')
rects5 = plt.bar(index, bwd_stall_other, bar_width,
                 alpha=opacity,
                 color='m',
                 bottom=bwd_stall_sync,
                 label='stall_other')
rects6 = plt.bar(index, bwd_stall_memory_dependency, bar_width,
                 alpha=opacity,
                 color='y',
                 bottom=bwd_stall_other,
                 label='stall_memory_dependency')
rects7 = plt.bar(index, bwd_stall_pipe_busy, bar_width,
                 alpha=opacity,
                 color='k',
                 hatch='/',
                 bottom=bwd_stall_memory_dependency,
                 label='stall_pipe_busy')
rects8 = plt.bar(index, bwd_stall_constant_memory_dependency, bar_width,
                 alpha=opacity,
                 color='0.25',
                 hatch='//',
                 bottom=bwd_stall_pipe_busy,
                 label='stall_constant_memory_dependency')
rects9 = plt.bar(index, bwd_stall_memory_throttle, bar_width,
                 alpha=opacity,
                 color='0.5',
                 hatch='-',
                 bottom=bwd_stall_constant_memory_dependency,
                 label='stall_memory_throttle')
rects9 = plt.bar(index, bwd_stall_not_selected, bar_width,
                 alpha=opacity,
                 color='0.75',
                 hatch='o',
                 bottom=bwd_stall_memory_throttle,
                 label='stall_not_selected')

plt.xlabel('Benchmarks')
plt.ylabel('Percentage of Stall Reason(%)')
plt.ylim((0, 100))
plt.grid()
plt.xticks(index + bar_width/2, benchmark_list)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1, mode="expand", borderaxespad=0.)
plt.tight_layout()
plt.savefig(prefix + '.pdf', format='pdf', bbox_inches='tight')


# Collect data and generate plot for L1/shared utilization
metric = "eligible_warps_per_cycle"
prefix = metric
fwd_eligible_warps = []
bwd_eligible_warps = []
for i in range(0, benchmark_num):
  fwd_kernel = filename_kernel_dict[fwd_benchmark_list[i]]
  fwd_eligible_warps.append(metric_dict[fwd_kernel[-1]][metric])
  bwd_kernel = filename_kernel_dict[bwd_benchmark_list[i]]
  bwd_eligible_warps.append(metric_dict[bwd_kernel[-1]][metric])

plt.figure(1)
fig, ax = plt.subplots()

rects1 = plt.bar(index, fwd_eligible_warps, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Forward propagation')

rects2 = plt.bar(index + bar_width, bwd_eligible_warps, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Backward propagation')
plt.xlabel('Banchmarks')
plt.ylabel('Eligible Warps Per Cyle')
plt.ylim((0, 10))

plt.grid()
plt.xticks(index + bar_width, benchmark_list)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
plt.tight_layout()
plt.savefig(prefix + '.pdf', format='pdf', bbox_inches='tight')

exit()


