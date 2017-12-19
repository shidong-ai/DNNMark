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

def BarChartByTitle(title, batch_size_list, arch, figure_dir):
  #--arch
  #----propagation
  #------batch size
  gpu_dict = collections.OrderedDict()
  xtick_name = collections.OrderedDict()
  count = collections.OrderedDict()

  # Query the database
  for batch_size in batch_size_list:
    for arch in gpu_trace_dict:
      for gpu_trace in gpu_trace_dict[arch][batch_size]:
        full_name = gpu_trace.full_name
        invocation_order = gpu_trace.invocation_order
        kernel = gpu_trace.kernel_name

        query_layer_name_str = "SELECT layername FROM "+table_name+" WHERE batchsize = "+batch_size+" AND fullname = '"+full_name+"' AND invocationorder = "+str(invocation_order)
        c.execute(query_layer_name_str)
        layer_name_from_sql = c.fetchall()
        if len(layer_name_from_sql) == 0:
          if "dgrad" not in kernel:
            print "Illegal layers!!!"
            print kernel
            print layer_name_from_sql
            exit()
          else:
            continue
        layer_name = layer_name_from_sql[0][0].encode("utf-8")

        query_propagation_str = "SELECT propagation FROM "+table_name+" WHERE batchsize = "+batch_size+" AND fullname = '"+full_name+"' AND invocationorder = "+str(invocation_order)
        c.execute(query_propagation_str)
        propagation_from_sql = c.fetchall()
        if len(propagation_from_sql) == 0:
          if "dgrad" not in kernel:
            print "Ilegal propagation!!!"
            print kernel
            print propagation_from_sql
            exit()
          else:
            continue
        propagation = propagation_from_sql[0][0].encode("utf-8")

        if arch not in gpu_dict:
          gpu_dict[arch] = collections.OrderedDict()
          xtick_name[arch] = collections.OrderedDict()
          count[arch] = collections.OrderedDict()
        if propagation not in gpu_dict[arch]:
          gpu_dict[arch][propagation] = collections.OrderedDict()
          xtick_name[arch][propagation] = collections.OrderedDict()
          count[arch][propagation] = collections.OrderedDict()
        if batch_size not in gpu_dict[arch][propagation]:
          gpu_dict[arch][propagation][batch_size] = []
          xtick_name[arch][propagation][batch_size] = []
          count[arch][propagation][batch_size] = 0

        count[arch][propagation][batch_size] += 1
        if propagation == "Forward":
          xtick_name[arch][propagation][batch_size].append(layer_name)
        elif propagation == "Backward":
          xtick_name[arch][propagation][batch_size].insert(0, layer_name)

        if title == "":
          exit()

        if propagation == "Forward":
          if title == "RunningTime":
            gpu_dict[arch][propagation][batch_size].append(float(gpu_trace.duration))
          elif title == "BlockSize":
            gpu_dict[arch][propagation][batch_size].append(int(gpu_trace.block_size))
          elif title == "NumRegister":
            gpu_dict[arch][propagation][batch_size].append(int(gpu_trace.num_register))
          elif title == "SharedMemSize":
            gpu_dict[arch][propagation][batch_size].append(float(gpu_trace.shared_mem_usage))
        elif propagation == "Backward":
          if title == "RunningTime":
            gpu_dict[arch][propagation][batch_size].insert(0, float(gpu_trace.duration))
          elif title == "BlockSize":
            gpu_dict[arch][propagation][batch_size].insert(0, int(gpu_trace.block_size))
          elif title == "NumRegister":
            gpu_dict[arch][propagation][batch_size].insert(0, int(gpu_trace.num_register))
          elif title == "SharedMemSize":
            gpu_dict[arch][propagation][batch_size].insert(0, float(gpu_trace.shared_mem_usage))

  # Plot parameter
  opacity = 1.0
  bar_width = 1.
  #color_map = ['r', 'c', 'y', 'g', 'grey', 'b', 'm']
  color_map = ['0.5', '0.5', '0.95', '0.95', '0.3', '0.35', '0.65', '0.75', '0.45', '0.5', '0.8']
  hatch_map = ['', '/', '', '/', '', '\\', '', '\\', '-', '+']
  del_idx_dict = collections.OrderedDict()
  exclude_conv = False
  conv_only = True

  for arch in gpu_dict:
    for propagation in gpu_dict[arch]:
      batch_size_keys = xtick_name[arch][propagation].keys()
      xtick_name_list = xtick_name[arch][propagation][batch_size_keys[0]]
      num_batch_size = len(batch_size_keys)
      n_groups = count[arch][propagation][batch_size_keys[0]]

      i = 0
      matplotlib.rc('font', size=40)
      fig = plt.figure(figsize=(70, 10))
      ax = fig.add_subplot(111)

      if arch not in del_idx_dict:
        del_idx_dict[arch] = collections.OrderedDict()
      if propagation not in del_idx_dict:
        del_idx_dict[arch][propagation] = []
      for name in xtick_name_list:
        if exclude_conv and not conv_only:
          if "conv" in name:
            #if len(del_idx_dict[arch][propagation]) > 0 and xtick_name_list.index(name) == del_idx_dict[arch][propagation][-1]:
            #  del_idx_dict[arch][propagation].append(xtick_name_list.index(name) + 1) 
            del_idx_dict[arch][propagation].append(xtick_name_list.index(name))
        if conv_only and not exclude_conv:
          if "conv" not in name:
            #if len(del_idx_dict[arch][propagation]) > 0 and xtick_name_list.index(name) == del_idx_dict[arch][propagation][-1]:
            #  del_idx_dict[arch][propagation].append(xtick_name_list.index(name) + 1) 
            del_idx_dict[arch][propagation].append(xtick_name_list.index(name))

      sorted_del_idx = sorted(del_idx_dict[arch][propagation], reverse=True)
      if len(sorted_del_idx) > 0:
        for del_idx in sorted_del_idx:
          del xtick_name_list[del_idx]

      n_groups -= len(del_idx_dict[arch][propagation])
      index = np.arange(n_groups) * 2 * num_batch_size * bar_width
      for batch_size in gpu_dict[arch][propagation]:
        if exclude_conv or conv_only:
          for del_idx in sorted_del_idx:
            del gpu_dict[arch][propagation][batch_size][del_idx]
              
        ax.bar(index + i * bar_width, gpu_dict[arch][propagation][batch_size], bar_width,
               align='center',
               alpha=opacity,
               color=color_map[i],
               hatch=hatch_map[i],
               label="Batch Size: "+batch_size)
        i += 1
 
      ax.set_xlabel('Layers')
      if title == "RunningTime":
        ax.set_ylabel('Running Time(ms)')
      if title == "BlockSize":
        ax.set_ylabel('Number of Thread Per Block')
      if title == "NumRegister":
        ax.set_ylabel('Number of Registers')
      if title == "SharedMemSize":
        ax.set_ylabel('Shared Mem Size(KB)')
      
      ax.grid(True)
      ax.set_xticks(index + bar_width * num_batch_size / 2)
      #ax.set_xticklabels(layer_list[len(layer_list)-len(gpu_dict[kernel][metric]):])
      ax.set_xticklabels(xtick_name_list, rotation='45', ha='right')
      #ax.legend(bbox_to_anchor=(1.05, 1), loc=2,
      #           ncol=1, borderaxespad=0.)
      if num_batch_size > 1:
        ax.legend(bbox_to_anchor=(0.4, 1.02, 0.6, .102), loc=3,
                  ncol=num_batch_size, mode="expand", borderaxespad=0.)
      else:
        ax.legend(bbox_to_anchor=(0.4, 1.02, 0.6, .102), loc=3,
                  ncol=1, mode="expand", borderaxespad=0.)
      ax.set_xlim(min(index)-bar_width, max(index)+bar_width*num_batch_size)

      #ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
      fig.tight_layout()
      if num_batch_size > 1:
        if conv_only:
          indicator = "_conv_only"
        elif exclude_conv:
          indicator = "_exclude_conv"
        else:
          indicator = ""
        fig.savefig(os.getcwd()+'/figures/'+title+"_"+arch+'_'+propagation+''+indicator+'.pdf', format='pdf', bbox_inches='tight')
      else:
        fig.savefig(os.getcwd()+'/figures/'+title+"_"+arch+'_'+propagation+"_"+batch_size+'.pdf', format='pdf', bbox_inches='tight')

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
  parser.add_option("-c", "--command", type="string", dest="command",
                    help="command seperated by coma", metavar="COMMAND")
  parser.add_option("-n", "--batch-size", type="string", dest="batchsize",
                    help="Batch size seperated by coma", metavar="BATCH_SIZE")
  parser.add_option("-t", "--title", type="string", dest="title",
                    help="Plot title", metavar="TITLE")

  (options, args) = parser.parse_args()
  if len(args) > 0:
    print "Arguments parsing fail. Possible reason: space occurred between arguments"
    exit()

  command = ""
  batchsize_list = []
  title = ""
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

  if options.command != None:
    command = options.command
  if options.batchsize != None:
    batchsize_list = options.batchsize.split(",")
  if options.title != None:
    title = options.title

  ObtainGPUTraceFiles(trace_dir)
  
  GenerateGPUDict()
  if len(gpu_trace_dict) == 0:
    print "NO trace found!!!"
    exit()

  command_options = ["genDict", "Plot"]
  if command not in command_options:
    print usage
    print "Command options: ", command_options
    exit()

  title_options = ["RunningTime", "BlockSize", "NumRegister", "SharedMemSize"]

  if command == command_options[0]:
    # Manually adding information is needed
    filename = "layer_kernel_dict"+"_"+time.strftime("%m%d%Y")+".csv"
    writer = csv.writer(open(filename, 'wb'))
    for arch in gpu_trace_dict:
      for batchsize in gpu_trace_dict[arch]:
        #invocation_dict = collections.OrderedDict()
        for param in gpu_trace_dict[arch][batchsize]:
          row = []
          #if param.full_name not in invocation_dict:
          #  invocation_dict[param.full_name] = 0
          #invocation_dict[param.full_name] += 1
          #param.invocation_order = invocation_dict[param.full_name]
          row.append(arch)
          row.append(batchsize)
          row.append(param.kernel_name)
          row.append(param.full_name)
          row.append(param.invocation_order)
          writer.writerow(row)
  elif command == command_options[1]:
    BarChartByTitle(title, batchsize_list, arch, figure_dir)

if __name__ == '__main__':
  main()
