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

metrics_trace_list = []

regex_kernel = re.compile(r"(?:void)?\s?(?:\w+(?:::))*(\w+)(<[^\(]+>)?\(?.*\)?")
regex_int = re.compile(r"^(?:[a-zA-z]+)*\s*\(*(\d+)\)*$")
regex_float = re.compile(r"(\d+\.\d+(?:e\+)*\d*)(\w+)?(?:\/\w+|%)*")

kernel_idx = 1
metric_name_idx = 3
metric_avg_value_idx = 7

db_path = os.getcwd()

def ObtainMetricTraceFiles(trace_dir):
  current_dir = os.getcwd()+'/'+trace_dir
  os.chdir(current_dir)
  for f in glob.glob("*-metrics.csv"):
      metrics_trace_list.append(f)

  if len(metrics_trace_list) == 0:
    print "No Metric trace file in current directory"
    exit()

#--batch size
#----layer
#------propagation
#--------kernel
#----------metric
metric_trace_dict = collections.OrderedDict()
def GenerateMetricDict(arch):
  # Number of useless lines
  num_useless_lines = 6
  for f in metrics_trace_list:
    #print "Process metric trace file: ", f
    reader = csv.reader(open(f, 'rb'))

    # The metric files has to be like <propagation>_<layer_name>_config_<batch_size>.csv
    kernel_name = f[0:-4].split('-')[0]

      
    batch_size = f[0:-4].split('-')[1]
    invocation_order = f[0:-4].split('-')[2]
    #print "  Kernel name:  ", kernel_name 
    #print "  Batch size: ", batch_size
    #print "  Invocation order: ", invocation_order
    if batch_size not in metric_trace_dict:
      metric_trace_dict[batch_size] = collections.OrderedDict()
    # Query the database
    conn = sqlite3.connect(db_path+'/alexnet_kernel.db')
    table_name = 'alexnet_kernel_dict'
    c = conn.cursor()

    for i in range(num_useless_lines):
      next(reader)
    for row in reader:
      # Get rid of other useless lines
      if len(row) < 8:
        continue
      if regex_kernel.match(row[kernel_idx]):
        content = row[kernel_idx]
        kernel_name_from_trace = regex_kernel.match(content).group(1)
        template_arg_from_trace = regex_kernel.match(content).group(2) if\
          regex_kernel.match(content).group(2) != None else ""

      # Check the consistency of kernel name
      if kernel_name != kernel_name_from_trace:
        print kernel_name, kernel_name_from_trace
        print "Kernel name not consistent!!!"
        exit()
      full_name = kernel_name + template_arg_from_trace
      query_layer_name_str = "SELECT layername FROM "+table_name+" WHERE batchsize = "+batch_size+" AND fullname = '"+full_name+"' AND invocationorder = "+invocation_order+" AND arch = '"+arch+"'"
      c.execute(query_layer_name_str)
      layer_name_from_sql = c.fetchall()
      #if len(layer_name_from_sql) > 1 or len(layer_name_from_sql) == 0:
      if len(layer_name_from_sql) > 1:
        print layer_name_from_sql
      if len(layer_name_from_sql) == 0:
        if "dgrad" not in kernel_name:
          print kernel_name
          print full_name
          print invocation_order
          print arch
          print "Illegal layers!!!"
          exit()
        else:
          continue
      layer_name = layer_name_from_sql[0][0].encode("utf-8")

      query_propagation_str = "SELECT propagation FROM "+table_name+" WHERE batchsize = "+batch_size+" AND fullname = '"+full_name+"' AND invocationorder = "+invocation_order+" AND arch = '"+arch+"'"
      c.execute(query_propagation_str)
      propagation_from_sql = c.fetchall()
      #if len(propagation_from_sql) > 1 or len(propagation_from_sql) == 0:
      if len(propagation_from_sql) > 1:
        print propagation_from_sql
      if len(propagation_from_sql) == 0:
        if "dgrad" not in kernel_name:
          print "Ilegal propagation!!!"
          exit()
        else:
          continue
      propagation = propagation_from_sql[0][0].encode("utf-8")

      if layer_name not in metric_trace_dict[batch_size]:
        metric_trace_dict[batch_size][layer_name] = collections.OrderedDict()
      if propagation not in metric_trace_dict[batch_size][layer_name]:
        metric_trace_dict[batch_size][layer_name][propagation] = collections.OrderedDict()

      if kernel_name not in metric_trace_dict[batch_size][layer_name][propagation]:
        metric_trace_dict[batch_size][layer_name][propagation][kernel_name] = collections.OrderedDict()
      # Obtain the metric value
      if regex_int.match(row[metric_avg_value_idx]):
        content = row[metric_avg_value_idx]
        value = int(regex_int.match(content).group(1))
      elif regex_float.match(row[metric_avg_value_idx]):
        content = row[metric_avg_value_idx]
        value = float(regex_float.match(content).group(1))
        if regex_float.match(content).group(2) == "MB":
          value /= float(1024)
      metric_name = row[metric_name_idx]
      if "utilization" in metric_name:
        value = value * 10
      metric_trace_dict[batch_size][layer_name][propagation][kernel_name][metric_name] = value

def BarChartByLayerName(title, layer_list, batch_size, metrics_list, stacked, arch, figure_dir):
  metrics_dict = collections.OrderedDict()
  xtick_name = collections.OrderedDict()
  count = collections.OrderedDict()
  for propagation in ["Forward", "Backward"]:
    if propagation not in metrics_dict:
      metrics_dict[propagation] = collections.OrderedDict()
      xtick_name[propagation] = []
      count[propagation] = 0
      for layer in layer_list:
        l_list = [layer]
        if "Backward" in propagation:
          if "conv" in layer:
            l_list = []
            l_list.append(layer+"_w")
            if "conv1" not in layer:
              l_list.append(layer+"_d")
          if "fc" in layer:
            l_list = []
            l_list.append(layer+"_w")
            l_list.append(layer+"_d")
        for l in l_list:
          if propagation in metric_trace_dict[batch_size][l]:
            for kernel in metric_trace_dict[batch_size][l][propagation]:
              count[propagation] += 1
              xtick_name[propagation].append(l)
              hit_rate = 0.0
              new_key = ""
              cumulative_value = 0.0
              for metric in metrics_list:
                if title == "CacheHitRate" and arch == "kepler":
                  new_key = "l1_hit_rate"
                  if new_key not in metrics_dict[propagation]:
                    metrics_dict[propagation][new_key] = []
                  if metric == "l1_cache_global_hit_rate" or metric == "l1_cache_local_hit_rate":
                    hit_rate += metric_trace_dict[batch_size][l][propagation][kernel][metric]
                    continue
                if title == "CacheHitRate" and arch == "pascal":
                  if metric == "l2_tex_read_hit_rate" or metric == "l2_tex_write_hit_rate":
                    new_key = "l2_hit_rate"
                    if new_key not in metrics_dict[propagation]:
                      metrics_dict[propagation][new_key] = []
                    hit_rate += metric_trace_dict[batch_size][l][propagation][kernel][metric]
                    continue
                key = metric
                if key not in metrics_dict[propagation]:
                  metrics_dict[propagation][key] = []
                if metric not in metric_trace_dict[batch_size][l][propagation][kernel]:
                  value = 0.0
                else:
                  value = metric_trace_dict[batch_size][l][propagation][kernel][metric]
                cumulative_value += value
                metrics_dict[propagation][key].append(value)
              if title == "CacheHitRate":
                metrics_dict[propagation][new_key].append(hit_rate)

  #print metrics_dict

  if title == "":
    exit()
  # Plot parameter
  opacity = 1.0
  bar_width = 1.
  #color_map = ['r', 'c', 'y', 'g', 'grey', 'b', 'm']
  #color_map = ['0.5', '1.0', '0.95', '0.95', '0.3', '0.35', '0.65', '0.75', '0.45', '0.5', '0.8']
  color_map = [
    '#283593',
    '#E8EAF6',
    '#C5CAE9',
    '#3949AB',
    '#1A237E',
    '#303F9F',
    '#9FA8DA',
    '#7986CB',
    '#5C6BC0',
    '#3F51B5',
  ]
  hatch_map = ['', '/', '', '/', '', '\\', '', '\\', '-', '+']
  for propagation in metrics_dict:
    n_groups = count[propagation]

    if stacked:
      num_metrics = 1
    else:
      num_metrics = len(metrics_dict[propagation])

    index = np.arange(n_groups) * 2 * num_metrics * bar_width

    i = 0
    matplotlib.rc('font', size=50)
    if "Stall" in title:
      fig = plt.figure(figsize=(50, 10))
    else:
      fig = plt.figure(figsize=(40, 10))
    ax = fig.add_subplot(111)
    bottom_list = [0] * n_groups
    for metric in metrics_dict[propagation]:
      if stacked:
        if i == 0:
          ax.bar(index, metrics_dict[propagation][metric], bar_width,
                 align='center',
                 alpha=opacity,
                 color=color_map[i],
                 hatch=hatch_map[i],
                 label=metric)       
          bottom_list = [sum(x) for x in zip(bottom_list, metrics_dict[propagation][metric])]
        else:
          ax.bar(index, metrics_dict[propagation][metric], bar_width,
                 bottom=bottom_list,
                 align='center',
                 alpha=opacity,
                 color=color_map[i],
                 hatch=hatch_map[i],
                 label=metric)       
          bottom_list = [sum(x) for x in zip(bottom_list, metrics_dict[propagation][metric])]
      else:
        ax.bar(index + i * bar_width, metrics_dict[propagation][metric], bar_width,
               align='center',
               alpha=opacity,
               color=color_map[i],
               hatch=hatch_map[i],
               label=metric)
      i += 1
 
    ax.set_xlabel('Layers')
    if title == "CacheHitRate":
      ax.set_ylabel('Cache Hit Rate')
    elif title == "MemTransaction1" or title == "MemTransaction2":
      ax.set_ylabel('Number of Memory Transactions')
    elif title == "MemThroughput1" or title == "MemThroughput2":
      ax.set_ylabel('Troughput(GB/s)')
    elif title == "MemUtilization":
      ax.set_ylabel('Memory Utilization(%)')
    elif title == "CuUtilization":
      ax.set_ylabel('Utilization of Computing Resources(%)')
    elif title == "Efficiency":
      ax.set_ylabel('Efficiency')
    elif title == "RegionalMemoryThroughput1":
      ax.set_ylabel('Throughput')
    elif title == "RegionalMemTransaction1":
      ax.set_ylabel('Number of Memory Transactions')
    elif title == "RegionalMemTransaction2":
      ax.set_ylabel('Number of Memory Transactions')
    elif title == "ReplayRate":
      ax.set_ylabel('Averaged Number of Inst Replays')
    elif title == "IPC":
      ax.set_ylabel('IPC')
    elif title == "StallReason":
      ax.set_ylabel('Percentage(%)')

    if "Stall" not in title:
      ax.yaxis.grid()
    else:
      ax.grid(False)
    ax.set_xticks(index + bar_width * (num_metrics-1) / 2)
    #ax.set_xticklabels(layer_list[len(layer_list)-len(metrics_dict[kernel][metric]):])
    #ax.set_xticklabels(xtick_name[propagation], rotation='45', ha='right')
    ax.set_xticklabels(xtick_name[propagation])
    #ax.legend(bbox_to_anchor=(1.05, 1), loc=2,
    #           ncol=1, borderaxespad=0.)
    if "Stall" not in title:
      ax.legend(bbox_to_anchor=(0.25, 1.02, 0.75, .102), loc=3,
                 ncol=2, mode="expand", borderaxespad=0.)
    else:
      ax.legend(bbox_to_anchor=(0, 1.02, 1., .102), loc=3,
                 ncol=4, mode="expand", borderaxespad=0.)   
    ax.set_xlim(min(index)-bar_width, max(index)+bar_width*num_metrics)

    if "Utilization" in title or "Stall" in title:
      ax.set_ylim([0,100])
    #ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    fig.tight_layout()
    fig.savefig(os.getcwd()+'/figures/'+title+"_"+propagation+"_"+batch_size+'.pdf', format='pdf', bbox_inches='tight')

def OutputMetrics2CSV(layer_list, batch_size, metrics_list, arch):
  for metric in metrics_list:
    f = arch+"-"+metric+"-"+batch_size+".csv"
    writer = csv.writer(open(f, 'wb'))
    for propagation in ["Forward", "Backward"]:
      if propagation == "Backward":
        layers = layer_list[::-1]
      else:
        layers = layer_list
      for layer in layers:
        l_list = [layer]
        if "Backward" in propagation:
          if "conv" in layer:
            l_list = []
            l_list.append(layer+"_w")
            if "conv1" not in layer:
              l_list.append(layer+"_d")
          if "fc" in layer:
            l_list = []
            l_list.append(layer+"_w")
            l_list.append(layer+"_d")
        for l in l_list:
          row = []
          if propagation in metric_trace_dict[batch_size][l]:
            for kernel in metric_trace_dict[batch_size][l][propagation]:
              if metric not in metric_trace_dict[batch_size][l][propagation][kernel]:
                value = 0.0
              else:
                value = metric_trace_dict[batch_size][l][propagation][kernel][metric]
              if propagation == "Forward":
                row.append(l+"_fwd")
              else:
                row.append(l+"_bwd")
              row.append(value)
              writer.writerow(row)
                
    
  
usage = "usage: %prog [option1] arg1,arg2 [option2] arg1,arg2"
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
  parser.add_option("-l1", "--layer1", type="string", dest="layername1",
                    help="Layer names seperated by coma", metavar="LAYER")
  parser.add_option("-l2", "--layer2", type="string", dest="layername2",
                    help="Layer names seperated by coma", metavar="LAYER")
  parser.add_option("-k", "--kernels", type="string", dest="kernels",
                    help="Kernels name seperated by coma", metavar="KERNELS")
  parser.add_option("-m", "--metrics", type="string", dest="metrics",
                    help="Metrics name seperated by coma", metavar="METRICS")
  parser.add_option("-t", "--title", type="string", dest="title",
                    help="Plot title", metavar="TITLE")

  (options, args) = parser.parse_args()
  if len(args) > 0:
    print "Arguments parsing fail. Possible reason: space occurred between arguments"
    exit()

  layer_list_1 = []
  layer_list_2 = []
  batchsize_list = []
  kernels_list = []
  metrics_list = []
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

  if options.layername1 != None:
    layer_list1 = options.layername1.split(",")
  if options.layername2 != None:
    layer_list2 = options.layername2.split(",")
  if options.batchsize != None:
    batchsize_list = options.batchsize.split(",")
  if options.kernels != None:
    kernels_list = options.kernels.split(",")
  if options.metrics != None:
    metrics_list = options.metrics.split(",")
  if options.title != None:
    title = options.title

  ObtainMetricTraceFiles(trace_dir)

  GenerateMetricDict(arch)
  if len(metric_trace_dict) == 0:
    print "NO trace found!!!"
    exit()

  is_kernel_list_empty = len(kernels_list) == 0
  for b_size in batchsize_list:
    if b_size not in metric_trace_dict:
      print " Batch size INVALID!!!"
      exit()
    print "For batch size: ", b_size
    for layer in layer_list1:
      if layer not in metric_trace_dict[b_size]:
        print "Layer name INVALID!!!"
        print layer
        exit()
      print "In layer: ", layer
      for propagation in ["Forward", "Backward"]:
        l_list = [layer]
        if "Backward" in propagation:
          if "conv" in layer:
            l_list = []
            l_list.append(layer+"_w")
            if "conv1" not in layer:
              l_list.append(layer+"_d")
          if "fc" in layer:
            l_list = []
            l_list.append(layer+"_w")
            l_list.append(layer+"_d")       
        print "Propagation: ", propagation
        for l in l_list:
          if is_kernel_list_empty:
            kernels_list = metric_trace_dict[b_size][l][propagation]
          for kernel in kernels_list:
            if kernel not in metric_trace_dict[b_size][l][propagation]:
              print "Kernel name INVALID!!!", kernel
              exit()
            print "   In Kernel: ", kernel
            for metric in metrics_list:
              if metric not in metric_trace_dict[b_size][l][propagation][kernel]:
                print "Metric name: ", metric, " INVALID!!!"
                exit()
              value = metric_trace_dict[b_size][l][propagation][kernel][metric]
              print "     Metric: ", metric," ", value 

  # Plot
  for batch_size in batchsize_list:
    if "Stall" in title:
      BarChartByLayerName(title, layer_list1, batch_size, metrics_list, True, arch, figure_dir)
    else:
      BarChartByLayerName(title, layer_list1, batch_size, metrics_list, False, arch, figure_dir)
  # Output metrics to CSV
  OutputMetrics2CSV(layer_list2, "128", metrics_list, arch)

if __name__ == '__main__':
  main()
