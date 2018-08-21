#!/usr/bin/env python

# Prepares and runs multiple tasks on multiple GPUs: one task per GPU.
# Waits if no GPUs available. For GPU availability check uses "nvidia-smi dmon" command.

# 2018 (C) Peter Bryzgalov @ CHITECH Stair Lab

import multigpuexec
import time
import os
import math

gpus = range(0,1)
batchsizes = range(10,30,10)
datasetsize = 50000
# List of convolutional layer configurations
conv_sizes = [512]
backfilterconvalgos=[0,1,3]
imsizes = [32]
nvprof = False
tasks = []
logdir = "logs/dnnmark_Mouse_gpu_traces_composed_model_backfilterconvalgo_mod2/"

command = "./run_dnnmark_template.sh"
if not os.path.exists(logdir):
    os.makedirs(logdir)
print "Logdir",logdir
runs = 5

logfile_base="dnn_convolution"
for imsize in imsizes:
    for batch in batchsizes:
        iterations = int(math.ceil(datasetsize/batch))
        print "BS: {}, Iterations: {}".format(batch,iterations)
        for conv in conv_sizes:
            for algo in backfilterconvalgos:
                for run in range(runs):
                    logname = "{}_imsize{}_conv{}_bs{}_algo{}".format(logfile_base,imsize,conv,batch,algo)
                    logfile = os.path.join(logdir,"{}_{}.log".format(logname,run))
                    command_pars = command+" -n {} -k {} -w {} -h {} --algo {} --debug --iter {}".format(batch,conv,imsize,imsize,algo,iterations)
                    if os.path.isfile(logfile):
                        print "file",logfile,"exists."
                    else:
                        task = {"comm":command_pars,"logfile":logfile,"batch":batch,"conv":conv,"nvsmi":True}
                        tasks.append(task)
                if nvprof:
                    logfile = os.path.join(logdir,"{}_%p.nvprof".format(logname))
                    if os.path.isfile(logfile):
                        print "file",logfile,"exists."
                    else:
                        profcommand = "nvprof -u s --profile-api-trace none --unified-memory-profiling off --profile-child-processes --csv --log-file {} {}".format(logfile,command_pars)
                        task = {"comm":profcommand,"logfile":logfile,"batch":batch,"conv":conv,"nvsmi":False}
                        tasks.append(task)

print "Have",len(tasks),"tasks"
gpu = -1
for i in range(0,len(tasks)):
    gpu = multigpuexec.getNextFreeGPU(gpus, start=gpu+1,c=4,d=1,nvsmi=tasks[i]["nvsmi"])
    gpu_info = multigpuexec.getGPUinfo(gpu)
    f = open(tasks[i]["logfile"],"w+")
    f.write("b{} conv{}\n".format(tasks[i]["batch"],tasks[i]["conv"]))
    f.write("GPU: {}\n".format(gpu_info))
    f.close()
    multigpuexec.runTask(tasks[i],gpu,nvsmi=tasks[i]["nvsmi"],delay=0)
    print "{}/{} tasks".format(i+1,len(tasks))
    time.sleep(1)


