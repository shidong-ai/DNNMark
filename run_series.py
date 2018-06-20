#!/usr/bin/env python

# Prepares and runs multiple tasks on multiple GPUs: one task per GPU.
# Waits if no GPUs available. For GPU availability check uses "nvidia-smi dmon" command.

# 2018 (C) Peter Bryzgalov @ CHITECH Stair Lab

import multigpuexec
import time
import os

gpus = range(0,1)
runs = 1
batchsizes = range(10,410,10)
# List of convolutional layer configurations
conv_sizes = [32, 64, 128, 256]
epochs = 1
tasks = []
logdir = "logs/dnnmark_timings_and_profiles/"
logfile_base="dnn_bwd_conv"
command = "./run_dnnmark.sh"
if not os.path.exists(logdir):
    os.makedirs(logdir)
print "Logdir",logdir


for batch in batchsizes:
    for conv in conv_sizes:
        logname = "{}_conv{}_bs{}".format(logfile_base,conv,batch)
        logfile = os.path.join(logdir,"{}.log".format(logname))
        command_pars = command+" -n {} -k {} --debug".format(batch,conv)
        if os.path.isfile(logfile):
            print "file",logfile,"exists."
        else:
            task = {"comm":command_pars,"logfile":logfile,"batch":batch,"conv":conv,"nvsmi":True}
            tasks.append(task)
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


