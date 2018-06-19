# MutliGPU series execution support
# 2018 (C) Peter Bryzgalov @ CHITECH Stair Lab

from __future__ import print_function
import subprocess
import re
import time


def message(s,col=112):
    print("\033[38;5;{}m{}\033[0m".format(col,s))

# Returns True if GPU #i is not used.
# Uses nvidia-smi command to monitor GPU SM usage.
def GPUisFree(i,c=4,d=1,mode="dmon",debug=False):
    if mode=="dmon":
        # Doesn't work in (AWS) VMs
        command = "nvidia-smi dmon -c {} -d {} -i {} -s u".format(c,d,i)
        if debug:
            print(command)
        out_pattern = re.compile(r"^\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)") # dmon
    else:
        # Doesn't work in docker containers
        # Doesn't work when GPU card used by X Window server
        command = "nvidia-smi pmon -c {} -d {} -i {} -s u".format(c,d,i)
        out_pattern = re.compile(r"^\s+(\d+)\s+([0-9\-]+)\s+([CG\-])\s+([0-9\-]+)\s") # pmon
    proc = subprocess.Popen(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, shell=False)
    u = 0
    for line in iter(proc.stdout.readline, b''):
        line = line.decode('utf-8')
        if debug:
            print(line)
        m = out_pattern.search(line)
        if m:
            print(".",end="")
            uplus = 0
            try:
                uplus = int(m.group(2))
            except ValueError:
                pass
            u += uplus
    if u < 1:
        return True
    return False


# Returns GPU info
def getGPUinfo(i,query="name,memory.total"):
    command = "nvidia-smi -i {} --query-gpu={} --format=csv,noheader".format(i,query)
    proc = subprocess.Popen(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, shell=False)
    output = ""
    for line in iter(proc.stdout.readline, b''):
        line = line.decode('utf-8')
        output += line
    return output


# Returns number of a free GPU.
# gpus  -- GPU number or list of numbers.
# start -- number of GPU to start with.
def getNextFreeGPU(gpus,start=-1,c=4,d=1,nvsmi=False,mode="dmon",debug=False):
    if not isinstance(gpus,list):
        gpus = [gpus]
    if start > gpus[-1]:
        # Rewind to GPU 0
        start = 0
    while True:
        for i in range(0,len(gpus)):
            gpu = gpus[i]
            if gpu < start:
                continue
            print("checking GPU",gpu,end="")
            if GPUisFree(gpu,c=c,d=d,mode=mode,debug=debug):
                return gpu
            print("busy")
            time.sleep(3)
            start = -1 # Next loop check from 1


# Runs a task on specified GPU
def runTaskContainer(task,gpu,verbose=False):
    f = open(task["logfile"],"ab")
    #f.write("gpu"+str(gpu)+"\n")
    command = task["comm"]
    #command = "python --version"
    # IMPORTANT: remote double spaces or they will become empty arguments!
    command = re.sub(' \s+',' ',command).strip()
    command = "NV_GPU="+str(gpu)+" ./run_container.sh "+command
    print("Starting ",command)
    if not verbose:
        pid = subprocess.Popen(command, stdout=f, stderr=f, bufsize=1, shell=True).pid
        print(pid)
    else:
        p = subprocess.Popen(command,stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, shell=True)
        for line in iter(p.stdout.readline,''):
            print(line.rstrip())
            f.write(line)
    f.close()

# Runs a task on specified GPU
def runTask(task,gpu,nvsmi=False,delay=10):
    with open(task["logfile"],"ab") as f:
        #f.write("gpu"+str(gpu)+"\n")
        command = task["comm"]
        # IMPORTANT: remove double spaces or they will become empty arguments!
        command = re.sub(' \s+',' ',command).strip()
        print("Starting")
        message(command)
        pid = subprocess.Popen(command.split(" "),stdout=f,stderr=subprocess.STDOUT,bufsize=1).pid
        print(pid)

    if (nvsmi):
        # Save memory usage info
        # Wait before process starts using GPU
        sampling_rate = 5 #msec
        sampling_period = 2 #sec
        time.sleep(delay)
        # Save stdout to logfile
        logfile = task["logfile"]+".nvsmi"
        fl = open(logfile,"w")
        command="nvidia-smi -i {} -lms {} --query-gpu=timestamp,name,memory.total,memory.used --format=csv,noheader,nounits".format(gpu,sampling_rate)
        p = subprocess.Popen(command.split(" "),stdout=fl, stderr=subprocess.STDOUT, bufsize=1, shell=False)
        time.sleep(sampling_period)
        p.kill()
        fl.close()
