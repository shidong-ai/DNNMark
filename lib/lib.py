import subprocess, os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import random


# Read minibatch size and 1st epoch time from files.
# Store in a DataFrame.
def fileToDF(logfile_path, pars, debug=False):
    filename_pattern = None # Get columns values from filename

    if "columns" in pars:
        filename_pattern = pars["filename_pattern"]
        fix_columns = pars["columns"]

    var_columns =  ["time"]
    output_pattern = pars["output_pattern"]
    
    remove_str = None
    if "remov_str" in pars:
        remove_str = pars["remove_str"]


    logfile_path = logfile_path.strip(" \n")
    logfile = os.path.basename(logfile_path)
    if debug:
        print "Reading",logfile
        print "columns=",fix_columns + var_columns
    with open(logfile_path,"r") as f:
        fix_values = []
        ind = 0 # DataFrame row numebr (index)

        if filename_pattern is not None:
            ms = filename_pattern.match(logfile)
            if ms:
                for i in range(len(fix_columns)):
                    fix_values.append(ms.group(i+1))
                if debug:
                    print "Parsed file name to:",fix_values
            else:
                print logfile,"didnt match pattern",filename_pattern.pattern

        df = pd.DataFrame(data=None,columns=fix_columns + var_columns)
        time = 0
        ind = 0 # DataFrame row numebr (index)
        row = []
        lines = f.readlines()
        for line in lines:
            s = line.strip(' \n')
            if remove_str:
                s = s.replace(remove_str,"")
            m2 = output_pattern.search(s)
            if m2:
                time = float(m2.group(1))
                if debug:
                    print "\"{}\" found in \"{}\"".format(output_pattern.pattern,s)
                    print time
                row = fix_values + [time]
                if debug: print "Appending row:",row
                df.loc[ind] = row
                ind += 1
                continue

    if debug:
        print df.head()
    return df


# Read file logs from logdir directory
def readLogs(logdir, pars, debug=False):
    filename_pattern = pars["filename_pattern"]

    list_command = "ls -1 "+logdir
    if debug: print "Looking in",logdir
    files=[]
    proc = subprocess.Popen(list_command.split(" "),
                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    for line in iter(proc.stdout.readline,b''):
        line = line.strip(" \n")
        m = filename_pattern.match(line)
        if m:
            files.append(os.path.abspath(os.path.join(logdir,line)))

    if debug:
        print len(files),"files"
        print files

    df = None

    for file in files:
        df1 = fileToDF(file, pars=pars, debug=debug)
        if len(df1) > 0:
            if df is None:
                df = df1
            else:
                df = pd.concat([df,df1],ignore_index=True)
    return df


# Read file logs from logdir directory
# Chainer log files from AWS has extra garbage to be removed
def readLogsAWS(logdir, pars, debug=False):
    filename_pattern = pars["filename_pattern"] # Log files file names pattern
    batch_learn_pattern = pars["batch_learn_pattern"] # BS and LR read from file pattern
    output_pattern = pars["output_pattern"] # Read Chainer output pattern
    remove_str = pars["remove_str"]  # Remove strings list for cleaning output lines before parsing
    list_command = "ls -1 "+logdir
    files=[]
    proc = subprocess.Popen(list_command.split(" "),
                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if debug: maxfiles = 5
    else: maxfiles = 100000000
    for line in iter(proc.stdout.readline,b''):
        line = line.strip(" \n")
        m = filename_pattern.match(line)
        if m:
            files.append(os.path.abspath(os.path.join(logdir,line)))

    #if debug: print "files:",files
    df = pd.DataFrame(data=None,columns=["batch","learn","epoch","time"])

    filecounter=0
    for file in files:
        if debug: print file
        df1 = fileToDF_AWS(file,batch_learn_pattern,output_pattern,remove_str, debug)
        if len(df1) > 0:
            df = pd.concat([df,df1],ignore_index=True)
        else:
            print "No data from",file
        filecounter +=1
        if filecounter>=maxfiles:
            return df
    return df



# Read minibatch size and 1st epoch time from files.
# Store in a DataFrame.
def fileToDF_AWS(logfile, batch_learn_pattern,output_pattern,remove_str, debug=False):
    logfile = logfile.strip(" \n")
    filename = os.path.basename(logfile)
    if debug: print "FILE",filename
    batch = 0
    learn = 0
    m = re.search(batch_learn_pattern, filename)
    if m:
        batch = int(m.group(1))
        learn = float(m.group(2))
        if debug: print "BS,LR:",batch,learn

    with open(logfile,"r") as f:
        lines = f.readlines()
        time = 0
        epoch = 0
        ind = 0 # DataFrame row numebr (index)
        df = pd.DataFrame(data=None,columns=["batch","learn","epoch","time"])
        for line in lines:
            s = line.strip(' \n')
            for rmstr in remove_str:
                s = s.replace(rmstr,"")
            m2 = output_pattern.match(s)
            if m2:
                if debug: print s,
                epoch = int(m2.group(1))
                time = float(m2.group(6))
                if debug: print "BS,LR,epoch,time:",batch,learn,epoch,time
                df.loc[ind] = [batch,learn,epoch,time]
                ind += 1

    return df


# Returns replication count as a function of x "importance" - smaller values more important
# used in replicating important training samples.
# ratio - number of replications for most important samples,
# degree - "stepness" of the function curve
def getMultiplier(x,xmax,xmin,ratio=5,degree=2):
    x = x-xmin
    mxx = xmax - xmin
    y = (mxx-x)/(mxx+x)
    y = (y**degree)*(ratio-1) +1
    return np.round(y).astype(int)


# Return list of n indexes uniformly spreaded in 0-l range
def pickSampleIndexes(l,n):
    x = []
    step = float(l)/float(n)
    for i in range(l):
        pos = int(round(step*i))
        if pos < l:
            x.append(pos)
    return x


# Replicate samples proportionally to their inverted value (time):
# Samples with small values get replicated more.
def Stratify(idx, df, time_min,time_max, ratio=5,degree=2):
    newlist=[]
    for i in idx:
        time = df.iloc[i]["time"]
        koeff = getMultiplier(time,time_max,time_min,ratio=ratio,degree=degree).astype(int)
        newlist.append(i)
        # Insert value i koeff-1 times
        if koeff > 1:
            if df.iloc[i]["GPU"] == "K80":
                print time,"(s) koeff=",koeff
            l = [i]*(koeff-1)
            newlist = newlist + l
    return newlist


# Pick equally spaced N samples from Dataframe df where "GPU" column is GPU
def pickSamplesForGPU(df,GPU,trainN,testN,stratify=False):
    # Use equally spaced samples for training set
    df_tmp = df[df["GPU"]==GPU]
    l = len(df_tmp.index)
    idx_train = pickSampleIndexes(l, trainN)
    # idx_train is a list of positions in df subset (rows for specific GPU model)
    # Exclude training set positions from list of row positions in df subset
    invert_list =  [i for i in range(l) if i not in idx_train]
    #print "inverted list size:",len(invert_list)
    if len(invert_list) > testN:
        # Pick testN samples from positions list without rtaining samples randomly
        idx = np.random.choice(len(invert_list),testN,replace=False)
        # Convert a list of positions to a list of indexes in df subset
        idx_test = [invert_list[i] for i in idx]
    else:
        idx_test = invert_list

    # Stratification: replicate samples with lower values (times)
    print GPU,
    if stratify:
        ratio=int(stratify[0])
        degree=int(stratify[1])
        print "before",len(idx_train),
        time_max = df["time"].max()
        time_min = df["time"].min()
        idx_train = Stratify(idx_train, df_tmp, time_min,time_max,ratio=ratio,degree=degree)
        random.shuffle(idx_train)
        print "after",len(idx_train)
    print len(idx_train),"/",len(idx_test)

    samples_df = df_tmp.sort_values(by=["batch"])
    train_df = samples_df.iloc[idx_train]
    test_df  = samples_df.iloc[idx_test]
    #print "return:",train_df.shape,test_df.shape
    return (train_df,test_df)

# Returns to DataFrames: with training samples and test samples
def makeTrainingTestDFs(df, n, trainN, testN, stratify=False):
    GPUs = df["GPU"].unique()
    df_train = None
    df_test  = None
    for GPU in GPUs:
        train_1, test_1 = pickSamplesForGPU(df, GPU, trainN/n, testN/n,stratify=stratify)
        if df_train is None:
            df_train = train_1
        else:
            df_train = pd.merge(df_train,train_1,how="outer")

        if df_test is None:
            df_test = test_1
        else:
            df_test = pd.merge(df_test,test_1,how="outer")
    return (df_train, df_test)


# Plot two plots with training samples and test samples
def plotTrainTestSamples(Xtrain, Ytrain, Xtest, Ytest):
    f, axarr = plt.subplots(1,2, sharex=True,figsize=(12,3))
    sc0 = axarr[0].scatter(x=Xtrain["batch"].values,y=Ytrain.values,s=2,alpha=0.1)
    sc1 = axarr[1].scatter(x=Xtest["batch"].values,y=Ytest.values,s=2,alpha=.3)
    axarr[0].set_title("training set")
    axarr[1].set_title("test set")
    axarr[0].grid(ls=":",alpha=0.1)
    axarr[1].grid(ls=":",alpha=0.1)
    plt.show()


# Returns Percentage Error
def PercentageError(h,y):
    h = np.array(h)
    y = np.array(y)
    err = np.mean(np.abs(y - h) / y * 100)
    return err


# Plot prediction line
# df - Dataframe with ALL samples
# idx - indexes of samples from the test set
def plotPredictions1(model,df,df_test,title,features):
    no_batch_features = features[1:]
    #no_batch_features.remove("batch")
    df_tmp = pd.DataFrame(columns=features)
    pad = 15
    bmin = df_test["batch"].min() - pad
    bmax = df_test["batch"].max() + pad
    x_ = np.arange(bmin,bmax,5)
    architectures = df_test["CUDA cap"].unique()
    architectures = sorted(architectures, key=str,reverse=True)
    #height = len(architectures) * 3
    fig,ax = plt.subplots(len(architectures),1,sharex=True,figsize=(9,9))
    ax[0].set_title(title)
    for i in range(len(architectures)):
        CUDA_cap = architectures[i]
        GPU = df[df["CUDA cap"]==CUDA_cap]["GPU"].iloc[0]
        add = df[df["CUDA cap"]==CUDA_cap][no_batch_features].iloc[0].values
        for j in range(len(x_)):
            df_tmp.loc[j] = np.insert(add,0,x_[j])
        y_ = model.predict(df_tmp)
#         x_ = df_test[df_test["GPU"]==GPU]["batch"].values
#         y_ = model.predict(df_test[df_test["GPU"]==GPU][features].values)
        ax[i].plot(x_,y_,c="r",label="prediction "+GPU)

        # Plot test samples
        Xc = df_test[df_test["GPU"] == GPU][features].values
        Yc = df_test[df_test["GPU"] == GPU]["time"].values
        Htest = model.predict(Xc)
        X = df_test[df_test["GPU"] == GPU]["batch"]
        ax[i].scatter(X, Yc,s=1,alpha=.5,label="test samples")
        MSE = "MSE={:.5f}".format(mean_squared_error(Yc, Htest))
        MPE = "MPE={:.5f}".format(PercentageError(Yc, Htest))
        #print text
        ax[i].set_ylabel("time (s)")
        ax[i].grid(ls=":",alpha=0.3)
        ax[i].legend()
        ax[i].text(1.01,0.9,MSE,transform=ax[i].transAxes,size=12)
        ax[i].text(1.01,0.75,MPE,transform=ax[i].transAxes,size=12)
    ax[-1].set_xlabel("batch size")
    fig.show()


