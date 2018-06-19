#!/usr/bin/env python

# Library of plotting functions

# 2018 (C) Peter Bryzgalov @ CHITECH Stair Lab

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def plotHeatMap(df,title=None,cmap=None,ax=None,zrange=None, format=".3f"):
    if ax is None:
        fig,ax = plt.subplots()
    if cmap is None:
        cmap = "viridis"
    if zrange is None:
        cmesh=ax.pcolormesh(df,cmap=cmap)
    else:
        cmesh=ax.pcolormesh(df,cmap=cmap,vmin=zrange[0],vmax=zrange[1])

    ax.set_yticks(np.arange(0.5, len(df.index), 1))
    ax.set_yticklabels(df.index)
    ax.set_xticks(np.arange(0.5, len(df.columns), 1))
    ax.set_xticklabels(df.columns)
    ax.tick_params(direction='in', length=0, pad=10)
    for y in range(df.shape[0]):
        for x in range(df.shape[1]):
            #if df.iloc[y,x]  0:
            ax.text(x+0.5,y+0.5,'{0:{fmt}}'.format(df.iloc[y,x],fmt=format),
                     color="black",fontsize=9,
                     horizontalalignment='center',
                     verticalalignment='center',
                     bbox={'facecolor':'white','edgecolor':'none', 'alpha':0.2, 'pad':0})
    ax.set_title(title,fontsize=16)
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    cbar = plt.colorbar(cmesh, ax=ax, pad =0.02)
    cbar.ax.tick_params(direction='out', length=3, pad=5)
    return (ax,cbar)


def rotateXticks(ax, angle):
    for tick in ax.get_xticklabels():
        tick.set_rotation(angle)


def rotateYticks(ax, angle):
    for tick in ax.get_yticklabels():
        tick.set_rotation(angle)

def getColorList(cmap,n):
    cmap = cm.get_cmap(cmap, n)
    colors = []
    for i in range(cmap.N):
        c = matplotlib.colors.to_hex(cmap(i),keep_alpha=True)
        colors.append(c)
    return colors


def testColorMap(cmap):
    x = np.arange(0, np.pi, 0.1)
    y = np.arange(0, 2*np.pi, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.log((X+Y+1)*(X+Y+2))

    plt.rcParams['figure.figsize'] = 5,2
    fig, ax = plt.subplots()
    im = ax.imshow(Z, interpolation='nearest', origin='lower', cmap=cmap)
    ax.set_title("N bins: %s" % 256)
    fig.colorbar(im, ax=ax)
    plt.show()