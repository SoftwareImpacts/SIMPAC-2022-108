# -*- coding: utf-8 -*-
"""
Created on Sat May 12 21:56:22 2018
@author: Steven shangzhou Wang

Utility Functions:
    
"""

import os
import glob2
# import cv2
import datetime
import numpy as np
import pandas as pd
import time
import warnings
#import pickle
#import gzip
import matplotlib.pyplot as plt
from matplotlib import gridspec
# from keras.utils import np_utils
# import imutils
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings("ignore")



## set trading label

def index_to_label(label_ind):
    
    trading_index_dict = dict({0: 'No CD', 1: '1 CD', 2: '>=2 CDs'})
    trading_label = [trading_index_dict[l] for l in label_ind]

    return trading_label

def label_to_index(label_vec):
    
    trading_label_dict = dict({"No CD":0, "1 CD":1, ">= CDs":2})
    
    trading_ind = [trading_label_dict[l] for l in label_vec]
    
    return trading_ind



def plot_confusion_matrix(CM, labels, title, image_name = None, fig_size = (8, 8)):
    
    ## shorten the label for plotting purpose
    label = [l for l in labels]
    norm_CM = []
    for i in CM:
        a = 0
        norm_row = []
        a = sum(i, 0)
        for j in i:
            if a > 0:
                norm_row.append(float(j)/float(a))
            else:
                norm_row.append(0)
        norm_CM.append(norm_row)
    
    fig = plt.figure(figsize = fig_size)
    

    plt.clf()
    plt.xticks(rotation=90)
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.set_xticks([i for i in range(len(label))])
    ax.set_xticklabels(label)
    ax.set_yticks([i for i in range(len(label))])
    ax.set_yticklabels(label)
    ax.set_ylabel('------Actual------>')
    ax.set_xlabel('------Predicted------>')
    ax.set_title(title,fontdict={'fontsize':14})
    res = ax.imshow(np.array(norm_CM), cmap=plt.cm.jet, interpolation='nearest')

    w, h = pd.DataFrame(CM).shape
    
    for x in range(w):
        for y in range(h):
            ax.annotate(str(int(CM[x][y])), 
                        xy=(y, x), 
                        color = 'w',
                        size = "medium",
                        weight = "bold",
                        horizontalalignment='center',
                        verticalalignment='center')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size = "5%", pad = 0.2)
#    _ = fig.colorbar(res, cax = cax)
    plt.colorbar(res, cax = cax)
    plt.tight_layout()
    if image_name is not None:
        plt.savefig(image_name, dpi = 600)
    else:
        plt.show()
        
        
        
  