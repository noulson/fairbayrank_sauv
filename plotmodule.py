# -*- coding: utf-8 -*-

import os
import pprint
import tempfile

from typing import Dict, Text
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

#plotting and saving function

def plotline_save_as_pdf(title='',axis_x = None, xlabel = '', axis_y1 = None, axis_y2 = None , axis_y3 = None, ylabel ='', linewidth=3, plot_file = '',legendy1='',legendy2='',legendy3=''):
    if axis_y1 is not None:
        plt.plot(axis_x, axis_y1, color='orange', linewidth = linewidth, label = legendy1)
    if axis_y2 is not None:
        plt.plot(axis_x, axis_y2, 'g', linewidth = linewidth, label = legendy2)
    if axis_y3 is not None:
        plt.plot(axis_x, axis_y3, 'b', linewidth = linewidth, label = legendy3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()

    plt.savefig(plot_file)

def savepdf_barplot_color(title='',axis_x = None, xlabel = '', axis_y1 = None, axis_y2 = None, axis_y3 = None, ylabel ='', barwidth=0.4, plot_file = '',legendy1 = '',legendy2 = '',legendy3 = ''):
      
    X_axis = np.arange(len(axis_x))
    plt.bar(2*X_axis - barwidth, axis_y1, width = barwidth, color="orange", label = legendy1)
    plt.bar(2*X_axis + 0, axis_y2, width = barwidth, color="g",label = legendy2)
    plt.bar(2*X_axis + barwidth, axis_y3, width = barwidth, color="b",label = legendy3)
    
    plt.xticks(2*X_axis, axis_x)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.savefig(plot_file)