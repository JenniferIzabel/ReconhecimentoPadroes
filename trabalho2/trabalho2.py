#!/usr/bin/env python
# coding: utf-8

# # Trabalho 2
# 

# Objetivo: ALPHA

# In[1]:


import random

import glob
from subprocess import getoutput as gop

#from urllib.request import urlopen, urlretrieve
import os

from re import search
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense

from sklearn.metrics import confusion_matrix, accuracy_score

from mne import set_eeg_reference as car
import mne


# In[ ]:




