# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:59:36 2022

@author: Yoren Mo
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import warnings 
from pathlib import Path
import scipy.io as sio
import streamlit as st
import pandas as pd
import torch 
import numpy as np
#from utilsResFNO import ResFNO, FNO1d, RangeNormalizer,Predict,T_random
#from utils import chart
import random
import os
# from utils.Shapley import Shapley
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import chart
from utils import chart_shap
import altair as alt
import utilsAV.streamlit_av as F2
import utilsAV.streamlit_low_av as F3


directory = './Result'

if 1:
    Result = sio.loadmat(directory+'/plot_shapley_results.mat')
    plot_points0 = Result['plot_points'][0]
    RH_scores = Result['RH_scores'][0]
    RL_scores = Result['RL_scores'][0]
    AH_scores = Result['AH_scores'][0]
    AL_scores = Result['AL_scores'][0]
    
    ny = len(plot_points0)   
    list_RH   = ['RH' for i in range(ny)]
    list_RL   = ['RL' for i in range(ny)]
    
    Shapley_list = np.linspace(5,580,116)
    Shapley_value    = np.hstack((RH_scores, RL_scores))
    Shapley_Method   = np.hstack((list_RH, list_RL))
    Shapley_num      = np.hstack((Shapley_list, Shapley_list))
    
    
    
    performance_remove_shap = pd.DataFrame({
        'num'        : Shapley_num, 
        'method'     : Shapley_Method,
        'value'      : Shapley_value})
    
    
    chart_shap = chart_shap.get_chart_shap(performance_remove_shap)
    st.altair_chart(chart_shap, use_container_width=True)