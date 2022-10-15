# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:49:01 2022

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
#%%定义页面格式和标题
st.set_page_config(layout="centered", page_icon="💬", page_title="Commenting app")

# Data visualisation part
#%%--------步骤1：计算Shapley值
st.title("💬 Sampling via aggregation value ")

st.subheader('Step1 : Evaluation of model training based on Shapley value')
directory = './Result'
Result = sio.loadmat(directory+'/plot_shapley_results.mat')

if 1:
    
    col1, col2 = st.columns(2)        
    plot_points0 = Result['plot_points'][0]
    RH_scores = Result['RH_scores'][0]
    RL_scores = Result['RL_scores'][0]
    AH_scores = Result['AH_scores'][0]
    AL_scores = Result['AL_scores'][0]
    
    ny = len(plot_points0)   
    list_RH   = ['H' for i in range(ny)]
    list_RL   = ['L' for i in range(ny)]
    
    Shapley_list = np.linspace(5,580,116)
    Shapley_value_R    = np.hstack((RH_scores, RL_scores))
    Shapley_Method   = np.hstack((list_RH, list_RL))
    Shapley_num      = np.hstack((Shapley_list, Shapley_list))
    
    
    
    performance_remove_shap = pd.DataFrame({
        'num'        : Shapley_num, 
        'method'     : Shapley_Method,
        'value'      : Shapley_value_R})
    
    
    chart_shap_R = chart_shap.get_chart_shap_R(performance_remove_shap)
    
        
    Shapley_value_A    = np.hstack((AH_scores, AL_scores))  
    performance_add_shap = pd.DataFrame({
        'num'        : Shapley_num, 
        'method'     : Shapley_Method,
        'value'      : Shapley_value_A})
    
    
    chart_shap_A = chart_shap.get_chart_shap_A(performance_add_shap)
    
    
    with col1:
        st.altair_chart(chart_shap_R, use_container_width=True)

    with col2:
        st.altair_chart(chart_shap_A, use_container_width=True)
    
#%%--------步骤2：基于聚合价值采样
st.subheader('Step2 : Model training evaluation based on aggregate value')
#%%改变高聚合价值中的α

agree = st.checkbox('Change the α of HighAV')
if agree:
    result_directory = './Result'
    task = 'Regression'
    basemodel = 'GP' 
    t = st.select_slider(
    'Select α of HighAV',
    options=[0.01, 0.1, 1, 10, 100, 1000],value=100)
    st.write('α is', t)
    
    F2.evaluate_results(t,
                        basemodel, 
                        number_initial_points = 5, 
                        num_plot_markers = 300, 
                        num_interval_points = 3, 
                        directory = result_directory, 
                        task=task)
#%%改变低聚合价值中的α
if 0:
    agree = st.checkbox('Change the α of LowAV')
    if agree:
        result_directory = './Result'
        task = 'Regression'
        basemodel = 'GP' 
        t = st.select_slider(
        'Select α of LowAV',
        options=[0.01, 0.1, 1, 10, 100, 1000],value=100)
        st.write('α is', t)
        
        F3.evaluate_results(t,
                            basemodel, 
                            number_initial_points = 5, 
                            num_plot_markers = 300, 
                            num_interval_points = 3, 
                            directory = result_directory, 
                            task=task)
    
#%%导入已存数据

warnings.simplefilter("ignore")

result_directory = './Result'
Title = 'Results on the Composite task'

plot_points    = sio.loadmat(result_directory +'/plot_points.mat')['plot_points'][0]
HighSV_scores  = sio.loadmat(result_directory +'/HighSV_scores.mat')['HighSV_scores']
Cluster_scores = sio.loadmat(result_directory +'/Cluster_scores.mat')['Cluster_scores']
HighAV_scores  = sio.loadmat(result_directory +'/HighAV_scores.mat')['HighAV_scores']
LowAV_scores   = sio.loadmat(result_directory +'/LowAV_scores.mat')['LowAV_scores']
Random_scores  = sio.loadmat(result_directory +'/Random_scores.mat')['Random_scores']
AL_scores = Result['AL_scores'][0]
#%%airalt画图

if 1:    
    nx = len(plot_points)   
    list_HighSV   = ['HighSV' for i in range(nx)] 
    list_HighAV   = ['HighAV'  for i in range(nx)]
    
    num_list = np.linspace(5,302,100)
    
    data_value      = np.hstack((HighSV_scores, HighAV_scores)).reshape(-1,)
    sampling_Method = np.hstack((list_HighSV, list_HighAV))
    sampling_num    = np.hstack((num_list, num_list))
    
    performance_index_av = pd.DataFrame({
        'num'        : sampling_num, 
        'method'     : sampling_Method,
        'value'      : data_value})
  
    chart_av = chart.get_chart_av(performance_index_av)
    st.altair_chart(chart_av, use_container_width=True)
    
#%%--------步骤3：各种采样方法对比
st.subheader('Step3 : Comparison of sampling results of different methods')
#%%画图
if 0:
    plt.figure(figsize=(6,4.3))  
    size = 18
    fontfamily = 'arial'
    font = {'family':fontfamily,
            'size': 14,
            'weight':25}
    
    ax = plt.subplot()
    plt.subplots_adjust(left=0.15, right=0.98, bottom=0.15,top=0.9,wspace = 0.2, hspace = 0.05)
    
    plt.plot(plot_points.reshape(-1), np.mean(Random_scores, 0).reshape(-1), 
              '#757575', lw=2, zorder=9, label='Random')
    plt.fill_between(plot_points.reshape(-1), 
                    np.min(Random_scores, 0).reshape(-1), 
                    np.max(Random_scores, 0).reshape(-1), 
                    color='gray', alpha=0.3) 

    plt.plot(plot_points.reshape(-1), np.mean(Cluster_scores, 0).reshape(-1), 
             '#81c784', lw=2, zorder=9, label='Cluster')
   
    plt.plot(plot_points.reshape(-1), np.mean(HighSV_scores, 0).reshape(-1), 
             '#039be5', lw=2, zorder=9, label='HighSV') 

    plt.plot(plot_points.reshape(-1), np.mean(LowAV_scores, 0).reshape(-1), 
              '#6b4e9b', lw=2, zorder=9, label='LowAV')
      
    plt.plot(plot_points.reshape(-1), np.mean(HighAV_scores, 0).reshape(-1), 
             '#f44336', lw=2, zorder=9, label='HighAV')
    
    ax.set_xlabel('Number of samples', fontproperties = fontfamily, size = size)
    ax.set_ylabel('MAE (K) ',fontproperties = fontfamily, size = size)
    ax.set_yscale('log')
    ax.set_yticks([3, 5, 10, 20, 40])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    plt.yticks(fontproperties = fontfamily, size = size) 
    plt.xticks(fontproperties = fontfamily, size = size) 
    plt.ylim( (2,40) )
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.minorticks_on()
    plt.tick_params(which='major',length=7,width=2)
    
    plt.axhspan(5, 100, facecolor='#f9f9f9', alpha=0.5)
    plt.axhspan(0, 5, facecolor='#b3e5fc', alpha=0.5)
    #plt.axvline(x=32,linestyle='--',color='#607d8b')
    
    # plt.title(Title)
    plt.legend(prop = font,framealpha=0.8,loc='upper right')
    # plt.ylim(2,30)
    plt.xlim(4,228)
    plt.grid(linestyle='-.',axis="y")
    plt.show()
    plt.savefig(result_directory + '/figs/Fig-'+ str(Title) +'.png',dpi=600,bbox_inches='tight')
    plt.savefig(result_directory + '/figs/Fig-'+ str(Title) +'.svg',format='svg',bbox_inches='tight')
    plt.savefig(result_directory + '/figs/Fig-'+ str(Title) +'.pdf',format='pdf',bbox_inches='tight')
    st.write(plt)
#%%airalt画图
if 1:    
    nx = len(plot_points)   
    list_HighSV   = ['HighSV' for i in range(nx)]
    list_Cluster  = ['Cluster'  for i in range(nx)]
    list_HighAV   = ['HighAV'  for i in range(nx)]
    list_LowAV    = ['LowAV'  for i in range(nx)]
    list_Random   = ['Random'  for i in range(nx)]
    
    i = st.slider(
        "Let's pick a random sample!", min_value=0, max_value=5, step=1, value=5
    )
    
    num_list = np.linspace(5,302,100)
    random_scores = Random_scores[i].reshape(1,-1)
    data_value    = np.hstack((HighSV_scores, Cluster_scores, HighAV_scores, LowAV_scores, random_scores)).reshape(-1,)
    sampling_Method = np.hstack((list_HighSV, list_Cluster, list_HighAV, list_LowAV, list_Random))
    sampling_num         = np.hstack((num_list, num_list, num_list, num_list, num_list))
    
    performance_index = pd.DataFrame({
        'num'        : sampling_num, 
        'method'     : sampling_Method,
        'value'      : data_value})
  
    chart = chart.get_chart(performance_index)
    st.altair_chart(chart, use_container_width=True)
