import os
import time
import argparse
from datetime import datetime
from scipy import signal
from multiprocessing import Process, Array
import mne
import math
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SEIZ_START_TIME=694000
SEIZ_END_TIME=738000

#Processes command-line arguments
def process_arguments():
   parser = argparse.ArgumentParser(description='Parameters of plot coherence')
   parser.add_argument('--factor_config_dir',action='store',dest='factor_config_dir',help='factor_config_dir',default='')
   parser.add_argument('--factor_config_file',action='store',dest='factor_config_file',help='factor_config_file',default='')
   parser.add_argument('--input_data_dir',action='store',dest='input_data_dir',help='input_data_dir',default='')
   parser.add_argument('--training_output_dir',action='store',dest='training_output_dir',help='training_output_dir',default='')
   parser.add_argument('--training_output_file',action='store',dest='training_output_file',help='training_output_file',default='')
   parser.add_argument('--num_winds_to_aggregate',action='store',dest='num_winds_to_aggregate',help='num_winds_to_aggregate',default='')
   parser.add_argument('--wind_dur',action='store',dest='wind_dur',help='wind_dur',default='')
   parser.add_argument('--look_ahead_msec',action='store',dest='look_ahead_msec',help='look_ahead_msec',default='30000')

   args=parser.parse_args()
   return args

#Computes the average of all of the values within the specified array
def compute_ave_across_time(factor_arr):
    return(np.mean(factor_arr))

#Finds the range of all of the values within the specified array
def find_range(factor_arr):
    #print(factor_arr.shape)
    #print(factor_arr)
    max=np.amax(factor_arr)
    min=np.amin(factor_arr)
    return(max-min)

#Computes the standard deviation of all of the values within the specified array
def compute_std_dev_across_time(factor_arr):
    return(np.std(factor_arr))

#Main method
if __name__ == '__main__':
    args=process_arguments()
    if args.factor_config_dir:
        args.factor_config_dir+="/"
    if args.input_data_dir:
        args.input_data_dir+="/"

    #Open and read file containing information (factor name, method to apply to factor, file containing data for factor) for each factor
    f=open(args.factor_config_dir+args.factor_config_file,"r")
    lines=f.readlines()
    num_info_lines_per_factor=3
    num_factors=len(lines)//num_info_lines_per_factor
    factors=[]
    methods=[]
    files=[]
    #Store information about each factor in specified in the config file in three arrays
    for lineInd in range(0,len(lines),num_info_lines_per_factor):
        factors.append(lines[lineInd][:-1])
        methods.append(lines[lineInd+1][:-1])
        files.append(lines[lineInd+2][:-1])

    #Create "y" training data, an array that
    #contains a "1" in each spot that corresponds to a window of which the middle is within 30 seconds before any point in the seizure and
    #contains a "0" in all other spots
    y_train=[]
    factor_arr=np.load(args.input_data_dir+files[0])
    num_winds=len(factor_arr)
    for start_wind_ind in range(0, num_winds-int(args.num_winds_to_aggregate)):
        end_wind_ind=start_wind_ind+int(args.num_winds_to_aggregate)
        mid_wind_ind=(start_wind_ind+end_wind_ind)//2
        if ((mid_wind_ind*int(args.wind_dur))+int(args.look_ahead_msec)>=SEIZ_START_TIME and (mid_wind_ind*int(args.wind_dur))+int(args.look_ahead_msec)<=SEIZ_END_TIME):
            y_train.append(1)
        else:
            y_train.append(0)

    #Create "X" training data
    #For each factor specified in the config file (as well as the factors array),
    #the data for the factor is extracted from the corresponding file of which the name is contained in the files array
    #and the corresponding method of which the name is contained in the methods array is applied to the data across time (for each subset of windows)
    #Each of the outputs produced by the method across time is stored in an array
    #This array is added to features, a 2D array
    features=[]
    for factor_ind in range(num_factors):
        X_train=[]
        factor_arr=np.load(args.input_data_dir+files[factor_ind])
        for start_wind_ind in range(0, num_winds-int(args.num_winds_to_aggregate)):
            end_wind_ind=start_wind_ind+int(args.num_winds_to_aggregate)
            temp_factor_arr=factor_arr[start_wind_ind:end_wind_ind]

            if methods[factor_ind]=="compute_ave_across_time":
                X_train.append(compute_ave_across_time(temp_factor_arr))

            elif methods[factor_ind]=="find_range":
                X_train.append(find_range(temp_factor_arr))

            else:
                X_train.append(compute_std_dev_across_time(temp_factor_arr))
        features.append(X_train)

    #Begins to apply machine learning
    #Trains machine by applying LinearDiscriminantAnalysis().fit() on X training data and y training data
    clf = LinearDiscriminantAnalysis()
    features=list(zip(*features)) #swap dimentions of features
    clf.fit(features, y_train)
    LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None, solver='svd', store_covariance=False, tol=0.0001)
 solver='svd', store_covariance=False, tol=0.0001)
