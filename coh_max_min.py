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
   parser = argparse.ArgumentParser(description='Parameters of coh_max_min')
   parser.add_argument('--input_data_dir',action='store',dest='input_data_dir',help='input data directory',default='')
   parser.add_argument('--input_data_file_name',action='store',dest='input_data_file',help='input data file',default='')
   parser.add_argument('--output_data_dir',action='store',dest='output_data_dir',help='output_data_dir',default='')
   parser.add_argument('--output_data_file_prefix',action='store',dest='output_data_file_prefix',help='output data file',default='')
   parser.add_argument('--num_chans',action='store',dest='num_chans',help='number of channels',default=102)

   args=parser.parse_args()
   return args

if __name__ == '__main__':
    args=process_arguments()
    if args.input_data_dir:
        args.input_data_dir+="/"
    if args.output_data_dir:
        args.output_data_dir+="/"
    if args.num_chans:
        num_chans=int(args.num_chans)

    window_duration=500
    seiz_start_wind=int(SEIZ_START_TIME/window_duration)
    seiz_end_wind=int(SEIZ_END_TIME/window_duration)+1

    preictal_start_msec=40000
    preictal_start_wind=int((SEIZ_START_TIME-preictal_start_msec)/window_duration)
    postictla_end_msec=40000
    postictal_end_wind=int((SEIZ_END_TIME+postictla_end_msec)/window_duration)

    for chan0_ind in range(0,num_chans):
        good_mins=[]
        good_maxs=[]
        coh_arr_curr_chan=np.load(args.input_data_dir+args.input_data_file+str(chan0_ind)+".npy")
        #print(coh_arr_curr_chan)
        #swapping time and channel, channel becomes index 0
        coh_arr_curr_chan=np.swapaxes(coh_arr_curr_chan,0,1)
        #swapping time and frequency, frequency becomes index 1
        coh_arr_curr_chan=np.swapaxes(coh_arr_curr_chan,1,2)
        print(coh_arr_curr_chan.shape)
        num_windows=coh_arr_curr_chan.shape[2]
        tot_time_msec=int(num_windows*window_duration)
        minute_max_distr=[0] * (int(tot_time_msec/60000)+1)
        minute_min_distr=[0] * (int(tot_time_msec/60000)+1)
        num_freqs=coh_arr_curr_chan.shape[1]
        for chan1_ind in range(0,num_chans):
            for freq_ind in range(0,num_freqs):
                ind_max_coh_curr_combo=np.argmax(coh_arr_curr_chan[chan1_ind][freq_ind])
                minute_max_distr[int(ind_max_coh_curr_combo*window_duration/60000)]+=1
                ind_min_coh_curr_combo=np.argmin(coh_arr_curr_chan[chan1_ind][freq_ind])
                minute_min_distr[int(ind_min_coh_curr_combo*window_duration/60000)]+=1
                if (ind_min_coh_curr_combo>=seiz_end_wind and ind_min_coh_curr_combo<=postictal_end_wind):
                    good_mins.append([chan0_ind,chan1_ind,freq_ind,int(ind_min_coh_curr_combo*window_duration)])
                if (ind_max_coh_curr_combo>=preictal_start_wind and ind_max_coh_curr_combo<=seiz_start_wind):
                    #if (ind_min_coh_curr_combo>=seiz_end_wind and ind_max_coh_curr_combo<=postictal_end_wind):
                    good_maxs.append([chan0_ind,chan1_ind,freq_ind,int(ind_max_coh_curr_combo*window_duration)])
        print("channel=",chan0_ind," num_interesting_max=",len(good_maxs),"minute_max_distr=",minute_max_distr)
        print("channel=",chan0_ind," num_interesting_min=",len(good_mins),"minute_min_distr=",minute_min_distr)

        if args.output_data_file_prefix:
             np.save(args.output_data_dir+args.output_data_file_prefix+"ch"+str(chan0_ind)+".npy", minute_max_distr)
