#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse
from datetime import datetime
import mne
import math
import numpy as np
import spectral
import matplotlib.pyplot as plt
from multiprocessing import Process, Array

#Set constants
MAX_CONCURRENT_PROCESSES=5

debug_mode=False

#Processes command-line arguments
def process_arguments():
   parser = argparse.ArgumentParser(description='Computation of coherence with jiggling.')
   parser.add_argument('--output_data_dir',action='store',dest='output_data_dir',help='output data directory',default='');
   parser.add_argument('--input_data_dir',action='store',dest='input_data_dir',help='input data directory',default='');
   parser.add_argument('--input_data_file',action='store',dest='input_data_file',help='input data file',default='');
   parser.add_argument('--output_matrix_file',action='store',dest='output_matrix_file',help='file with computed coherence matrix',default='');
   parser.add_argument('--window_dur',action='store',dest='window_dur',help='window duration',default='500',type=int);
   parser.add_argument('--jiggling_length',action='store',dest='jiggling_length',help='jiggling length',default='100',type=int);
   parser.add_argument('--window_overlap',action='store',dest='window_overlap',help='window overlap',default='0',type=int);
   parser.add_argument('--tmp_data_dir',action='store',dest='tmp_data_dir',help='directory for temporary memory-mapped files',default='');
   parser.add_argument('--max_channels_to_process',action='store',dest='max_channels_to_process',help='maximum number of channels to process',type=int, default='-1');
   parser.add_argument('--plot_matrices_across_time',action='store_true',dest='plot_matrices_across_time',help='plot coherence matrices for each time window, if specified',default=False);
   parser.add_argument('--debug',action='store_true',dest='debug',help='output debugging info if specified',default=False);
   args=parser.parse_args()

   #Adds slashes to ends of arguments pertaining to directories
   if args.input_data_dir:
     	args.input_data_dir=args.input_data_dir+"/"
   if args.output_data_dir:
     	args.output_data_dir=args.output_data_dir+"/"
   if args.tmp_data_dir:
     	args.tmp_data_dir=args.tmp_data_dir+"/"
   #If the name of the input_data_file was not provided as a command-line parameter, raises Exception
   if not args.input_data_file:
      raise Exception ('input_data_file was not defined')
   return args

#wait for all worker processes to finish their work and exit
def wait_for_results(worker_processes):
    for p in worker_processes:
       p.join()
    return

#start a separate process to compute coherence for a subset of time windows, between  start_time and end_time, for all channels
def request_coherence_some_windows_all_channels(data,start_window,start_time, end_time, total_time, num_chans, window_dur, jiggling_length, window_loop_step, sampling_frequency, debug_mode, coh_arrays,worker_processes):
    if debug_mode:
         print("debug: before Process, max processes ", MAX_CONCURRENT_PROCESSES)
    p = Process(target=compute_coherence_some_windows_all_channels, args=(data,start_window,start_time, end_time, total_time, num_chans, window_dur, jiggling_length, window_loop_step, sampling_frequency, debug_mode, coh_arrays))
    p.start()
    worker_processes.append(p)
    return

#Computes coherence arrays for certain number of channels across certain time period (specified in method parameters)
def compute_coherence_some_windows_all_channels(data,start_window, start_time, end_time, total_time, num_chans, window_dur, jiggling_length, window_loop_step, sampling_frequency, debug_mode, coh_arrays):

   state_initialized_flag=False
   coherence_states={}
   window_index=start_window

   #Uses modified version of scipy coherence method (found in spectral.py) to compete coherence between each pair of channels
   for wind_start_time0 in range(start_time, end_time-window_dur+1 , window_loop_step): #iterate through all full time windows within given period of time
       coh_vals_currWind_allChans = []
       chan_pair_index=0
       for chan_ind0 in range(0,num_chans): #iterate through all channels
           coh_vals_currWind_currChan = []
           for chan_ind1 in range(0,num_chans): #iterate through all channels
                if chan_ind0 == chan_ind1:
                     coh_val_currPair = 1
                else:
                     coh_vals_currPair = []
                     range_start=max(wind_start_time0-jiggling_length, 0)
                     if (jiggling_length==0):
                         range_end=range_start+1
                         increment=1 
                     else:
                         range_end=min(wind_start_time0+window_dur+jiggling_length+1,total_time-window_dur+1)
                         increment=jiggling_length
                     for wind_start_time1 in range(range_start, range_end, increment): #iterate through each window of window duration msec, starting from jiggling_length msec before start of current window to jiggling_length msec after end of current window
                          chan_ind0_seg = data[chan_ind0][wind_start_time0:wind_start_time0+window_dur]
                          chan_ind1_seg = data[chan_ind1][wind_start_time1:wind_start_time1+window_dur]
                          #Uses modified version of scipy coherence method to compute coherence for current pair of channels
                          if window_dur >= 512:
                              nfft=256
                          else:
                              nfft=int(window_dur/2)
                          nperseg=nfft
                          if (state_initialized_flag==False):
                                 initial_coherence_state=spectral.compute_initial_coherence_state(chan_ind0_seg,nfft=nfft,nperseg=nperseg)
                                 state_initialized_flag=True
                          if coherence_states.get(wind_start_time1)==None:
                                 compute_coherence_states_for_time_window(coherence_states,data,wind_start_time1,window_dur,num_chans,sampling_frequency,initial_coherence_state)
                                 remove_old_coherence_states(coherence_states,wind_start_time1-2*window_dur)
                          if coherence_states.get(wind_start_time0)==None:
                                 compute_coherence_states_for_time_window(coherence_states,data,wind_start_time0,window_dur,num_chans,sampling_frequency,initial_coherence_state)
                                 remove_old_coherence_states(coherence_states,wind_start_time0-2*window_dur)
                          freqs,coh_winds_currPair_currWind1 = spectral.compute_coherence_from_states(get_coherence_state_for_channel(coherence_states,chan_ind0,wind_start_time0),get_coherence_state_for_channel(coherence_states,chan_ind1,wind_start_time1))
                          #compute mean of coherence vals returned by plt.cohere (which returns one coherence val per frequency
                          #this mean represents the coh val of the current 2 chan segments
                          coh_val_currPair_currWind1=np.mean(coh_winds_currPair_currWind1)
                          coh_vals_currPair.append(coh_val_currPair_currWind1) #add coh val of the current 2 chan segments to array
                     #find maximum coh val among the coh vals computed for each pair of chan segments for the current 2 chans
                     #set the coh value of the current pair of chans to this max
                     coh_val_currPair = max(coh_vals_currPair)
                coh_arrays[chan_pair_index][window_index]=coh_val_currPair
                chan_pair_index += 1
       window_index+=1

#compute per-channel coherence states for a given time window 
def compute_coherence_states_for_time_window(coherence_states,data,window_start_time,window_duration,num_chans,sampling_frequency,initial_coherence_state):
    coherence_state_for_time={}
    for chan in range (0,num_chans):
         chan_seg = data[chan][window_start_time:window_start_time+window_duration]
         coherence_state=spectral.compute_coherence_state_for_one_side(chan_seg,initial_coherence_state,fs=sampling_frequency)
         coherence_state_for_time[chan]=coherence_state
    coherence_states[window_start_time]=coherence_state_for_time

#remove per-channel coherence states for time windows all samples in which were fully processed, with jiggling taken into account  
def remove_old_coherence_states(coherence_states,highest_window_time_to_remove):
    keys=coherence_states.keys()
    keys_list=list(keys)
    for key in keys_list:
       if (key <= highest_window_time_to_remove):
          coherence_states.pop(key)

#return per-channel coherence state, for a given channel
def get_coherence_state_for_channel(coherence_states,channel,window_time):
    return coherence_states.get(window_time).get(channel)

#main method
if __name__ == '__main__':
   args=process_arguments()
   debug_mode=args.debug
   print ("start at ", datetime. now())

   #Get neural data
   filtered_data = mne.io.read_raw_fif(args.input_data_dir+args.input_data_file, preload=True)
   print ("done preloading raw data at ", datetime. now())
   tot_time = len(filtered_data);
   num_chans=len(filtered_data.ch_names)

   #If a valid number of channels to process was passed as a command-line parameter, the default max_channels_to_process is set to this value
   #Otherwise, max_channels_to_process retains its default value
   if (args.max_channels_to_process < num_chans and args.max_channels_to_process > 0):
        num_chans=args.max_channels_to_process

   window_dur=args.window_dur	
   jiggling_length=args.jiggling_length
   window_overlap=args.window_overlap
   window_loop_step=window_dur-window_overlap

   worker_processes=[]

   chan_pair_index=0

   sampling_frequency=filtered_data.info['sfreq']

   data=filtered_data.get_data()
   window_index=0
   coherence_states={}

   #Calculates number of windows per process (each child process will process the same number of windows)
   if tot_time < window_dur:
       num_windows=0
   else:
       num_windows=(tot_time-window_dur)//window_loop_step+1
   if num_windows == 0:
       tot_time=0
   else:
       tot_time=(num_windows-1)*window_loop_step+window_dur
   num_windows_per_process=num_windows//MAX_CONCURRENT_PROCESSES
   if (num_windows % MAX_CONCURRENT_PROCESSES !=0):
      num_windows_per_process += 1

   #prepare multiprocessing arrays that will contain the results
   coh_arrays = []
   for chan_ind0 in range(0,num_chans): #iterate through all channels
       for chan_ind1 in range(0,num_chans): #iterate through all channels
          coh_array = Array('d', num_windows , lock=False)
          coh_arrays.append(coh_array)

   #initiate multi-processing
   #Starts each child process (on certain number of windows (see above))
   window_counter=0
   prevProcessEndWindow=0
   prevProcessEndTime=0
   for wind_start_time0 in range(0, tot_time-window_dur+1 , window_loop_step): #iterate through all full time windows
      if window_counter==num_windows_per_process:
         processEndTime=wind_start_time0-window_loop_step+window_dur
         request_coherence_some_windows_all_channels(data,prevProcessEndWindow,prevProcessEndTime, processEndTime, tot_time, num_chans, window_dur, jiggling_length, window_loop_step, sampling_frequency, debug_mode, coh_arrays,worker_processes)
         prevProcessEndTime=wind_start_time0
         prevProcessEndWindow+=num_windows_per_process
         window_counter=0
      window_counter += 1
   request_coherence_some_windows_all_channels(data,prevProcessEndWindow,prevProcessEndTime, wind_start_time0+window_dur, tot_time, num_chans, window_dur, jiggling_length, window_loop_step, sampling_frequency, debug_mode, coh_arrays,worker_processes)

   wait_for_results(worker_processes)

   #fill numpy array with results
   coh_vals_allWinds_allChans = []
   window_index = 0
   for wind_start_time0 in range(0, tot_time-window_dur+1 , window_loop_step): #iterate through all full time windows
       coh_vals_currWind_allChans = []
       chan_pair_index = 0
       for chan_ind0 in range(0,num_chans): #iterate through all channels
           coh_vals_currWind_currChan = []
           for chan_ind1 in range(0,num_chans): #iterate through all channels
               coh_val_currPair = coh_arrays[chan_pair_index][window_index]
               if (coh_val_currPair==0):
                    print ("window_index=",window_index)
               chan_pair_index += 1
               coh_vals_currWind_currChan.append(coh_val_currPair)
           coh_vals_currWind_allChans.append(coh_vals_currWind_currChan)
       #this list (below) will be our final 3D matrix
       coh_vals_allWinds_allChans.append(coh_vals_currWind_allChans)
       window_index += 1

   #writes each numerical value in 3D array of matrices to file
   print ("done building matrix at ", datetime. now())
   if args.output_matrix_file:
      np.save(args.output_data_dir+args.output_matrix_file, coh_vals_allWinds_allChans)

   #print final 3D coh array
   #print(coh_vals_allWinds_allChans)
   print ("done printing matrix at ", datetime. now())

   if args.plot_matrices_across_time:
       #Plots 2D matrices across time
       #One fig is created for each window
       #Fig contains: 1) a colormap plot of the matrix in the window it corresponds to, and 2) a colorbar
       #Each fig is saved to its own file
       #A movie can be made from the consecutive fig images
       coh_allWinds_allChans =coh_vals_allWinds_allChans
       fig_num = 1
       for wind_mat_ind in range(num_windows):
           #Set title of curr fig
           title = str(wind_mat_ind*window_loop_step)+" to "+str(wind_mat_ind*window_loop_step+window_dur)+" ms"

           #Label axes of curr fig
           ax_labels = [str(0), str(int(num_chans/2)), str(num_chans)]

           wind_coh_mat = coh_allWinds_allChans[wind_mat_ind]
           coh_fig, ax2 = plt.subplots()
           coh_fig.suptitle(title)
           img = ax2.imshow(wind_coh_mat, cmap='viridis')

           y_lims = ax2.get_ylim()
           y_min = y_lims[0]
           y_max = y_lims[1]
           y_mid = (y_max+y_min)/2
           y_label_locs = np.asarray([y_min, y_mid, y_max])
           ax2.set_yticks(y_label_locs)
           ax2.set_yticklabels(ax_labels)

           x_lims = ax2.get_xlim()
           x_min = x_lims[0]
           x_max = x_lims[1]
           x_mid = (x_max+x_min)/2
           x_label_locs = np.asarray([x_min, x_mid, x_max])
           ax2.set_xticks(x_label_locs)
           ax2.set_xticklabels(ax_labels)

           #Create colorbar
           img.set_clim(0,1)
           cbi = coh_fig.colorbar(img, orientation='horizontal', shrink=0.8,extend='both',spacing='uniform')

           #Save fig in its own file
           coh_fig.savefig(args.output_data_dir+'coh_pic%04d.png'%fig_num)

           plt.close(coh_fig)

           fig_num += 1

   print ("end at ", datetime. now())
