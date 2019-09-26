#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import time
import argparse
from datetime import datetime
#from scipy import signal
import mne
import math
import numpy as np
import spectral
import matplotlib.pyplot as plt
from multiprocessing import Process, Array

WINDOW_DUR = 500 # ms
JIGGLING_LENGTH = 100 # ms
WINDOW_OVERLAP = 0
LOOP_STEP = WINDOW_DUR-WINDOW_OVERLAP

MAX_CONCURRENT_PROCESSES=5
debug_mode=False

#INPUT_DATA_FILE="data_meg_mag_notch_filtered_JV.fif"

def process_arguments():
   parser = argparse.ArgumentParser(description='Computation of coherence with jiggling.')
   parser.add_argument('--output_data_dir',action='store',dest='output_data_dir',help='output data directory',default='');
   parser.add_argument('--input_data_dir',action='store',dest='input_data_dir',help='input data directory',default='');
   parser.add_argument('--input_data_file',action='store',dest='input_data_file',help='input data file',default='');
   parser.add_argument('--output_matrix_file',action='store',dest='output_matrix_file',help='file with computed coherence matrix',default='');
   parser.add_argument('--tmp_data_dir',action='store',dest='tmp_data_dir',help='directory for temporary memory-mapped files',default='');
   parser.add_argument('--max_channels_to_process',action='store',dest='max_channels_to_process',help='maximum number of channels to process',type=int, default='-1');
   parser.add_argument('--debug',action='store_true',dest='debug',help='output debugging info if specified');
   args=parser.parse_args()	
   if args.input_data_dir:
     	args.input_data_dir=args.input_data_dir+"/"
   if args.output_data_dir:
     	args.output_data_dir=args.output_data_dir+"/"
   if args.tmp_data_dir:
     	args.tmp_data_dir=args.tmp_data_dir+"/"
   if not args.input_data_file:
      raise Exception ('input_data_file was not defined')
   return args	
	
def wait_for_results(worker_processes):
    for p in worker_processes:
       p.join()
    return

def request_coherence_some_windows_all_channels(data,start_time, end_time, total_time, num_chans, window_dur, jiggling_length, sampling_frequency, debug_mode, coh_arrays,worker_processes):
    if debug_mode:
         print("debug: before Process, max processes ", MAX_CONCURRENT_PROCESSES)
    p = Process(target=compute_coherence_some_windows_all_channels, args=(data,start_time, end_time, total_time, num_chans, window_dur, jiggling_length, sampling_frequency, debug_mode, coh_arrays))
    p.start()
    worker_processes.append(p)
    return

def compute_coherence_some_windows_all_channels(data,start_time, end_time, total_time, num_chans, window_dur, jiggling_length, sampling_frequency, debug_mode, coh_arrays):

   state_initialized_flag=False
   max_time_with_preprocessed_coherence=-1	
   coherence_states={}
   window_index=start_time//window_dur

   for windStartTime0 in range(start_time, end_time-window_dur+1 , window_dur): #iterate through all full time windows
       coh_vals_currWind_allChans = []
       chan_pair_index=0	
       for chanInd0 in range(0,num_chans): #iterate through all channels
           coh_vals_currWind_currChan = []
           for chanInd1 in range(0,num_chans): #iterate through all channels
                if chanInd0 == chanInd1: 
                     coh_val_currPair = 1
                else:
                     coh_vals_currPair = []
                     for windStartTime1 in range(max(windStartTime0-jiggling_length, 0), min(windStartTime0+window_dur+jiggling_length,total_time-window_dur+1), jiggling_length): #iterate through each window of window duration msec, starting from jiggling_length msec before start of current window to jiggling_length msec after end of current window
                          chanInd0_seg = data[chanInd0][windStartTime0:windStartTime0+window_dur]
                          chanInd1_seg = data[chanInd1][windStartTime1:windStartTime1+window_dur]
                          if (state_initialized_flag==False):
                                 initial_coherence_state=spectral.compute_initial_coherence_state(chanInd0_seg,nfft=250,nperseg=250)
                                 state_initialized_flag=True
                          if (windStartTime1 > max_time_with_preprocessed_coherence):
                                 compute_coherence_states_for_time_window(coherence_states,data,windStartTime1,window_dur,num_chans,sampling_frequency,initial_coherence_state)
                                 max_time_with_preprocessed_coherence=windStartTime1
                                 remove_old_coherence_states(coherence_states,windStartTime1-2*window_dur)
                          if (windStartTime0 > max_time_with_preprocessed_coherence):
                                 compute_coherence_states_for_time_window(coherence_states,data,windStartTime0,window_dur,num_chans,sampling_frequency,initial_coherence_state)
                                 max_time_with_preprocessed_coherence=windStartTime0
                                 remove_old_coherence_states(coherence_states,windStartTime0-2*window_dur)
                          freqs,coh_winds_currPair_currWind1 = spectral.compute_coherence_from_states(get_coherence_state_for_channel(coherence_states,chanInd0,windStartTime0),get_coherence_state_for_channel(coherence_states,chanInd1,windStartTime1))
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

def compute_coherence_states_for_time_window(coherence_states,data,window_start_time,window_duration,num_chans,sampling_frequency,initial_coherence_state):
    coherence_state_for_time={}
    for chan in range (0,num_chans):
         chan_seg = data[chan][window_start_time:window_start_time+window_duration]
         coherence_state=spectral.compute_coherence_state_for_one_side(chan_seg,initial_coherence_state,fs=sampling_frequency)
         coherence_state_for_time[chan]=coherence_state
    coherence_states[window_start_time]=coherence_state_for_time
  
def remove_old_coherence_states(coherence_states,highest_window_time_to_remove):
    keys=coherence_states.keys()
    keys_list=list(keys)
    for key in keys_list:
       if (key <= highest_window_time_to_remove):
          coherence_states.pop(key)

def get_coherence_state_for_channel(coherence_states,channel,window_time):
    return coherence_states.get(window_time).get(channel)

if __name__ == '__main__': 
   args=process_arguments()

   debug_mode=args.debug
   
   print ("start at ", datetime. now())
   #Get neural data
#   filtered_data = mne.io.read_raw_fif(args.input_data_dir+args.input_data_file, preload=args.tmp_data_dir+"mmap_file.tmp")
   filtered_data = mne.io.read_raw_fif(args.input_data_dir+args.input_data_file, preload=True)
   print ("done preloading raw data at ", datetime. now())
   tot_time = len(filtered_data);
   num_chans=len(filtered_data.ch_names)
   if (args.max_channels_to_process < num_chans and args.max_channels_to_process > 0):
        num_chans=args.max_channels_to_process


   worker_processes=[]

   chan_pair_index=0	
	
   sampling_frequency=filtered_data.info['sfreq']

   data=filtered_data.get_data()	
   window_index=0
   coherence_states={}

   num_windows=tot_time//WINDOW_DUR
   tot_time=num_windows*WINDOW_DUR
   num_windows_per_process=num_windows//MAX_CONCURRENT_PROCESSES	
   if (num_windows % MAX_CONCURRENT_PROCESSES !=0):
      num_windows_per_process += 1

   #prepare multiprocessing arrays that will contain the results	
   coh_arrays = []
   for chanInd0 in range(0,num_chans): #iterate through all channels
       for chanInd1 in range(0,num_chans): #iterate through all channels
          coh_array = Array('d', num_windows , lock=False)  
          coh_arrays.append(coh_array)

   #initiate multi-processing
   window_counter=0
   prevWindStartTime=0
   for windStartTime0 in range(0, tot_time , WINDOW_DUR): #iterate through all full time windows
      if window_counter==num_windows_per_process:
         request_coherence_some_windows_all_channels(data,prevWindStartTime, windStartTime0, tot_time, num_chans, WINDOW_DUR, JIGGLING_LENGTH, sampling_frequency, debug_mode, coh_arrays,worker_processes)
         prevWindStartTime=windStartTime0
         window_counter=0
      window_counter += 1
   request_coherence_some_windows_all_channels(data,prevWindStartTime, windStartTime0+WINDOW_DUR, tot_time, num_chans, WINDOW_DUR, JIGGLING_LENGTH, sampling_frequency, debug_mode, coh_arrays,worker_processes)

   wait_for_results(worker_processes)

   #fill numpy array with results
   coh_vals_allWinds_allChans = []
   window_index = 0
   for windStartTime0 in range(0, tot_time-WINDOW_DUR+1 , WINDOW_DUR): #iterate through all full time windows
       coh_vals_currWind_allChans = [] 
       chan_pair_index = 0
       for chanInd0 in range(0,num_chans): #iterate through all channels
           coh_vals_currWind_currChan = []
           for chanInd1 in range(0,num_chans): #iterate through all channels
               coh_val_currPair = coh_arrays[chan_pair_index][window_index]   
               chan_pair_index += 1
               coh_vals_currWind_currChan.append(coh_val_currPair)
           coh_vals_currWind_allChans.append(coh_vals_currWind_currChan)
       #this list (below) will be our final 3D matrix
       coh_vals_allWinds_allChans.append(coh_vals_currWind_allChans)
       window_index += 1
 

   print ("done building matrix at ", datetime. now())
   if args.output_matrix_file:
      f=open(args.output_data_dir+args.output_matrix_file,"w+")
      f.write(str(num_chans)+"\n")
      for window in coh_vals_allWinds_allChans:
         for cohMat in window:
            for cohVal in cohMat:
                f.write(str(cohVal)+"\n")
      f.close()

   #print final 3D matrix
   print(coh_vals_allWinds_allChans)
   print ("done printing matrix at ", datetime. now())

   #Plots 3D matrices across time
   #One fig. is created for each window
   #Fig contains colormap made from made from the matrix in the window it corresponds to, as well as colorbar
   #Each fig is saved to its own file
   #A movie can be made from the consecutive fig images
   coh_allWinds_allChans =coh_vals_allWinds_allChans 
   num_winds = int(tot_time/WINDOW_DUR)
   figNum = 1
   for wind_mat_ind in range(num_winds):
       #Set title of curr fig
       title = str(wind_mat_ind*LOOP_STEP)+" to "+str(wind_mat_ind*LOOP_STEP+WINDOW_DUR)+" ms"

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
       CBI = coh_fig.colorbar(img, orientation='horizontal', shrink=0.8,extend='both',spacing='uniform')

       #Save fig in its own file
       coh_fig.savefig(args.output_data_dir+'coh_pic%04d.png'%figNum)

       plt.close(coh_fig)

       figNum += 1


   print ("end at ", datetime. now())

