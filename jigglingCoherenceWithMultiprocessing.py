#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

WINDOW_DUR = 500 # ms
JIGGLING_LENGTH = 100 # ms
WINDOW_OVERLAP = 0
LOOP_STEP = WINDOW_DUR-WINDOW_OVERLAP



MAX_CONCURRENT_PROCESSES=8
debug_mode=False

#INPUT_DATA_FILE="data_meg_mag_notch_filtered_JV.fif"

def process_arguments():
   parser = argparse.ArgumentParser(description='Computation of coherence with jiggling.')
   parser.add_argument('--output_data_dir',action='store',dest='output_data_dir',help='output data directory',default='');
   parser.add_argument('--input_data_dir',action='store',dest='input_data_dir',help='input data directory',default='');
   parser.add_argument('--input_data_file',action='store',dest='input_data_file',help='input data file',default='');
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

def compute_coh_vals_someWinds_currChanPair(chanInd0_data,chanInd1_data,start_time,end_time,start_window_index,window_dur,jiggling_length,sampling_frequency,debug_mode,coh_array):
    tot_time=len(chanInd1_data)
    window_index=start_window_index

    if debug_mode==True:
         print("debug: in compute tot_time ",tot_time," window_index ",window_index," jiggling_length ",jiggling_length," window_dur ",window_dur," start_time ",start_time," end_time ",end_time)

    for windStartTime0 in range(start_time, end_time , window_dur): #iterate through all time window
        coh_vals_currPair = []
        for windStartTime1 in range(max(windStartTime0-jiggling_length, 0), min(windStartTime0+window_dur+jiggling_length,tot_time), jiggling_length): #iterate through each 5 sec window starting from 1000 ms before start of current window of 1st chan to 1000 ms after end of current window of first chan
            chanInd0_seg = chanInd0_data[windStartTime0:windStartTime0+window_dur]
            chanInd1_seg = chanInd1_data[windStartTime1:windStartTime1+window_dur]
            if (len(chanInd0_seg)!=len(chanInd1_seg)):
                 continue
            if debug_mode==True:
                 print("debug: before cohere")
#            coh_winds_currPair_currWind1, freqs = plt.cohere(chanInd0_seg, chanInd1_seg, Fs=sampling_frequency,NFFT=250 ) #call plt.cohere on current window/segment of 1st chan and on current window/segment of 2nd chan
            freqs,coh_winds_currPair_currWind1 = signal.coherence(chanInd0_seg, chanInd1_seg, fs=sampling_frequency,nfft=250,nperseg=250)
            if debug_mode==True:
                print("debug: after cohere num_freqs=",len(freqs))
            #compute mean of coherence vals returned by plt.cohere (which returns one coherence val per frequency
            #this mean represents the coh val of the current 2 chan segments
            mean_of_windMeans_currPair_currWind1 = 0
            for coh_wind in coh_winds_currPair_currWind1:
               mean_of_windMeans_currPair_currWind1 += np.mean(coh_wind)
            coh_val_currPair_currWind1 = mean_of_windMeans_currPair_currWind1/len(coh_winds_currPair_currWind1)
            coh_vals_currPair.append(coh_val_currPair_currWind1) #add coh val of the current 2 chan segments to array
        coh_array[window_index]=max(coh_vals_currPair)
        if debug_mode==True:
            print("debug: after windStartTime0=",windStartTime0)
        window_index+=1
    return

def request_coh_vals_someWinds_currChanPair(chanInd0_data,chanInd1_data,start_time,end_time,start_window_index,window_dur,jiggling_length,sampling_frequency,coh_array,worker_processes):
    if __name__ == '__main__':
        if (len(worker_processes) >= MAX_CONCURRENT_PROCESSES):
            worker_processes[0].join()
            worker_processes.pop(0)
        if debug_mode==True:
             print("debug: before Process, max processes ",MAX_CONCURRENT_PROCESSES)
        p=Process(target=compute_coh_vals_someWinds_currChanPair, args=(chanInd0_data, chanInd1_data,start_time,end_time,start_window_index,window_dur,jiggling_length,sampling_frequency,debug_mode,coh_array))
        p.start()
        worker_processes.append(p)
    return



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
   array_size=tot_time//WINDOW_DUR
   if (tot_time % WINDOW_DUR !=0):
      array_size+=1

   coh_arrays=[]
   chan_pair_index=0

   sampling_frequency=filtered_data.info['sfreq']

   min_chanInd0=0
   max_chanInd0=num_chans-1
   min_chanInd1=0
   max_chanInd1=num_chans-1

   for chanInd0 in range(min_chanInd0,max_chanInd0+1): #iterate through all channels
      chanInd0_picks=[chanInd0]
      chanInd0_data=filtered_data.get_data(picks=chanInd0_picks)
      for chanInd1 in range(min_chanInd1,max_chanInd1+1): #iterate through all channels
         if chanInd0 == chanInd1:
            coh_array= Array('d',array_size,lock=False)
            for i in range (0,array_size):
                coh_array[i]=1
         else:
            chanInd1_picks=[chanInd1]
            chanInd1_data=filtered_data.get_data(picks=chanInd1_picks)
            coh_array= Array('d',array_size,lock=False)
            max_windows_to_process=1000
            for windStartTime in range(0, tot_time , WINDOW_DUR * max_windows_to_process): #iterate through all time windows
               last_time=windStartTime+WINDOW_DUR * max_windows_to_process
               if (last_time > tot_time):
                    last_time=tot_time
               start_window_index=windStartTime//WINDOW_DUR
               request_coh_vals_someWinds_currChanPair(chanInd0_data[0],chanInd1_data[0],windStartTime,last_time,start_window_index, WINDOW_DUR,JIGGLING_LENGTH,sampling_frequency,coh_array,worker_processes)
         chan_pair_index+=1
         coh_arrays.append(coh_array)

   wait_for_results(worker_processes)

   coh_vals_allWinds_allChans = []
   window_index=0
   for windStartTime0 in range(0, tot_time , WINDOW_DUR): #iterate through all time windows
       coh_vals_currWind_allChans = []
       chan_pair_index=0
       for chanInd0 in range(min_chanInd0,max_chanInd0+1): #iterate through all channels
           coh_vals_currWind_currChan = []
           for chanInd1 in range(min_chanInd1,max_chanInd1+1): #iterate through all channels
               coh_val_currPair=coh_arrays[chan_pair_index][window_index]
               chan_pair_index+=1
               #add this coh val to list of vals of coherence between current 1st/main chan (chan in 2nd-outer-most loop) and all other chans
               coh_vals_currWind_currChan.append(coh_val_currPair)
           #add this list to list of lists that are the same, except for different 1st/main chans
           coh_vals_currWind_allChans.append(coh_vals_currWind_currChan)
       #add this list to list of lists that are the same, except for different 1st/main time windows
       #this list (below) will be our final 3D matrix
       coh_vals_allWinds_allChans.append(coh_vals_currWind_allChans)
       window_index+=1

   print ("done building matrix at ", datetime. now())
   f=open(args.output_data_dir+"data_array1.txt","w+")
   #num_winds = int(tot_time/WINDOW_DUR)
   #f.write(num_winds)
   f.write(str(num_chans)+"\n")
   for wind in coh_vals_allWinds_allChans:
       for cohMat in wind:
           for cohVal in cohMat:
               f.write(str(cohVal)+"\n")
   f.close()

   #print final 3D matrix
   print(coh_vals_allWinds_allChans)

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
