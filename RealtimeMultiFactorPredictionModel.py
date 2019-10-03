#!/usr/bin/env python

"""
I have hypothesized that three factors--coherence across channels, number of HFOs, and correlation across channels--could facilitate an early prediction of epileptic seizures.
This program computes these factors in realtime. It feeds these continuously computed factors into the LinearDiscriminantAnalysis prediction module, which was previously trained on existing historical data from the same or different patients

This code represents a general model of how various facors can be computed across multiple MEG and EEG channels; some of those factors involve computations for all combinations of all channel pairs and are even aggregated across multip;e different time shifts between each pair of MEG/EEG channel pairs. The computations of such factors are very intensive. For example, a computation of the coherence factor for about 100 neurological channels, with a signal frequency of 1000 samples per second, across a time window of 500 msec, and with a time shift between -100 and 600 msec, with 100 msec between different the values of shift, requires 100*100*7=70,000 computations of coherence per second, with each computation applied to over 500 data points from each channel of the pair. And in order to process this in real-time, this computation cannot take more than 500 msec!

It is impossible to process data this quickly using the fastest implementatiuon of the coherence function available for Python, scipy.signal.coherence. 1000 invocations of that function on 500 datapoins per channel takes about 600 msec on the computer where the measurements were made (8-core 3.4 GHz Intel processor, 16GB RAM, on Windows). 70,000 computations would take about 40 seconds, and even if parallelized across all cores, it would take at best 5 seconds, instead of the target of 500 msec. And this is just for one of the factors, coherence!

The imported module spectral.py contains a modified implementation of coherence. It makes use of the fact that scipy.signal.coherence spends 2/3 of its time computing csd function on data fully from channel X or fully from challel Y. Thus, the result of that computation can be computed once for each channel for each window, for each jiggling offset, reducing the amount of this computation by 100 times. Additionally, each invocation of the csd function spends almost 1/3 of its time doing initialization wotk that depends only on the size of the time window (500 in our case). This initialization can be done once, making csd calls take 2/3 of time they take in scipy implementation. Furthermore, each csd call computed Fourier transform for X data and then for Y data. Again, this can be done once per channel for each window, for each jiggling offset, reducing amount of time spent of Fourier transform computation 100 times. As a result of these changes, csd computation done for each pair of channels is now takes about 25 msec per 100 invocations, instead of 600 msec. Also, the overall performance gain, when per-channel computations are included, is about 20 times.

Another challenge, however, was the parallelization of the above-mentioned logic. This is because Python Pool, from the multiprocessing package, needs all the parameters of the functions it processes to be serialized (using pickle). This serializaton is done automatically by the Pool module. Further, since the amount of data and the amount of per-channel state involved in coherence computations is large, this seralization makes Pool able to recieve far fewer requsts per second than it would otherwise. It was observed that only about 30 function calls per second could be handled by the Pool because of this serialization problem. This prevented use of per-channel function invocations with the Pool. Instead, the logic below uses per-window (500 msec) invocations, each taking about 2 seconds. Thus, each subprocess handles its own 500 msec window, reporting results with latency of about 2 seconds. This latency, while undesirable, is not critical for timely seizure predictions.

Since it takes 2 seconds for the Pool to process each request, and we need to submit a new request every 500 msec, i.e. for every time window, we use Pool.apply_async method, to request asynchronous processing, we then make sure that we process computed results in the order of the time windows for which they were computed.

In order to simulate the arrival of patient data in real-time, periodicMEGDataPublisher.py module was run in parallel with this code, producing a file in .fif format, with ten time windows, every 5 seconds, extracting data for those windows from a real file with patient data. In order to detect the arrival of a new .fif file in an efficient way, the watchdog Python package was used.

"""


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
from multiprocessing import Process, Pool, Array
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pickle
from det_HFOs_Method import det_HFOs

WINDOW_DUR = 500 # ms
JIGGLING_LENGTH = 100 # ms
WINDOW_OVERLAP = 0

compute_factors_pool=None
global_computation_state=None

debug_mode=False

class ComputationState:
   """
   This class maintains state of all computation, done for all computed factors, as well as seralized state of trained LinearDiscriminantAnalysis model
   """
   def __init__(self,args,window_dur,jiggling_length,min_required_continuous_windows,first_window_to_use_index,last_window_to_use_index):
      self.args=args

      #How many adjacent windows need to be supplied in order to compute factors for one window. Because of jiggling, we need a previous window and next window, thus this value is 3 for our case
      self.min_required_continuous_windows=min_required_continuous_windows

      #Index of the first window that we can compute. Because of jiggling, this index is 1
      self.first_window_to_use_index=first_window_to_use_index

      #Jiggling requires that we keep three latest windows in order to compute the middle one. Index of the last window that can be computed is also 1,which is the index of the middle window
      self.last_window_to_use_index=last_window_to_use_index

      self.window_dur=window_dur
      self.jiggling_length=jiggling_length

      #data arrays that hold three most recent windows per channel
      self.data_arrays_for_all_channels=None

      #Number of most recent windows currebtly loaded into  self.data_arrays_for_all_channels. It wil stay at 3 after three windows were received for input data files
      self.num_loaded_windows=0

      #an index into a list of per_channel_computation_states and initial_states_for_computation, for every computed factor
      self.per_factor_state_indexes={"COHERENCE":0,"NUM_HFOS":1}

      #per-channel state for our implementation of coherence function
      self.per_channel_computation_states=[{},{}]

      #pre-computed initial state for our implementation of coherence function
      self.initial_states_for_computation=[None,None]

      #maximum time in milliseconds, assuming a sample per millisecond, for which per-channel state for our implementation of coherence function was already computed
      self.max_time_with_preprocessed_coherence=-1

      #List of non-aggregated across channels results computed for each factor, per time window. The results for the most recent self.num_windows_to_maintain time windows are kept here
      self.results_for_computed_window=[]

      #List of aggregated across channels results, one number per factor, per time window. Aggregated results for the most recent self.num_windows_to_maintain time windows are kept here
      self.reduced_results_for_computed_window=[]

      #Number of the most recent time windows to keep in self.results_for_computed_window and self.reduced_results_for_computed_window
      self.num_windows_to_maintain=10

      #number of time windows for which computation of factors was requested so far
      self.num_submitted_windows=0

      #List of outstanding per-window requests to the Pool
      self.async_requests=[]

      #Trained state of LinearDiscriminantAnalysis model, used for predictions
      self.ml_training_state=None

class FifEventHandler(FileSystemEventHandler):
#this class is a handler for the discovery of new .fif files

   def __init__(self,computation_state):
      self.computation_state=computation_state

   def on_created(self,event):
     if event.src_path.endswith('.fif')==True:
         print(event.src_path+" got created");
         process_new_data_file(event.src_path,self.computation_state)
   def on_moved(self,event):
     #in order to prevent reading of incomplete files, those files are initially created by periodicMEGDataPublisher.py with .tmp extension, and then are renamed end with .fif
     if event.dest_path.endswith('.fif')==True:
         print(event.dest_path+" got created");
         process_new_data_file(event.dest_path,self.computation_state)

def process_arguments():
   parser = argparse.ArgumentParser(description='Computation of coherence with jiggling.')
   parser.add_argument('--output_data_dir',action='store',dest='output_data_dir',help='output data directory',default='');
   parser.add_argument('--input_data_dir',action='store',dest='input_data_dir',help='input data directory',default='');
   parser.add_argument('--output_matrix_file',action='store',dest='output_matrix_file',help='file with computed coherence matrix',default='');
   parser.add_argument('--input_training_state_file',action='store',dest='input_training_state_file',help='file with pickled state of LinearDiscriminantAnalysis',default='');
   parser.add_argument('--max_channels_to_process',action='store',dest='max_channels_to_process',help='maximum number of channels to process',type=int, default='-1');
   parser.add_argument('--debug',action='store_true',dest='debug',help='output debugging info if specified');
   args=parser.parse_args()
   if args.input_data_dir:
     	args.input_data_dir=args.input_data_dir+"/"
   if args.output_data_dir:
     	args.output_data_dir=args.output_data_dir+"/"
   return args

def request_computation_for_all_factors(data,start_time, end_time, total_time, time_origin_offset, num_chans, window_dur, jiggling_length, sampling_frequency, debug_mode,computation_state):
   """
   In real-time processing mode, coherence and similar computations need last three windows and processes the middle one, thus start_time is always window_dur.  For this reason, we also need time_origin_offset, which is number of samples from the beginning of all processing and until the first of the three windows passed into this function.
   """

   #Since watchdog runs in a separate process, and since its discovery of files initiates all computations, the global Pool could not be created in the main process, and this is why we create it here
   global compute_factors_pool
   if compute_factors_pool==None:
       compute_factors_pool=Pool(12)

   for windStartTime0 in range(start_time, end_time-window_dur+1 , window_dur): #iterate through all full time windows
       compute_factors_request=ComputeFactorsRequest(num_chans,windStartTime0,total_time,time_origin_offset,window_dur,jiggling_length)
       coherence_index_into_states=computation_state.per_factor_state_indexes["COHERENCE"]
       prepair_initial_and_per_channel_states_for_coherence(data, windStartTime0, total_time, time_origin_offset, num_chans, window_dur, jiggling_length, sampling_frequency, coherence_index_into_states, debug_mode,computation_state)
       compute_factors_request.add_factor("COHERENCE",coherence_index_into_states,computation_state.per_channel_computation_states[coherence_index_into_states])
       num_hfos_index_into_states=computation_state.per_factor_state_indexes["NUM_HFOS"]
       #for the num_hfos computation, the per-channel computation state is not needed
       #however, the data is needed, so it is stored in compute_factors_request in place of the per-channel computation state
       compute_factors_request.add_factor("NUM_HFOS",num_hfos_index_into_states,data)
       computation_state.async_requests.append([compute_factors_request,time_origin_offset+windStartTime0,False])
       #Initiate asynchronous processing of a time window
       compute_factors_pool.apply_async(compute_factors_from_states, args=(compute_factors_request,),callback=process_result_for_computed_window)

def prepair_initial_and_per_channel_states_for_coherence(data, windStartTime0, total_time, time_origin_offset, num_chans, window_dur, jiggling_length, sampling_frequency, coherence_index_into_states, debug_mode,computation_state):

   #precompute global and per-channel coherence states using our own implementation of coherence
   for chanInd0 in range(0,num_chans): #iterate through all channels
       if (computation_state.initial_states_for_computation[coherence_index_into_states]==None):
            chanInd0_seg = data[chanInd0][windStartTime0:windStartTime0+window_dur]
            computation_state.initial_states_for_computation[coherence_index_into_states]=spectral.compute_initial_coherence_state(chanInd0_seg,nfft=250,nperseg=250)
       for chanInd1 in range(0,num_chans): #iterate through all channels
            if chanInd0 != chanInd1:
                   for windStartTime1 in range(max(windStartTime0-jiggling_length, 0), min(windStartTime0+window_dur+jiggling_length,total_time-window_dur+1), jiggling_length): #iterate through each window of window duration msec, starting from jiggling_length msec before start of current window to jiggling_length msec after end of current window
                        if (time_origin_offset+windStartTime1 > computation_state.max_time_with_preprocessed_coherence):
                              compute_coherence_states_for_time_window(computation_state.per_channel_computation_states[coherence_index_into_states],data,windStartTime1,time_origin_offset,window_dur,num_chans,sampling_frequency,computation_state.initial_states_for_computation[coherence_index_into_states])
                              computation_state.max_time_with_preprocessed_coherence=time_origin_offset+windStartTime1
                        if (windStartTime0 > computation_state.max_time_with_preprocessed_coherence):
                              compute_coherence_states_for_time_window(computation_state.per_channel_computation_states[coherence_index_into_states],data,windStartTime0,time_origin_offset,window_dur,num_chans,sampling_frequency,computation_state.initial_states_for_computation[coherence_index_into_states])
                              computation_state.max_time_with_preprocessed_coherence=time_origin_offset+windStartTime0


class ComputeFactorsRequest:
   #Class for requesting a computation for all facors
   def __init__(self,num_chans,windStartTime0,total_time,time_origin_offset,window_dur,jiggling_length):
       self.num_chans=num_chans
       self.windStartTime0=windStartTime0
       self.total_time=total_time
       self.time_origin_offset=time_origin_offset
       self.jiggling_length=jiggling_length
       self.window_dur=window_dur
       self.factor_names=[]
       self.factor_indexes=[]
       self.per_channel_computation_states=[]

   def add_factor(self,factor_name,factor_index,per_channel_computation_state):
       self.factor_names.append(factor_name)
       self.factor_indexes.append(factor_index)
       self.per_channel_computation_states.append(per_channel_computation_state)

   def get_factor_index(self,factor_name):
       i=0
       for name in self.factor_names:
           if (name == factor_name):
              return self.factor_indexes[i]
           i+=1
       return -1

def compute_factors_from_states(compute_factors_request):
    coherence_index_into_states=compute_factors_request.get_factor_index("COHERENCE")
    coh_vals_curr_window_all_chan=compute_coherence_from_states(compute_factors_request,coherence_index_into_states)
    num_hfos_index_into_states=compute_factors_request.get_factor_index("NUM_HFOS")
    num_hfos_all_chan=compute_num_hfos(compute_factors_request,num_hfos_index_into_states)
    windStartTime0=compute_factors_request.windStartTime0
    time_origin_offset=compute_factors_request.time_origin_offset
    result_for_computed_window=[]
    result_for_computed_window.append(coh_vals_curr_window_all_chan)
    result_for_computed_window.append(num_hfos_all_chan)
    return [result_for_computed_window,time_origin_offset+windStartTime0]

def compute_coherence_from_states(compute_coherence_request,coherence_index_into_states):
    #compute num_channels x num_channels coherence matrix, using pre-computed initial and per-channel coherence states, for the new time window

    windStartTime0=compute_coherence_request.windStartTime0
    jiggling_length=compute_coherence_request.jiggling_length
    total_time=compute_coherence_request.total_time
    time_origin_offset=compute_coherence_request.time_origin_offset
    window_dur=compute_coherence_request.window_dur
    num_chans=compute_coherence_request.num_chans
    coherence_states=compute_coherence_request.per_channel_computation_states[coherence_index_into_states]
    coh_vals_curr_window_all_chan = []
    for chanInd0 in range(0,num_chans): #iterate through all channels
           coh_vals_curr_window_curr_chan = []
           for chanInd1 in range(0,num_chans): #iterate through all channels
                if chanInd0 == chanInd1:
                     coh_val_curr_pair = 1
                else:
                     coh_vals_curr_pair = []
                     for windStartTime1 in range(max(windStartTime0-jiggling_length, 0), min(windStartTime0+window_dur+jiggling_length,total_time-window_dur+1), jiggling_length): #iterate through each window of window duration msec, starting from jiggling_length msec before start of current window to jiggling_length msec after end of current window
                           freqs,coh_winds_curr_pair_curr_wind1 = spectral.compute_coherence_from_states(get_coherence_state_for_channel(coherence_states,chanInd0,time_origin_offset+windStartTime0),get_coherence_state_for_channel(coherence_states,chanInd1,time_origin_offset+windStartTime1))
                           #compute mean of coherence vals returned by plt.cohere (which returns one coherence val per frequency
                           #this mean represents the coh val of the current 2 chan segments
                           coh_val_curr_pair_curr_wind1=np.mean(coh_winds_curr_pair_curr_wind1)
                           coh_vals_curr_pair.append(coh_val_curr_pair_curr_wind1) #add coh val of the current 2 chan segments to array
                     #find maximum coh val among the coh vals computed for each pair of chan segments for the current 2 chans
                     #set the coh value of the current pair of chans to this max
                     coh_val_curr_pair = max(coh_vals_curr_pair)
                coh_vals_curr_window_curr_chan.append(coh_val_curr_pair)
           coh_vals_curr_window_all_chan.append(coh_vals_curr_window_curr_chan)
    return coh_vals_curr_window_all_chan

def compute_num_hfos(compute_coherence_request,num_hfos_index_into_states):
    #compute num_channels array with number of HFO events per channel, for the new time window
    windStartTime0=compute_coherence_request.windStartTime0
    total_time=compute_coherence_request.total_time
    window_dur=compute_coherence_request.window_dur
    num_chans=compute_coherence_request.num_chans
    data=compute_coherence_request.per_channel_computation_states[num_hfos_index_into_states]
    amp_thresh = 3*np.std(data)

    num_HFOs_all_chans=[]
    for chan in range(num_chans):
         chan_seg = data[chan][windStartTime0:windStartTime0+window_dur]
         #det_HFOs expects 2dim numpy array
         chan_seg=np.asarray([chan_seg])
         HFO_infos_curr_chan=det_HFOs(chan_seg,amp_thresh)
         num_HFOs_curr_chan=len(HFO_infos_curr_chan[0])
         num_HFOs_all_chans.append(num_HFOs_curr_chan)

    return num_HFOs_all_chans

def process_result_for_computed_window(result):
    #process results for computed time windows in the order of timestamps of those windows

    global global_computation_state
    async_requests=global_computation_state.async_requests
    result_for_computed_window=result[0]
    time=result[1]

    #we have only one request per time window, thus a request with matching timestamp is one for which this result was computed
    if (async_requests[0][1]==time):
       async_requests[0][2]=True
       i=0
    while  len(async_requests)>0  and async_requests[0][2]==True:
       global_computation_state.results_for_computed_window.append(result_for_computed_window)
       #keep only num_windows_to_maintain of the latest computed results
       while len(global_computation_state.results_for_computed_window) > global_computation_state.num_windows_to_maintain:
           global_computation_state.results_for_computed_window.pop(0)
       process_last_result_for_computed_window(global_computation_state)
       async_request_time=async_requests[0][1]
       async_request=async_requests[0][0]
       for factor_index in async_request.factor_indexes:
           remove_old_per_channel_computation_states(computation_state.per_channel_computation_states[factor_index],async_request_time-2*async_request.jiggling_length)
       async_requests.pop(0)
    #if this time is not the lowest in the queue of computation requests, find the request with matching timestamp and mark it as computed
    for async_request in async_requests:
       if async_request[1]==time:
           async_request[2]=True
       elif (async_request[1] > time):
            break
    print("num waiting async requests "+str(len(async_requests)))

def compute_coherence_states_for_time_window(coherence_states,data,window_start_time,origin_offset,window_duration,num_chans,sampling_frequency,initial_coherence_state):
    """
    In real-time processing mode, coherence needs last three windows and processes the middle one, thus start_time is always window_duration.  For this reason, we also need time_origin_offset, which is number of samples from the beginning of all processing and until the first of the three windows passed into this function.
    """
    coherence_state_for_time={}
    for chan in range (0,num_chans):
         chan_seg = data[chan][window_start_time:window_start_time+window_duration]
         coherence_state=spectral.compute_coherence_state_for_one_side(chan_seg,initial_coherence_state,fs=sampling_frequency)
         coherence_state_for_time[chan]=coherence_state
    coherence_states[origin_offset+window_start_time]=coherence_state_for_time

def remove_old_per_channel_computation_states(coherence_states,highest_window_time_to_remove):
    keys=coherence_states.keys()
    keys_list=list(keys)
    for key in keys_list:
       if (key <= highest_window_time_to_remove):
          coherence_states.pop(key)

def process_last_result_for_computed_window(computation_state):
   #for each factor, compute aggregated result, reducing num_channels*num_channels matrix or num_channel array into a single number
   #then aggregate these per-factor numbers across last N time windows
   #then submit a list of resulting numbers, one per per factor, into  LinearDiscriminantAnalysis prediction module

   last_results=computation_state.results_for_computed_window[-1]
   aggregated_results=[]
   factor_index=0
   if len(computation_state.reduced_results_for_computed_window)==0:
      for i in range(len(last_results)):
         computation_state.reduced_results_for_computed_window.append([])
   for factor_result in last_results:
      reduced_factor_result=reduce_result_for_factor(factor_result,computation_state,factor_index)
      computation_state.reduced_results_for_computed_window[factor_index].append(reduced_factor_result)
      while len(computation_state.reduced_results_for_computed_window[factor_index]) > computation_state.num_windows_to_maintain:
           computation_state.reduced_results_for_computed_window[factor_index].pop(0)
      if (len(computation_state.reduced_results_for_computed_window[factor_index])==computation_state.num_windows_to_maintain):
           aggregated_results.append(compute_aggregate_across_windows( computation_state,factor_index))
      factor_index+=1
   if (len(aggregated_results) > 0):
        make_prediction(computation_state,aggregated_results)

def reduce_result_for_factor(factor_result,computation_state,factor_index):
    #compute aggregated result for factor, reducing num_channels*num_channels matrix or num_channel array into a single number
    return np.mean(factor_result)
    pass

def compute_aggregate_across_windows(computation_state,factor_index):
    return np.mean(computation_state.reduced_results_for_computed_window[factor_index])
    pass

def make_prediction(computation_state,aggregated_results):
    if global_computation_state.ml_training_state!=None:
        aggregated_results_2dim=[aggregated_results]
        predicted_outcome=computation_state.ml_training_state.predict(aggregated_results_2dim)
        print("predicted_outcome:"+str(predicted_outcome[0]))

def get_coherence_state_for_channel(coherence_states,channel,window_time):
    return coherence_states.get(window_time).get(channel)

def process_new_data_file(filename,computation_state):
   #Since watchdog runs in a separate process, and since its discovery of files initiates all computations, which lead to changes in computation_state, global computation_state variable could not be set in the main process, and this is why we set it here
   global global_computation_state
   global_computation_state=computation_state

   #load trained ML module, if it was not yet loaded
   if global_computation_state.ml_training_state==None and global_computation_state.args.input_training_state_file:
       global_computation_state.ml_training_state=pickle.load(open(global_computation_state.args.input_data_dir+global_computation_state.args.input_training_state_file,'rb'))
   #Get neural data
   try:
      raw_data = mne.io.read_raw_fif(filename, preload=True)
   except (IOError) as e:
      #Permissioned denied error may be thrown if the file has just been created
      print ("failed to read file "+filename+".Retrying ...");
      time.sleep(1)
      raw_data = mne.io.read_raw_fif(filename, preload=True)
   total_time = len(raw_data);
   if total_time % computation_state.window_dur !=0:
        if (total_time % computation_state.window_dur==1): #can happen because mne.io.Raw.save includes end timestamp
           total_time-=1
        else:
           raise Exception ( "file "+filename+" contains fractional number of windows of size "+str(computation_state.window_dur)+" total_time="+str(total_time))
   num_windows_in_file= int (total_time // computation_state.window_dur)
   num_chans=len(raw_data.ch_names)
   sampling_frequency=raw_data.info['sfreq']
   data=raw_data.get_data()
   if (computation_state.args.max_channels_to_process < num_chans and computation_state.args.max_channels_to_process > 0):
        num_chans=computation_state.args.max_channels_to_process
   if computation_state.data_arrays_for_all_channels==None:
       	data_arrays_for_all_channels=[None] * num_chans
        for i in range (0,num_chans):
            data_arrays_for_all_channels[i]=np.zeros(shape=(computation_state.window_dur * computation_state.min_required_continuous_windows))
        computation_state.data_arrays_for_all_channels=data_arrays_for_all_channels

   #load data from a file into computation_state.data_arrays_for_all_channels
   for window_index in range (0,num_windows_in_file):
        if computation_state.num_loaded_windows < computation_state.min_required_continuous_windows:
             for chan_index in range (0,num_chans):
                 computation_state.data_arrays_for_all_channels[chan_index][computation_state.window_dur*computation_state.num_loaded_windows:computation_state.window_dur*(computation_state.num_loaded_windows+1)]=data[chan_index][computation_state.window_dur*window_index:computation_state.window_dur*(window_index+1)]
             computation_state.num_loaded_windows+=1
        else:
             for chan_index in range (0,num_chans):
                 np.roll(computation_state.data_arrays_for_all_channels[chan_index],-computation_state.window_dur)
                 computation_state.data_arrays_for_all_channels[chan_index][computation_state.window_dur*(computation_state.num_loaded_windows-1):computation_state.window_dur*computation_state.num_loaded_windows]=data[chan_index][computation_state.window_dur*window_index:computation_state.window_dur*(window_index+1)]
        if  computation_state.num_loaded_windows < computation_state.min_required_continuous_windows:
           return
        request_computation_for_all_factors(computation_state.data_arrays_for_all_channels,computation_state.first_window_to_use_index * computation_state.window_dur, (computation_state.last_window_to_use_index+1)*computation_state.window_dur, computation_state.min_required_continuous_windows * computation_state.window_dur, computation_state.num_submitted_windows * computation_state.window_dur, num_chans, computation_state.window_dur, computation_state.jiggling_length, sampling_frequency, debug_mode,computation_state)
        computation_state.num_submitted_windows+=1

if __name__ == '__main__':
   args=process_arguments()

   debug_mode=args.debug

   print ("start at ", datetime. now())

   min_required_continuous_windows=3 #coherence computation needs three adjacent windows, to support jiggling
   first_window_to_use_index=1 #only the middle window is fully used, the surrounding windows are only used for jiggling
   last_window_to_use_index=1 #only the middle window is fully used, the surrounding windows are only used for jiggling
   computation_state=ComputationState(args,WINDOW_DUR,JIGGLING_LENGTH,min_required_continuous_windows,first_window_to_use_index,last_window_to_use_index)

   event_handler=FifEventHandler(computation_state)
   observer = Observer()
   observer.schedule(event_handler, args.input_data_dir)
   #begin waiting for data files
   observer.start()
   while True:
     time.sleep(1)
   observer.join()

   #this code will not be reached because the observer keeps waiting for new files
   print ("end at ", datetime. now())
