#!/usr/bin/env python
# coding: utf-8

"""
This module opens a real .fif file and replays it, producing a new .fif file every <time_interval_sec_between_files> seconds. Each such generated file contains <time_interval_sec_between_files> of data. For an input .fif file with per-channel frequency of 1000 samples a second, files produced with time_interval_sec_between_files=5 contain 5000 samples per channel. 
"""

import os
import time
import argparse
from datetime import datetime
from datetime import timedelta
import mne
import math
import numpy as np

debug_mode=False

def process_arguments():
   parser = argparse.ArgumentParser(description='Publish MEG window every N seconds')
   parser.add_argument('--output_data_dir',action='store',dest='output_data_dir',help='output data directory',default='');
   parser.add_argument('--output_data_file_prefix',action='store',dest='output_data_file_prefix',help='prefix of output data files',default='');
   parser.add_argument('--input_data_dir',action='store',dest='input_data_dir',help='input data directory',default='');
   parser.add_argument('--input_data_file',action='store',dest='input_data_file',help='input data file',default='');
   parser.add_argument('--max_channels_to_process',action='store',dest='max_channels_to_process',help='maximum number of channels to process',type=int, default='-1');
   parser.add_argument('--time_interval_sec_between_files',action='store',dest='time_interval_sec_between_files',help='number of seconds before creation of each file',type=int, default='-1')
   parser.add_argument('--debug',action='store_true',dest='debug',help='output debugging info if specified');
   args=parser.parse_args()	
   if args.input_data_dir:
     	args.input_data_dir=args.input_data_dir+"/"
   if args.output_data_dir:
     	args.output_data_dir=args.output_data_dir+"/"
   if not args.input_data_file:
      raise Exception ('input_data_file was not defined')
   if not args.output_data_file_prefix:
      raise Exception ('output_data_file_prefix was not defined')
   if args.time_interval_sec_between_files <=0:
      raise Exception ('time_interval_sec_between_files was not defined or was set to non-positive value')
   return args	

def compute_file_name_for_file(output_data_dir,output_data_file_prefix,file_time):
    day=file_time.year*10000+file_time.month*100+file_time.day
    time=file_time.hour*10000+file_time.minute*1000+file_time.second;
    full_time=day*1000000+time
    return (output_data_dir+output_data_file_prefix+str(full_time)+".fif")
	
if __name__ == '__main__': 
   args=process_arguments()

   debug_mode=args.debug
   
   print ("start at ", datetime. now())
   #Get neural data
   raw_data = mne.io.read_raw_fif(args.input_data_dir+args.input_data_file, preload=False)
   print ("done loading raw data at ", datetime. now())
   sampling_frequency=raw_data.info['sfreq']
   total_time_sec = len(raw_data)//sampling_frequency;
   num_chans=len(raw_data.ch_names)
   if (args.max_channels_to_process < num_chans and args.max_channels_to_process > 0):
        num_chans=args.max_channels_to_process

   time_interval_sec_between_files=args.time_interval_sec_between_files
   num_time_intervals=total_time_sec // time_interval_sec_between_files
   num_time_intervals=int(num_time_intervals)
   total_time_sec=num_time_intervals*time_interval_sec_between_files

   file_time=datetime.now()
   channel_picks=slice(0,num_chans)
   if debug_mode:
           print ("channel_picks ", channel_picks)

   for i in range (0,num_time_intervals): 
       file_name=compute_file_name_for_file(args.output_data_dir,args.output_data_file_prefix,file_time)
       #give the file its final name, ending in .fif, only after it was fully written 
       raw_data.save(file_name+".tmp",picks=channel_picks,tmin=i*time_interval_sec_between_files,tmax=(i+1)*time_interval_sec_between_files)
       os.rename(file_name+".tmp",file_name)
       file_time=file_time+timedelta(0,time_interval_sec_between_files)
       current_time=datetime.now()
       sleep_sec=file_time.timestamp()-current_time.timestamp()
       if debug_mode:
           print ("sleep_sec ", str(sleep_sec))
       if (sleep_sec > 0):
           time.sleep(sleep_sec)
   print ("end at ", datetime. now())

