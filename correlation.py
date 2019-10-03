import os
import mne
import math
import numpy as np
import matplotlib.pyplot as plt
from det_HFOs_Method import det_HFOs

def process_arguments():
   parser = argparse.ArgumentParser(description='Parameters of plot coherence')
   parser.add_argument('--input_data_dir',action='store',dest='input_data_dir',help='input_data_dir',default='')
   parser.add_argument('--input_data_file',action='store',dest='input_data_file',help='input data file',default='');
   parser.add_argument('--output_data_dir',action='store',dest='output_data_dir',help='output_data_dir',default='')
   parser.add_argument('--output_data_file',action='store',dest='output_data_file',help='output data file',default='');
   parser.add_argument('--wind_dur',action='store',dest='wind_dur',help='wind_dur',default='')
   parser.add_argument('--max_channels_to_process',action='store',dest='max_channels_to_process',help='maximum number of channels to process',type=int, default='-1')

   args=parser.parse_args()
   return args

if __name__ == '__main__':
   args=process_arguments()
   debug_mode=args.debug
   print ("start at ", datetime. now())
   if args.input_data_dir:
     	args.input_data_dir+="/"
   if args.output_data_dir:
     	args.output_data_dir+="/"

    #Get neural data
    filtered_data=mne.io.read_raw_fif(args.input_data_dir+args.input_data_file, preload=False)
    dat=filtered_data.get_data()
    shape=dat.shape
    print(shape)

    num_chans=shape[0]
    if (int(args.max_channels_to_process) < num_chans and int(args.max_channels_to_process) > 0):
        num_chans=int(args.max_channels_to_process)
    window_dur=int(args.wind_dur)

    #Calculate/create an array of correlation matrices across time
    corr_allChanPairs_allWinds=[]
    for startTime in range(0,shape[1]-window_dur,window_dur): #for each window
        corr_allChanPairs_currWind=[]
        for chan0_ind in range(0,num_chans): #for each channel (first in pair)
            corr_allChanPairs_currChan_currWind=[]
            for chan1_ind in range(0,num_chans): #for each channel (second in pair)
                if (chan0_ind==chan1_ind): #if 2 channels are known to be same (have same index), automatically set correlation to max (21)
                    corr_currChanPair_currChan_currWind=21
                else:
                    temp_abr_chan0=dat[chan0_ind][startTime:startTime+window_dur] #take segment of first channel corresponding to current window
                    temp_abr_chan1=dat[chan1_ind][startTime:startTime+window_dur] #take segment of second channel corresponding to current window
                    corr_currChanPair_currChan_currWind=np.correlate(temp_abr_chan0,temp_abr_chan1)[0] #call numpy.correlate method on these 2 segments
                corr_allChanPairs_currChan_currWind.append(corr_currChanPair_currChan_currWind) #append output of numpy.correlate function to array
            corr_allChanPairs_currWind.append(corr_allChanPairs_currChan_currWind) #append above array to array
        corr_allChanPairs_allWinds.append(corr_allChanPairs_currWind) #append above array to array

    #Print above created array of correlation matrices across time, and save it to a file
    print(corr_allChanPairs_allWinds)
    np.save(args.output_data_dir+args.output_data_file, corr_allChanPairs_allWinds)
