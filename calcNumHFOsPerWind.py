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
    amp_thresh = 3*np.std(dat)

    num_chans=shape[0]
    if (int(args.max_channels_to_process) < num_chans and int(args.max_channels_to_process) > 0):
        num_chans=int(args.max_channels_to_process)

    #Calculate/create array of number of HFOs for each channel for each time window
    num_HFOs_allChans_allWinds=[]
    window_dur=int(args.wind_dur)
    for startTime in range(0,shape[1]-window_dur,window_dur):
        num_HFOs_allChans_currWind=[]
        for chanInd in range(0,num_chans):
            chan=dat[chanInd]
            temp_abr_chan=[chan[startTime:startTime+window_dur]]
            temp_abr_chan=np.asarray(temp_abr_chan)
            HFO_infos_currChan=det_HFOs(temp_abr_chan,amp_thresh)
            num_HFOs_currChan=len(HFO_infos_currChan[0])
            num_HFOs_allChans_currWind.append(num_HFOs_currChan);
        num_HFOs_allChans_allWinds.append(num_HFOs_allChans_currWind)

    #Print above created array, and save it to a file
    print(num_HFOs_allChans_allWinds)
    np.save(args.output_data_dir+args.output_data_file, num_HFOs_allChans_allWinds)
