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

WINDOW_DUR = 500
#NUM_CHANS=102
seiz_startTimes=[694000]
seiz_endTimes=[738000]

def process_arguments():
   parser = argparse.ArgumentParser(description='Parameters of plot coherence')
   parser.add_argument('--output_data_dir',action='store',dest='output_data_dir',help='output data directory',default='');
   parser.add_argument('--input_data_dir',action='store',dest='input_data_dir',help='input data directory',default='')
   args=parser.parse_args()
   if args.input_data_dir:
     	args.input_data_dir=args.input_data_dir+"/"
   if args.output_data_dir:
     	args.output_data_dir=args.output_data_dir+"/"
   return args

args=process_arguments()
output_path=args.output_data_dir+"cohPlot.png"
coh_vals_allWinds_allChans=[]

f=open(args.input_data_dir+"data_array1.txt","r")
lines=f.readlines()
num_lines=len(lines)
num_chans=int(lines[0])
numWinds=0
ave_coh_vals_allWinds=[]
for i in range(1,num_lines-(num_chans*num_chans),(num_chans*num_chans)):
    numWinds+=1
    ave_coh_vals_currWind=0
    for j in range(i,i+(num_chans*num_chans)):
        curr_coh_val=float(lines[j])
        print(curr_coh_val)
        ave_coh_vals_currWind+=curr_coh_val
    ave_coh_vals_currWind=ave_coh_vals_currWind/(num_chans*num_chans)
    ave_coh_vals_allWinds.append(ave_coh_vals_currWind)
print(ave_coh_vals_allWinds)

plt.plot(ave_coh_vals_allWinds)

plt.axvline(x=(seiz_startTimes[0]//WINDOW_DUR),color='r')
plt.axvline(x=(seiz_endTimes[0]//WINDOW_DUR),color='y')

x_label_locs=[]
for windStartTime in range(0,numWinds,WINDOW_DUR):
    x_label_locs.append(windStartTime)
x_labels=[]
for windStartTime in range(0,numWinds,WINDOW_DUR):
    x_labels.append(windStartTime*WINDOW_DUR)
plt.xticks(x_label_locs,x_labels)

plt.savefig(output_path)
plt.show()
