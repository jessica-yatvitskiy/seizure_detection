import argparse
import numpy as np
import matplotlib.pyplot as plt

SEIZ_START_TIMES=[694000]
SEIZ_END_TIMES=[738000]
MAX_TIME=1440000

#Processes command-line arguments
def process_arguments():
   parser = argparse.ArgumentParser(description='Parameters of plotSavedNpArray')
   parser.add_argument('--input_data_dir',action='store',dest='input_data_dir',help='input data directory',default='')
   parser.add_argument('--input_data_file',action='store',dest='input_data_file',help='input data file',default='')
   parser.add_argument('--output_data_dir',action='store',dest='output_data_dir',help='output_data_dir',default='')
   parser.add_argument('--output_data_file',action='store',dest='output_data_file',help='output data file',default='')
   parser.add_argument('--title',action='store',dest='title',help='title of plot',default='')
   parser.add_argument('--xlabel',action='store',dest='xlabel',help='label for X axis',default='')
   parser.add_argument('--ylabel',action='store',dest='ylabel',help='label for Y axis',default='')
   parser.add_argument('--window_dur',action='store',dest='window_dur',help='distance between time points, in msec',default=0,type=int)

   args=parser.parse_args()

   if args.input_data_dir:
        args.input_data_dir=args.input_data_dir+"/"
   if args.output_data_dir:
        args.output_data_dir=args.output_data_dir+"/"
   if not args.input_data_file:
        raise Exception ('input_data_file was not defined')
   if not args.output_data_file:
        raise Exception ('output_data_file was not defined')
   if not args.title:
        raise Exception ('title was not defined')
   if args.window_dur==0:
        raise Exception ('window_dur was not defined')

   return args


args=process_arguments()
data_to_plot=np.load(args.input_data_dir+args.input_data_file)
figNum=1

plt.figure(figNum,figsize=(14,12))
plt.title(args.title)
plt.plot(data_to_plot)
x_locs=[]
x_labels=[]
max_windows=int (MAX_TIME/args.window_dur)
if (MAX_TIME % args.window_dur) !=0:
   max_windows+=1
for i in range(0,max_windows):
   x_locs.append(i)
   x_labels.append(str(i))
plt.xticks(x_locs,x_labels)
plt.xlabel(args.xlabel,fontsize=18)
plt.ylabel(args.ylabel,fontsize=18)
plt.axvline(x=SEIZ_START_TIMES[0]/args.window_dur, color='r')
plt.axvline(x=SEIZ_END_TIMES[0]/args.window_dur, color='r')
#plt.show()
plt.savefig(args.output_data_dir+args.output_data_file)
