import argparse
import numpy as np
import matplotlib.pyplot as plt

SEIZ_START_TIME=694000
SEIZ_END_TIME=738000
TOTAL_TIME=1428000

#Processes command-line arguments
def process_arguments():
   parser = argparse.ArgumentParser(description='Parameters of coh_max_min')
   parser.add_argument('--input_matrix_file',action='store',dest='input_matrix_file',help='input matrix file',default='')
   parser.add_argument('--output_plot_file',action='store',dest='output_plot_file',help='output plot file',default='')
   args=parser.parse_args()
   if not args.input_matrix_file:
       raise Exception ('input_matrix_file was not specified')
   return args

args=process_arguments()
coh_matrix=np.load(args.input_matrix_file)
print(coh_matrix.shape)
mean_array=np.mean(coh_matrix,(1,2))
num_time_windows=len(mean_array)
window_dur=TOTAL_TIME//num_time_windows
figNum=1
plt.figure(figNum,figsize=(14,12))
plt.title("Avg Coherence over Time")
plt.plot(mean_array)
print(mean_array)
print(num_time_windows)
moving_ave=[]
moving_ave_wind=10 #50
for i in range(0,num_time_windows,moving_ave_wind):
    ave_curr_10=np.mean(mean_array[i:i+moving_ave_wind])
    for j in range(0,moving_ave_wind):
        moving_ave.append(ave_curr_10)
plt.plot(moving_ave,"k-")
plt.xlabel("Time (window #)")
x_locs=[]
x_labels=[]
for i in range(10):
   window=i*num_time_windows/10
   x_locs.append(window)
   x_labels.append(str((int)(window*window_dur/1000)))

plt.xticks(x_locs,x_labels)
plt.ylabel("Avg Coherence")
plt.axvline(x=SEIZ_START_TIME/window_dur,color='r')
plt.axvline(x=SEIZ_END_TIME/window_dur,color='r')
if args.output_plot_file:
    plt.savefig(args.output_plot_file)
else:
    plt.show()
