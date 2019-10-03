import os
import mne
import math
import numpy as np

#For all chans in across full time of dat, this method detects HFOs.
#It returns a 3D arr containing the info of all the HFOs (startTime, endTime)
#Dimensions of returned 3D arr: # of chans (102), # of HFO in each/curr chan (varies across chans), amount of info items about each HFO (2)
def det_HFOs(dat,amp_thresh):
    # Put constant values in capital letters and explain what they represent
    TPEAK_DERIV_AVER_THRESH=5
    TPEAK_DERIV_ASYM_THRESH=2
    PEAK_NUM_THRESH=6
    HFO_DUR_THRESH=25
    HFO_infos=[]

    shape=dat.shape
    num_chans=shape[0]
    len_chan=shape[0]

    # Iterating over all channels
    for chanInd in range(num_chans):
        # Get data of that channel
        chan0=dat[chanInd]
        # Computing first-order derivative
        chan_deriv1 = np.diff(chan0)
        #chan_deriv2 = np.diff(chan_deriv1)
        len_chan_deriv1=len(chan_deriv1)
        tpeak=[]
        vpeak=[]
        ind_tallEnough_peaks=[]
        HFOStart=[]
        HFOEnd=[]
        HFOPeakNum=[]
        ind_longHFO=[]
        HFO_infos_currChan=[]

        #EOI/HFO must contain peaks
        for i in range(1, len_chan_deriv1):
            if (chan_deriv1[i-1]*chan_deriv1[i]<0):
                tpeak.append(i) #time of peak relative to startTime of dat
                vpeak.append(chan0[i]) # amplitude of the peak

        len_vpeak=len(vpeak)

        #tpeak_deriv is arr that represents the deriv of tpeak, the arr containing times of peaks
        tpeak_deriv=np.diff(tpeak)
        tpeak_deriv_len=len(tpeak_deriv)

        #Now we want to compute average and "asymmetry" arrays
        #First we set first val of aver arr to first val of tpeak_deriv
        if tpeak_deriv_len!=0:
            tpeak_deriv_aver=[tpeak_deriv[0]]
        tpeak_deriv_asym=[0]

        #add averages of adjacent elements in tpeak_deriv to average array
        #add absolute difference of adjacent elements in tpeak_deriv to asymmetry array
        for i in range(0,tpeak_deriv_len-1):
            aver=(tpeak_deriv[i]+tpeak_deriv[i+1])/2
            tpeak_deriv_aver.append(aver)
            abs_diff=abs(tpeak_deriv[i]-tpeak_deriv[i+1])
            tpeak_deriv_asym.append(abs_diff)

        #average array ends with last val of tpeak_deriv
        if tpeak_deriv_len!=0:
            tpeak_deriv_aver.append(tpeak_deriv[tpeak_deriv_len-1])
        tpeak_deriv_asym.append(0)

        #peaks must pass amplitude, average, and asymmetry threshholds
        #amplitude threshhold makes sure that the peak is tall enough
        #average threshhold makes sure that the signal does not change too much around the peak
        #reason: Dmitri observed that this threshhold improves HFO detection
        #average threshhold makes sure that the signal does not change too quickly around the peak
        #reason 1: Same as above
        #reason 2: It is not biologically possible for HFOs/real neural signal components to have above a certain freq.. Usually, when a frequency exceeding this threshhold is observed in a signal, it is simply noise.
        for i in range(0, len_vpeak):
            if (abs(vpeak[i])>=amp_thresh and tpeak_deriv_aver[i]<=TPEAK_DERIV_AVER_THRESH and tpeak_deriv_asym[i]<=TPEAK_DERIV_ASYM_THRESH):
                ind_tallEnough_peaks.append(i)
        tallPeaks_deriv=np.diff(ind_tallEnough_peaks)
        len_tPD=len(tallPeaks_deriv)

        #EOI/HFO must have high enough freq.
        #a lot of 1's in a row in tallPeaks_deriv represents that many peaks are occuring in consecutive miliseconds, which is indicative of an HFO.
        #as such, I declare the start of a sequence of consecutive 1's to be the start of a potential HFO and the end of such a sequence to be the end of a potential HFO.
        if (len_tPD!=0 and tallPeaks_deriv[0]==1):
            HFOStart.append(0)
        for i in range(1,len_tPD):
            # Check if the derivative is decreasing
            if (tallPeaks_deriv[i]==1 and tallPeaks_deriv[i-1]>1):
                HFOStart.append(i)
            # Check if the derivative is increasing
            elif (tallPeaks_deriv[i-1]==1 and tallPeaks_deriv[i]>1):
                HFOEnd.append(i)
        len_HFOEnd=len(HFOEnd)

        #EOI/HFO must contain enough (>=6) peaks
        for i in range(0, len_HFOEnd):
            HFOPeakNum.append(HFOEnd[i]-HFOStart[i])
        len_HPN=len(HFOPeakNum)

        #ind_longHFO contains indices (with respect to tpeak arr) of HFOs that have >=6 peaks
        for i in range(0, len_HPN):
            if HFOPeakNum[i]>=PEAK_NUM_THRESH:
                ind_longHFO.append(i)

        #Add start time and end time of HFOs to arr containing HFO infos for curr chan, if HFO is long enough, timewise (>=25 ms long)
        for i in ind_longHFO:
            HFO_start_Time=tpeak[ind_tallEnough_peaks[HFOStart[i]]]
            HFO_end_Time=tpeak[ind_tallEnough_peaks[HFOEnd[i]]]
            if (HFO_end_Time-HFO_start_Time)>=HFO_DUR_THRESH:
                    HFO_infos_currChan.append([HFO_start_Time,HFO_end_Time])
        HFO_infos.append(HFO_infos_currChan)


    return HFO_infos
