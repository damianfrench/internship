# -*- coding: utf-8 -*-
"""
Created on Wed May 11 15:01:15 2022

@author: charlie
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 12:59:38 2022

@author: charlie
"""

'''Imports'''
import numpy as np
import pandas as pd
# import json
# from skimage.measure import block_reduce
import matplotlib.pyplot as plt
import scipy
import csv
# from scipy.fftpack import fft
# from scipy.fftpack import ifft
# from scipy.fftpack import fftfreq
# from scipy.spatial import distance
from scipy import stats
from numpy import trapz
import datetime
# import pandas
from scipy.signal import butter, sosfilt #, convolve 
import re


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning)



# set plot size
# SMALL_SIZE = 18
# MEDIUM_SIZE = 20
# BIGGER_SIZE = 30

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# define functions
'''Functions'''
#appends the individual data sets into collumns in an array 
def sort(file):
    rows=[]
    for row in file:
        if row[1] == 'Electrocardiogram':
            rows.append(row)
    return rows

#takes the comma seperated string of data from the list of lists, seperates it into floats and appends to values
#takes in the data lists and position in the list
def extract(rows_ex, m):
    string = rows_ex[m][5]
    time = rows_ex[m][0]
    str_values=string.split(',')
    values = []
    for x in str_values:
        try:
            value = eval(x)
            values = np.append(values, value)
        except SyntaxError:
            pass
    values = np.asarray(values)
    time = datetime.datetime.strptime(time,"%Y-%m-%dT%H:%M:%S%z")
    #??????????????????????????????????? impliment removal of unnesacery data
    
    
    
    '''
    time_data = [x - time_data[0] for x in time]
    time_data = [x.days for x in time_data]
    time_dif=np.diff(time_data)
    if time_dif[0] > 1 :
        time_data=time_data[1:]
        time_data=np.array(time_data)-float(time_data[0])
    else:
        pass
    
    time=time[len(time)-len(time_data)-1:]
    values=values[len(values)-len(time_data)-1:]
    '''
    return values, time


#checks the length of the array making it (9000,)
def length(l,z):
    g=np.zeros(z)
    if len(l)<len(g):
        shape=np.shape(l)
        g[:shape[0]]=l
    elif len(l)>len(g):
        g=l[:len(g)]
    else:
        g=l
    return g


#r_times=optimise_peaks(r_times, y, r_ind_s)
def optimise_peaks(h,j,l):
    i=[]
    #l=l[0]
    for n in range(len(h)):
        i=np.append(i, (j[int(l[n])])**0.5)
    min_i=min(i)
    k=0.5#0.5
    run=0
    i_mean=np.nanmean(i)
    while (min_i < (k*i_mean)) and run<30:
        run=run+1
        index_min = np.argmin(i)
        h=np.delete(h, index_min)
        l=np.delete(l, index_min)
        i=np.delete(i, index_min)
        min_i=min(i)
    g= r2r(h)
    run=0
    k=0.6#0.75
    run=0
    p=1.8
    min_g =min(g)
    g_mean=np.nanmean(g)
    while (min_g < k*g_mean) and run<60:
        run=run+1
        index_min = np.argmin(g)
        a=index_min +1
        b=index_min -1
        try:
            if g[b]+g[index_min] < (p*g_mean):
                h=np.delete(h,index_min)
                l=np.delete(l, index_min)
        except IndexError:
            pass
        try:
            if g[a]+g[index_min] < (p*g_mean):
                h=np.delete(h,a)
                l=np.delete(l, a)
        except IndexError:
            pass
        g=r2r(h)
        min_g=min(g)
    return l 



# identifies the peaks using the scipy peak finding function
#takes in the data as an array, the prominance of the data, the average distance to the next peak
def peaks(x,p,d):
    b = scipy.signal.find_peaks(x, prominence=p, distance=d, height=0)
    b = optimise_peaks(b[0]/300, x, b[0])
    return b



#finds the indicies of the two closest data points in two sorted arrays
def closest(a,b):#new algoithm of 2*n complexity
    #function defining the new 2*n complexity algorithm
    #takes the 2 datasets as inputs
    #returns index of min a, index of min b and stepcount respectively
    e = np.zeros_like(a)#array with closest difference for each index
    f= np.empty((len(a),2))#holds indices
    step_counter =0
    
    for i in range(len(a)):#iterates through array1
        if i == 0:#for the case of the first element begins with the first member 
            #of the second array and steps forward through the second array till its difference increase
            for j in range(len(b)-1):
                step_counter = step_counter+1
                current_diff = abs(a[i]-b[j])
                next_diff = abs(a[i] - b[j+1])
                if current_diff<next_diff:#checks if diff is increasing
                    #print(i,current_diff)
                    e[i] = current_diff#writes the difference and indices
                    f[i,0] = int(i)
                    f[i,1] = int(j)
                    break
                else:
                    e[i] = next_diff#if difference keeps decreasing the last element is saved
                    f[i,0] = i
                    f[i,1] = j+1
        if i>0 and i<len(a):
            guess_j = int(f[i-1,1])#starts checking at the location of the previous minima
            if guess_j == len(b)-1:#checks if the last minima was at the end of the second list as this will break the system to check for the previous and next difference
                guess_j = guess_j -1# if the last guess was indeed the last element then it sets it back by 1 to keep it from throwing
            guess_diff = a[i] - b[guess_j]#checks to see if the differnce at guess is more or less than the nearest neighbours 
            guess_next_diff = a[i] - b[guess_j+1]
            guess_prev_diff = a[i] - b[guess_j-1]
            if guess_diff < guess_next_diff and guess_diff <    guess_prev_diff: #if its less than both, it saves the guess as the correct one
                e[i] = guess_diff
                f[i,0] = i
                f[i,1] = guess_j
            elif guess_diff>guess_next_diff:#if difference is more than the next it steps forward from that point
                for j in range(guess_j,len(b)-1):#essentially the same as the i = 0 case just begins looking at guess index
                    step_counter = step_counter+1
                    current_diff = abs(a[i]-b[j])
                    next_diff = abs(a[i] - b[j+1])
                    if current_diff<next_diff:
                        #print(i,current_diff)
                        e[i] = current_diff
                        f[i,0] = i
                        f[i,1] = j
                        break
                    else:
                        e[i] = next_diff
                        f[i,0] = i
                        f[i,1] = j+1
            elif guess_diff>guess_prev_diff:#if the difference is more than the previos then it steps backward from that point
                for j in range(guess_j,1,-1):#same as the i = 0 case, just going backwards from guess to j =0
                    step_counter = step_counter+1
                    current_diff = abs(a[i]-b[j])
                    prev_diff = abs(a[i] - b[j-1])
                    if current_diff<prev_diff:
                        #print(i,current_diff)
                        e[i] = current_diff
                        f[i,0] = i
                        f[i,1] = j
                        break
                    else:
                        e[i] = prev_diff
                        f[i,0] = i
                        f[i,1] = j-1
    index_of_minima = np.argmin(e)
    a_value =f[index_of_minima,0]
    b_value = f[index_of_minima,1]
    return a_value,b_value


#function to make first peaks line up
# takes in the raw data, the data you are aligning it with and the manually found peaks
def align(raw, comp, p):
    raw_r_ind = np.asarray(scipy.signal.find_peaks(raw,prominence=p, distance=dif_dist, height=0)[0]) #d=200
    comp_r_ind = np.asarray(peaks(comp, dif_prom, dif_dist))
    i, j= closest(raw_r_ind, comp_r_ind)
    i= int(i)
    j= int(j)
    if raw_r_ind[i] < comp_r_ind[j]:
        dif= comp_r_ind[j]-raw_r_ind[i]
        del_ind = np.arange(dif)
        #print(del_ind)
        corrected = np.delete(comp, del_ind)
        comp = corrected
        
    elif comp_r_ind[j] < raw_r_ind[i]:
        dif= raw_r_ind[i]-comp_r_ind[j]
        del_ind = np.arange(dif)
        corrected = np.delete(raw, del_ind)
        raw = corrected
        
    return raw, comp


#R-peak re-identification
def r_peak_reasign(ind, raw, w):
    r=int(w)
    #print(len(ind), ind)
    if len(ind)>0:
        for i in range(len(ind)):
            try:
                a=int(ind[i])-r
                b=int(ind[i])+r
                seg=raw[a:b]
                peak_ind=scipy.signal.find_peaks(seg, distance=r)[0]
                if len(peak_ind)==0:
                    peak_ind = np.where(seg==max(seg))[0]
                elif len(peak_ind)>1:
                    peak_h=seg[peak_ind]
                    peak_ind=peak_ind[np.where(peak_h==max(peak_h))]
                else:
                    pass
                new_ind=peak_ind[0]+ind[i]-r
                ind[i]=new_ind
            except ValueError or TypeError:
                pass
    else: 
        pass
    return ind


#subtracting r peaks to get r-r values for smoothed and filtered data
# takes in the times of the R peaks
def r2r(a): 
    output= np.diff(a)
    return output


# finds the index of the value in an array closest to a given value
# takes in the comparisson value and the array
def closest_val (num, arr):
    indx=0
    arr2=arr.copy()
    #arr2=arr2[~np.isnan(arr2)]
    arr2=np.where(np.isnan(arr2),0,arr2)
    #print(arr2)
    if len(arr2)>0:
        curr = arr2[0]
        truth_val=True
        for val in arr2:
            if abs (num - val) < abs (num - curr):
                curr = val
                indx=np.where(arr2 == val)[0][0]
            else:
                pass
    else:
        truth_val=False
    #print(num,curr)
    return indx, truth_val


#preforms a five point first order differentiation oon the input data
def five_pd(z):
    b = np.array([2, 1, 0, -1, -2]) * (1/8) #96.90 34 69 98 99
    d5 = abs(np.convolve(z, b, mode='same'))
    return d5


#preforms a back propergation sum on the data
# takes in data (z) and number being summed (w)
def bc(z,w):
    window=np.ones(w)
    b=np.convolve(z,window, mode='same')
    return b


#Identifies T-wave peaks
#Takes in data (fil2) and r peak indicies (r_ind_s) 
def find_rt2(fil2, r_ind_s,u):
    f=300
    rr_ind=r2r(r_ind_s)
    rr=rr_ind/f

    fil3=fil2.copy()
    l=np.zeros(7)
    mc=np.zeros(7)
    
    rr_dif=np.diff(rr)
    #rr_dif=np.diff(r_ind_s) #???????????????????????????????
    rr1=[1]
    rr1=np.append(rr1, rr_dif)
    rr2=np.append(rr_dif, 1)
    
    for i in range(len(r_ind_s)-1):
        if rr2[i] <= 1.33*rr1[i] and rr1[i]+rr2[i]<=0.7*f :
            #reentry
            fil3[r_ind_s[i]-int(0.027*rr1[i]) : r_ind_s[i]+int(0.222*rr1[i])]=0
            l[0]=l[0]+1
            mc[0]=mc[0]+rr[i]
        elif rr2[i] <= 1.5*rr1[i] and rr1[i]+rr2[i]<=f :
            #interpolation
            fil3[r_ind_s[i]-int(0.027*rr1[i]) : r_ind_s[i]+int(0.25*rr1[i])]=0
            l[1]=l[1]+1
            mc[1]=mc[1]+rr[i]
        elif rr2[i] <= 1.76*rr1[i] and rr1[i]+rr2[i]<=1.8*f :
            #reset
            fil3[r_ind_s[i]-int(0.027*rr1[i]) : r_ind_s[i]+int(0.333*rr1[i])]=0 
            l[2]=l[2]+1
            mc[2]=mc[2]+rr[i]
        elif rr2[i] <= 1.35*rr1[i] and rr1[i]+rr2[i]<=2*f :
            #compensation
            fil3[r_ind_s[i]-int(0.027*rr1[i]) : r_ind_s[i]+int(0.278*rr1[i])]=0
            l[3]=l[3]+1
            mc[3]=mc[3]+rr[i]
        elif rr2[i] >= 0.9*f and rr1[i]>=0.9*f :
            #bigeminy
            fil3[r_ind_s[i]-int(0.027*rr1[i]) : r_ind_s[i]+int(0.111*rr1[i])]=0
            l[4]=l[4]+1
            mc[4]=mc[4]+rr[i]
        elif rr2[i]>=1.2*f:
            #trigeminy
            fil3[r_ind_s[i]-int(0.027*rr1[i]) : r_ind_s[i]+int(0.139*rr1[i])]=0
            l[5]=l[5]+1
            mc[5]=mc[5]+rr[i]
        else:
            #normal
            fil3[r_ind_s[i]-int(0.027*rr1[i]) : r_ind_s[i]+int(0.25*rr1[i])]=0 
            l[6]=l[6]+1
            mc[6]=mc[6]+rr[i]
    mc=mc/(l)
    k=max(mc)
    #print(k)
    w1=int(0.07*k*f) #0.07
    w2=int(0.14*k*f) #0.14
    #print(k)
    mapeak=np.convolve(fil3, np.ones(w1)/w1, mode='same')
    matwave=np.convolve(fil3, np.ones(w2)/w2, mode='same')
    
    t_ind=[]
    t_ind_plot=[]
    tnum=0
    for i in range(len(r_ind_s)-1):
        seg=fil3[r_ind_s[i]:r_ind_s[i+1]]
        mapeaki=mapeak[r_ind_s[i]:r_ind_s[i+1]]
        matwavei=matwave[r_ind_s[i]:r_ind_s[i+1]]
        block_ind_i=np.where(mapeaki>matwavei)[0]
        dmin=int((0.23*rr_ind[i]))#int(51*0.5) #int((0.17*rr[i])*f)
        dmax=int((0.8*rr_ind[i]))#int(200*0.5) #int(240*0.5) #int((0.8*rr[i])*f)
        block_ind=block_ind_i[np.where(dmin<block_ind_i)[0]]
        block_ind=block_ind[np.where(block_ind<dmax)[0]]
        #print(np.split(block_ind,np.where(np.diff(block_ind) > 1)[0]))
        #print(np.array(np.split(block_ind,np.where(np.diff(block_ind) > 1)[0]),dtype=object))
        if len(block_ind)>1:
            block_ind =np.split(block_ind, np.array(np.where(np.diff(block_ind) > 1)[0]) + 1)
        else:
            block_ind = np.array(np.split(block_ind, np.array(np.where(np.diff(block_ind) > 1)[0]) + 1))
        #print('space')
        if len(block_ind)>2 :
            block_dist1=block_ind[1][0]-block_ind[0][len(block_ind[0])-1]
            block_dist2=block_ind[2][0]-block_ind[1][len(block_ind[1])-1]
            if block_dist1<20:
                if block_dist2<20:
                    block_h1=seg[block_ind[0]]
                    block_h2=seg[block_ind[1]]
                    block_h3=seg[block_ind[2]]
                    if max(block_h1)>(max(block_h2)and max(block_h3)):
                        block_ind=block_ind[0]
                    elif max(block_h3)>(max(block_h2)and max(block_h1)):
                        block_ind=block_ind[2]
                    else:
                        block_ind=block_ind[1]
                else:
                    block_h1=seg[block_ind[0]]
                    block_h2=seg[block_ind[1]]
                    if max(block_h1)>max(block_h2):
                        block_ind=block_ind[0]
                    else:
                        block_ind=block_ind[1]
            else:
                block_ind=block_ind[0]
        elif len(block_ind)==2:
            block_dist=block_ind[1][0]-block_ind[0][len(block_ind[0])-1]
            if block_dist<15:
                block_h1=seg[block_ind[0]]
                block_h2=seg[block_ind[1]]
                if max(block_h1)>max(block_h2):
                    block_ind=block_ind[0]
                else:
                    block_ind=block_ind[1]
            else:
                block_ind=block_ind[0]
        else:
            block_ind=block_ind[0]
        block_ind=block_ind.astype(int)
        block_h=seg[block_ind]
        try:
            peak_ind = np.where(block_h==max(block_h))[0][0]
            t_peak_plot=block_ind[peak_ind]
            t_peak=t_peak_plot+r_ind_s[i]
            #t_time=t_peak/f
            tnum=tnum+1
        except ValueError:
            print('No T wave detected in data set', u, 'interval', i)
            t_peak=float('Nan')
            t_peak_plot=float('Nan')
        
        t_ind=np.append(t_ind, t_peak)
        t_ind_plot=np.append(t_ind_plot, t_peak_plot)
        
    rt_times=[]
    rt_rr_frac=[]
    hr=60/(np.mean(rr_ind)/f)
    for m in range(len(r_ind_s)-1):
        try:
            rt_time=(t_ind[m]-r_ind_s[m])/f
            rt_times=np.append(rt_times, rt_time)
            if 55<hr<65:
                frac=rt_time
            elif 65<=hr<=100:
                frac=(t_ind[m]-r_ind_s[m])/(f*((rr_ind[m]/f)**(1/2)))
            else:
                frac = (t_ind[m]-r_ind_s[m])/(f*((rr_ind[m]/f)**(1/3)))
            rt_rr_frac=np.append(rt_rr_frac, frac)
        except IndexError:
            rt_times = np.append(rt_times, float('NaN'))
            rt_rr_frac = np.append(rt_rr_frac, float('NaN'))   
    #print('data set',u, 'number of segments=',len(segmax), 'number of T waves=', tnum)
    tfound=(tnum/len(rr))*100
    return t_ind, rt_times, rt_rr_frac, t_ind_plot, tfound


#calculates the skew of an array rr
def skew(rr):
    x=rr[:len(rr)-1]
    y=rr[1:]
    try:
        dskew=stats.skew(rr, bias=False)
    except AttributeError:
        dskew=float('Nan')
    
    cood = np.asarray(np.c_[x, y])
    p=np.asarray([[0,0],[3,3]])
    dis_r2r=[]
    for i in range(len(x)):
        dis = np.asarray(np.linalg.norm(np.cross((p[1,:]-p[0,:]),(p[0,:]-cood[i,:]))/np.linalg.norm(p[1,:]-p[0,:])))
        dis_r2r=np.append(dis_r2r, dis)
    
    return dskew, dis_r2r



#works out the varience of a set of values
#takes in values (r) and degrees of freedom (ddof)
def variance(r, ddof=0):
    n= len(r)
    mean = sum(r) / n
    return sum ((x-mean)**2 for x in r) / (n-ddof)


# works out standard deviation from the varience
#takes in a set of values (r)
def std(r):
    try:
        var= variance(r)
    except ZeroDivisionError:
        var=float('Nan')
    std_dev = np.sqrt(var)
    return std_dev


# functions for the bandpass filter
from scipy.signal import butter, lfilter
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 1.0* fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def __butter_bandstop( lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], btype="bandpass", output="sos")
        return sos
    
def __butter_bandstop_filter( data, lowcut, highcut, fs, order):
        sos = __butter_bandstop(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data).astype(np.float32)
        return y

#check to see if data is too noisy to use
def noise(x):
    x[:600]=0
    p=peaks(x, 0, dif_dist)
    rr_time=r2r(p)/300
    av_hr=np.nanmean(60/rr_time)
    ratio=max(rr_time)/min(rr_time)
    #print ('data set', u, 'ratio =', ratio)
    correlation=0
    if 40<=av_hr<=180 and max(rr_time)<=3 and ratio<3.2:
        med=np.nanmedian(r2r(p))
        remainder= ((int(med)%2))
        segments=np.empty((int(med)-remainder,0))
        for i in range(len(p)-1):
            ind1=p[i]-int(med/2)
            ind2=p[i]+int(med/2)
            seg=x[ind1 : ind2]
            segments=np.c_[segments,seg]
        av_seg=np.mean(segments,axis=1)
        corr=[]
        for j in range(len(segments[0])):
            corrj=scipy.stats.pearsonr(av_seg, segments[:,j])
            corr=np.append(corr,corrj)
        correlation=np.mean(corr)
    if correlation<0.4: #0.4
        noise=True
    else:
        noise=False
    return noise, correlation, ratio

def pvc_check(fil2, r_ind_s):
    f=300
    rr_ind=r2r(r_ind_s)
    rr=rr_ind/f
    rr_dif=np.diff(rr)
    rr1=[1]
    rr1=np.append(rr1, rr_dif)
    pvc=[False]
    areas=[]
    seg_end=[]
    seg_start=[]
    for i in range(len(r_ind_s)-1):
            seg=fil2[r_ind_s[i]-int(0.027*rr1[i]) : r_ind_s[i]+int(0.25*rr1[i])]
            seg_end1 = r_ind_s[i]+int(0.25*rr1[i]) 
            seg_start1 = r_ind_s[i]-int(0.027*rr1[i])
            a=trapz(seg, dx=0.01)
            areas=np.append(areas, a)
            med= np.median(areas)
            if a > 1.3*med:
                #print('PVC detected in data set', u, 'segment', i)
                truth=True
            else:
                truth=False
            seg_end=np.append(seg_end, seg_end1)
            seg_start=np.append(seg_start,seg_start1)
            pvc=np.append(pvc, truth)
    percent=sum(pvc)/len(pvc)
    #print(percent)
    if percent>0.75:
        pvc[:]=False
    return pvc, seg_end, seg_start

#p wave detection
def path_check(fil2, r_ind_s,heart_rate, std_r2r, t_ind,u):
    #Detecting AFIB
    p2r_times=[]
    x=np.linspace(0,9000/300,9000)
    #plt.plot(x,fil2)
    if np.mean(heart_rate)>100 and std_r2r>0.12:
        print('Afib detected in data set', u)
        #p_waves=[float('Nan')]
        p_waves = np.empty((len(r_ind_s)-1))
        p_waves[:] = np.nan
        #Detecting PVC
        pvc, seg_end, seg_start = pvc_check(fil2, r_ind_s)
        
        for i in range(len(r_ind_s)-1):
            if pvc[i+1]==True:
                print('Anomaly detected in data set', u, 'segment', i)
                
    else:
        rr=np.diff(r_ind_s)
        p_waves=[]
        diss_p=[]
        pvc, seg_end, seg_start = pvc_check(fil2, r_ind_s)
        
        for i in range(len(r_ind_s)-1):
            if pvc[i+1]==True:
                print('Anomaly detected in data set', u, 'segment', i)
                p=float('Nan')
            else:
                seg=fil2[r_ind_s[i]+int(0.71*rr[i]) : r_ind_s[i+1]-int(0.07*rr[i])-int(3)]#18
                seg_start = r_ind_s[i]+int(0.71*rr[i])
                seg_end = r_ind_s[i+1]-int(0.07*rr[i])-int(3) #18
                #pind=scipy.signal.find_peaks(seg)[0] #, distance=600)[0]
                #p=np.where(seg==max(seg[pind]))[0][0]
                p=np.where(seg==max(seg))[0][0]
                #print('no pvc')
                #dissociated p wave finder
                if rr[i] > 1.6*rr[i-1] and rr[i] > int(480):
                    #print ('Anomaly found data set',u, 'segment', i)
                    try:
                        seg = fil2[int(t_ind[0, i-1]+int(60)) : int(p[i] -int(105))] #120
                        seg_start = int(t_ind[0, i-1]+int(60))
                        seg_end = int(p[i] -int(105))
                        #pind=scipy.signal.find_peaks(seg)[0] #, distance=600)[0]
                        #p=np.where(seg==max(seg[pind]))[0][0]
                        p=np.where(seg==max(seg))[0][0]
                        #print('dissociated1')
                    except IndexError or ValueError:
                        #print('maybe')
                        p=float('Nan')
                        #print('dissociated2')
                        #previous p wave dissociated criteria 
                if rr[i-1] > 1.6*rr[i-2] and rr[i-1] > int(480) and rr[i] > 0.8*rr[i-1]: 
                    #print ('Dissocited p wave found in segment', i)
                    try:
                        seg = fil2[int(t_ind[0, i-1]+int(60)) : int(p[i] -int(105))]
                        seg_start = int(t_ind[0, i-1]+int(60))
                        seg_end = int(p[i] -int(105))
                        
                        #pind=scipy.signal.find_peaks(seg)[0] #, distance=600)[0]
                        #p=np.where(seg==max(seg[pind]))[0][0]
                        p=np.where(seg==max(seg))[0][0]
                        #print('dissociated3')
                    except IndexError or ValueError:
                        #print('dissociated4')
                        p=float('Nan')
                        #print('breaking')
                
                #making sure only 1 p wave found
                #print(p)
                if p==float('Nan'):
                    h=[fil2[p]]
                    h2=[fil2[r_ind_s]]
                else:
                    h=[float('Nan')]
                    h2=[fil2[r_ind_s]]
                p=p+seg_start
                '''
                if len(p)>1:
                    p=p+ seg_start
                    mp=np.where(h==max(h),)[0]
                    p=mp
                elif len(p) == 1:
                    p=p+ seg_start
                else:   
                    p=float('Nan')
                    #print('breaking')
                '''
            #print(p)
            p_waves=np.append(p_waves, p)
            
            #p wave verification
            try:
                if p_waves[i] > t_ind[i-1] +int(200) and h[i]> 0.05*h2[i]:
                    pass
                else:
                    p_waves[i]=float('Nan')
            except IndexError:
                pass
            #plt.axvspan(seg_start/300, seg_end/300, color='red', alpha=0.5)
            #plt.axvline(x=r_ind_s[i]/300, color='green')
            #plt.axvline(x=p_waves[i-1]/300, linestyle='--', color='pink')
            #plt.axvline(x=m[i]/300, linestyle='--', color='purple')
            '''
            try:
                pr=(r_ind_s[i]-p_waves[i-1])/300
            except IndexError:
                pr=float('Nan')
            p2r_times=np.append(p2r_times, pr)
            '''
    #print(p_waves)
    #p2r_times=(r_ind_s[1:] - p_waves)/300
    return p_waves


 #accuracy function for two comparable arrays
#takes in the correct values (x) and the comparrison values (z)
def accuracy(x,z):
    if len(x)< len(z):
        b=len(x)
    else:
        b=len(z)  
    a=100-abs(100*(z[:b]-x[:b])/x[:b])
    return a 


#finds the p-peak accuracy, the number of found peaks that are skipped and what they are
#takes in the correct peaks and the comparison peaks, and e data set
def p_peak_acc(x,z):
    a=[]
    b= []
    #miss=0
    time_diff=0
    #print(x)
    #print(z)
    for n in range(len(x)-1):
        m, truth_val= closest_val(x[n], z)
        if truth_val==True:
            #print(m)
            #b= np.append(b, m)
            #p-peak accuracy
            err = 100 - ((abs(((z[m]-x[n])/x[n])*100))*(n+1))
            err2 = (abs(z[m]-x[n])*1000)/300
            if err < 75:
                err2 = 0 #float('Nan')
                #miss=miss+1
            else:
               b= np.append(b, m) 
            a= np.append(a,err)
            time_diff=time_diff+err2
        else:
            time_diff=np.nan
            a = np.empty((len(x)-1))
            a[:] = np.NaN
            b = np.empty((len(x)-1))
            b[:]=np.nan
    extra_n=len(b)-len(list(dict.fromkeys(b)))
    '''
    try:
        b = b.tolist()
    except AttributeError:
        pass
    for d in range(len(z)):
        try:
            b.index(d)
        except ValueError:
            extra_ind=np.append(extra_ind, d)
    '''
    miss_ind=[j for j in range(len(x)) if j not in b] 
    #print(miss_ind)       
    miss = len(miss_ind)
    #print(time_diff)
    return a, extra_n, miss, time_diff



#plot the graphs over time
def time_graph(date, hr, r2rsd, r2t, r2tsd, p2r, p2rsd,u):
    
    plt.figure(s)
    plt.clf()
    
    
    d=[]
    d=np.argwhere(np.isnan(p2r))
    date=np.delete(date,d)
    hr=np.delete(hr,d)
    r2rsd=np.delete(r2rsd,d)
    r2t=np.delete(r2t,d)
    r2tsd=np.delete(r2tsd,d)
    p2r=np.delete(p2r,d)
    p2rsd=np.delete(p2rsd,d)
    poly_date=np.linspace(0,1,len(date))
    #print(poly_date)
    poly_degree = 1
    hrsd= (r2rsd*((np.mean(hr))**2))/60
    
    
    d=[]
    #d=np.argwhere(date<'2022-03-03')
    
    
    date=np.delete(date,d)
    #print(date)
    hr=np.delete(hr,d)
    r2rsd=np.delete(r2rsd,d)
    r2t=np.delete(r2t,d)
    r2tsd=np.delete(r2tsd,d)
    p2r=np.delete(p2r,d)
    p2rsd=np.delete(p2rsd,d)
    #poly_date=np.linspace(0,1,len(date))
    poly_date = [x - date[0] for x in date]
    poly_date = [x.days for x in poly_date]
    poly_degree = 1
    hrsd= (r2rsd*((np.mean(hr))**2))/60
    
    axis00=plt.subplot(3,2,1)
    #axis00.plot_date(date, hr, label='Heart rate',linestyle='solid')
    coeffs_hr, hrerr= np.polyfit(poly_date, hr, poly_degree, w=1/hrsd, cov=True)
    poly_eqn_hr = np.poly1d(coeffs_hr)
    #print(coeffs_hr)
    poly_hr = poly_eqn_hr(poly_date)
    axis00.plot_date(date, hr, label='Heart rate', color='steelblue')
    axis00.plot_date(date, poly_hr, color='steelblue', linewidth=2, linestyle='-', marker='None')
    axis00.set(xlabel='Date and time', ylabel='Heart rate (bpm)')
    axis00.grid(True)
    axis00.axis('tight')
    axis00.legend(loc='upper left')
    axis00.axhspan(60,100, color='green', alpha=0.4)
    print('Gradient for HR =', coeffs_hr[0],'bpm', '+-',np.sqrt(hrerr[0][0]))
    ghr="{:.2f}".format(coeffs_hr[0])
    ghrerr="{:.2f}".format(np.sqrt(hrerr[0][0]))
    ghr1=str(ghr)+'\u00B1'+ str(ghrerr)
    #print(ghr1)
    
    
    '''
    axis01=plt.subplot(3,2,2, sharex=axis00)
    #axis01.plot_date(date, r2rsd, label='R-R interval standard deviation',linestyle='solid')
    coeffs_r2rsd = np.polyfit(poly_date, r2rsd, poly_degree)
    poly_eqn_r2rsd = np.poly1d(coeffs_r2rsd)
    poly_r2rsd = poly_eqn_r2rsd(poly_date)
    axis01.plot_date(date, r2rsd, label='R-R interval standard deviation', color='steelblue')
    axis01.plot_date(date, poly_r2rsd, color='steelblue', linewidth=2, linestyle='-', marker='None')
    axis01.set(xlabel='Date and time', ylabel='R-R interval standard deviation (s)')
    axis01.grid(True)
    axis01.axis('tight')
    axis01.legend(loc='upper left')
    axis01.axhspan(0,0.12, color='green', alpha=0.4)
    print('Gradient for RR SD =', coeffs_hr[0] )
    '''
    
    axis01=plt.subplot(3,2,2, sharex=axis00)
    #axis01.plot_date(date, r2rsd, label='R-R interval standard deviation',linestyle='solid')
    coeffs_hrsd = np.polyfit(poly_date, hrsd, poly_degree)
    poly_eqn_hrsd = np.poly1d(coeffs_hrsd)
    poly_hrsd = poly_eqn_hrsd(poly_date)
    axis01.plot_date(date, hrsd, label='HR interval standard deviation', color='steelblue')
    axis01.plot_date(date, poly_hrsd, color='steelblue', linewidth=2, linestyle='-', marker='None')
    axis01.set(xlabel='Date and time', ylabel='HR interval standard deviation (bpm)')
    axis01.grid(True)
    axis01.axis('tight')
    axis01.legend(loc='upper left')
    axis01.axhspan(0,(0.12*((np.mean(hr))**2))/60, color='green', alpha=0.4)
    print('Gradient for HR SD =', coeffs_hrsd[0], 'bpm' )
    ghrsd="{:.2f}".format(coeffs_hrsd[0])
    
    
    axis10=plt.subplot(3,2,3, sharex=axis00)
    #axis10.plot_date(date, r2t, label='R-T interval',linestyle='solid')
    coeffs_r2t, rterr = np.polyfit(poly_date, r2t, poly_degree, w=1/r2tsd, cov=True)
    poly_eqn_r2t = np.poly1d(coeffs_r2t)
    poly_r2t = poly_eqn_r2t(poly_date)
    axis10.plot_date(date, r2t, label='R-T interval', color='steelblue')
    axis10.plot_date(date, poly_r2t, color='steelblue', linewidth=2, linestyle='-', marker='None')
    axis10.set(xlabel='Date and time', ylabel='R-T interval (s)')
    axis10.grid(True)
    axis10.axis('tight')
    axis10.legend(loc='upper left')
    axis10.axhspan(0.225,0.36, color='green', alpha=0.4)
    print('Gradient for RT =', coeffs_r2t[0]*1000, 'ms', '+-' ,np.sqrt(rterr[0][0])*1000 )
    grt="{:.2f}".format(coeffs_r2t[0]*1000)
    grterr="{:.2f}".format(np.sqrt(rterr[0][0])*1000)
    grt1=str(grt)+'\u00B1'+ str(grterr)
    
    
    axis11=plt.subplot(3,2,4, sharex=axis00)
    #axis11.plot_date(date, r2tsd, label='R-T interval standard deviation',linestyle='solid')
    coeffs_r2tsd = np.polyfit(poly_date, r2tsd, poly_degree)
    poly_eqn_r2tsd = np.poly1d(coeffs_r2tsd)
    poly_r2tsd = poly_eqn_r2tsd(poly_date)
    axis11.plot_date(date, r2tsd, label='R-T interval standard deviation', color='steelblue')
    axis11.plot_date(date, poly_r2tsd, color='steelblue', linewidth=2, linestyle='-', marker='None')
    axis11.set(xlabel='Date and time', ylabel='R-T interval standard deviation (s)')
    axis11.grid(True)
    axis11.axis('tight')
    axis11.legend(loc='upper left')
    print('Gradient for RT SD =', coeffs_r2tsd[0]*1000, 'ms' )
    grtsd="{:.2f}".format(coeffs_r2tsd[0]*1000)
    
    
    
    axis20=plt.subplot(3,2,5, sharex=axis00)
    #axis20.plot_date(date, p2r, label='P-R interval',linestyle='solid')
    coeffs_p2r, prerr = np.polyfit(poly_date, p2r, poly_degree, w=1/p2rsd, cov=True)
    poly_eqn_p2r = np.poly1d(coeffs_p2r)
    poly_p2r = poly_eqn_p2r(poly_date)
    axis20.plot_date(date, p2r, label='P-R interval', color='steelblue')
    axis20.plot_date(date, poly_p2r, color='steelblue', linewidth=2, linestyle='-', marker='None')
    axis20.set(xlabel='Date and time', ylabel='P-R interval (s)')
    axis20.grid(True)
    axis20.axis('tight')
    axis20.legend(loc='upper left')
    axis20.axhspan(0.1,0.23, color='green', alpha=0.4)
    print('Gradient for PR =', coeffs_p2r[0]*1000, 'ms', '+-',np.sqrt(prerr[0][0])*1000 )
    gpr="{:.2f}".format(coeffs_p2r[0]*1000)
    gprerr="{:.2f}".format(np.sqrt(prerr[0][0])*1000)
    gpr1=str(gpr)+'\u00B1'+ str(gprerr)
    
    
    axis21=plt.subplot(3,2,6, sharex=axis00)
    #axis21.plot_date(date, p2rsd, label='P-R interval standard deviation',linestyle='solid')
    coeffs_p2rsd = np.polyfit(poly_date, p2rsd, poly_degree)
    poly_eqn_p2rsd = np.poly1d(coeffs_p2rsd)
    poly_p2rsd = poly_eqn_p2rsd(poly_date)
    axis21.plot_date(date, p2rsd, label='P-R interval standard deviation', color='steelblue')
    axis21.plot_date(date, poly_p2rsd, color='steelblue', linewidth=2, linestyle='-', marker='None')
    axis21.set(xlabel='Date and time', ylabel='P-R interval standard deviation (s)')
    axis21.grid(True)
    axis21.axis('tight')
    axis21.legend(loc='upper left')
    print('Gradient for PR SD =', coeffs_p2rsd[0]*1000, 'ms' )
    gprsd="{:.2f}".format(coeffs_p2rsd[0]*1000)
    
    return ghr1, ghrsd, grt1, grtsd, gpr1, gprsd
    
    
    
    
    
#Preforms the data analysis
#takes in raw ecg data (raw), the data identifier (u), times of the manual peaks (m) and what graphs to plot (plot)
#plot takes a string
def run(raw, u, plot):
    import numpy as np
    
    noisy, correlation, ratio = noise(raw)
    if  noisy==False:
        
        #processes the data
        # Sample rate and desired cutoff frequencies (in Hz).
        fs = 300.0
        lowcut = 1.3 #1.3
        highcut = 45#45
        # set the length of the ecg
        T = 9000/300
        nsamples = int(T * fs)
        t = np.linspace(0, T, nsamples, endpoint=False)
        #desired array size
        raw= length(raw,9000)
        #filters the raw data 
        fil = butter_bandpass_filter(raw, lowcut, highcut, fs, order=5) # order=6
        #fil = __butter_bandstop_filter( raw, lowcut, highcut, fs, order=1)
        
        #fil=raw.copy()
        fil2=fil.copy()
        
        fil= five_pd(fil2)
        fil=bc(fil,30) #30
        #m=r_peak_reasign(m, raw, w=10)
        #m = m[(m > 600)]
        #m=m.copy()-12
        raw = length(raw,9000)
        fil = length(fil,9000)
        raw,fil2 = align(raw,fil2, p=100)
        fil2,fil = align(fil2, fil, p=60)
        fil = length(fil,9000)
        fil2 = length(fil2,9000)
        raw = length(raw,9000)
        t2=t[:len(fil)]
        fil=fil**2
        fil[:600]=0
        fil2[:600]=0
        raw[:600]=0
        #plt.plot(t, raw, color='orange', alpha=0.25)
        #plt.plot(t,fil2, color='red', alpha=0.25)
        
        #plt.show
        
        # finds the r peaks of the smoothed filtered signal fil
        r_ind_s = peaks(fil, dif_prom, dif_dist) #indicies
        r_ind_s = r_peak_reasign(r_ind_s, raw, 40)
        r_times=r_ind_s.copy()/300 #times
        
        #remove duplicates
        dup=np.unique(r_times)
        r_times=dup
        
        
        
        #subtracting r peaks to get r-r values for smoothed and filtered data
        r2r_time = r2r(r_times)
        r2r_skew, r2r_dis = skew(r2r_time)
        std_r2r = std(r2r_time)
        
        #finding the heart rate
        heart_rate=60/r2r_time
        
        # find rt
        t_ind, r2t_times, r2t_times_corrected, t_ind_plot, t_found = find_rt2(fil2,r_ind_s,u)
        r2t_times_nan = r2t_times[~np.isnan(r2t_times)]
        r2t_skew, r2t_dis = skew(r2t_times_nan)
        std_r2t = std(r2t_times_nan)
        
        #pathology check and P-wave identification
        p_ind=path_check(fil2, r_ind_s,heart_rate, std_r2r, t_ind,u)
        p_ind = r_peak_reasign(p_ind, raw, 30)
        p2r_times=(r_ind_s[1:]-p_ind[:])/300
        p2r_times_nan = p2r_times[~np.isnan(p2r_times)]
        std_p2r=std(p2r_times_nan)
        
        
        if plot =='ecg':
            plt.figure(u+1)
            plt.clf()
            #plots the raw signal, shifted in time to match the smooth filtered signal
            plt.plot(t, raw, label='Noisy signal', linewidth=2) 
            #labels the graph
            plt.title('ECG (data set %g)' % u)
            plt.xlabel('time (s)')
            plt.ylabel('voltage (micro Volts)')
            plt.grid(True)
            plt.axis('tight')
            #plots the r peaks of the smoothed filtered signal z
            plt.axvline(x=r_times[0], color='b', linestyle='--', linewidth=1, label='code identified peaks')
            for xc in r_times[1:]:
                plt.axvline(x=xc, color='b', linestyle='--', linewidth=1)
            plt.legend(loc='upper left')
            
        elif plot =='fil':
            plt.figure(u+1)
            plt.clf()
            #plots the raw signal, shifted in time to match the smooth filtered signal
            #plots the filtered signal, shifted in time to match the smooth filtered signal
            plt.plot(t2, fil2, label='Filtered signal', linewidth=2)
            #labels the graph
            plt.title('Filtered ECG (data set %g)' % u)
            plt.xlabel('time (s)')
            plt.ylabel('voltage (micro volts squared)')
            plt.grid(True)
            plt.axis('tight')
            #plots the r peaks of the smoothed filtered signal z
            plt.axvline(x=r_times[0], color='b', linestyle='--', linewidth=1, label='code identified peaks')
            for xc in r_times[1:]:
                plt.axvline(x=xc, color='b', linestyle='--', linewidth=1)
            plt.legend(loc='upper left')
            
        elif plot =='seg':
            plt.figure(u+1)
            plt.clf()
            #segmentation graph
            t = np.linspace(0, (r_ind_s[1]-r_ind_s[0])/300, r_ind_s[1]-r_ind_s[0], endpoint=False)
            plt.plot(t, fil2[r_ind_s[0]:r_ind_s[1]], linewidth=0.5, label='Segmented data')
            for n in range(1,len(r_ind_s)-1):
                t = np.linspace(0, (r_ind_s[n+1]-r_ind_s[n])/300, r_ind_s[n+1]-r_ind_s[n], endpoint=False)
                plt.plot(t, fil2[r_ind_s[n]:r_ind_s[n+1]], linewidth=0.5)
            t_times_seg = (t_ind_plot)/300
            plt.axvline(x=t_times_seg[0], color='grey', linestyle='--', linewidth=1, label='T peaks')
            for xt in t_times_seg[1:]:
                plt.axvline(x=xt, color='grey', linestyle='--', linewidth=1)
            p_ind_plot=p_ind-r_ind_s[:len(p_ind)]
            p_times_seg = (p_ind_plot)/300
            plt.axvline(x=p_times_seg[0], color='pink', linestyle='--', linewidth=1, label='P peaks')
            for xp in p_times_seg[1:]:
                plt.axvline(x=xp, color='pink', linestyle='--', linewidth=1)
            plt.title('Segmentation and T waves (data set %g)' % u)
            plt.xlabel('time (s)')
            plt.ylabel('Filtered ECG (micro volts)')
            plt.grid(True)
            plt.axis('tight')
            plt.legend(loc='upper left')
            
        elif plot =='rrd':
            plt.figure(u+1)
            plt.clf()
            x_dis=np.arange(len(r2r_dis))*(30/len(r2r_dis))
            plt.scatter(x_dis, r2r_dis, linewidth=1, label='R-R time displacement')
            plt.title('R-R time displacement (data set %g)' % u)
            plt.xlabel('ECG time (s)')
            plt.ylabel('Displacement time (s)')
            plt.grid(True)
            plt.axis('tight')
            plt.legend(loc='upper left')
            
        elif plot =='rtd':
            plt.figure(u+1)
            plt.clf()
            x_dis=np.arange(len(r2t_dis))*(30/len(r2t_dis))
            plt.scatter(x_dis, r2t_dis, linewidth=1, label='R- time displacement')
            plt.title('R-T time displacement (data set %g)' % u)
            plt.xlabel('ECG time (s)')
            plt.ylabel('Displacement time (s)')
            plt.grid(True)
            plt.axis('tight')
            plt.legend(loc='upper left')
            
        elif plot =='hr':
            plt.figure(u+1)
            plt.clf()
            #heart rate graph
            x_rate_ind=np.arange(len(heart_rate))*(30/len(heart_rate))
            plt.plot(x_rate_ind, heart_rate, linewidth=2, label='bpm')
            plt.title('Heart rate (data set %g)' % u)
            plt.xlabel('time (s)')
            plt.ylabel('Heart rate (bpm)')
            plt.grid(True)
            plt.axis('tight')
            plt.legend(loc='upper left')
            
        elif plot =='rrp':
            plt.figure(u+1)
            plt.clf()
            plt.plot(np.arange(0,3), np.arange(0,3), linewidth=1, color = 'grey', linestyle = '--')
            plt.scatter(r2r_time[:len(r2r_time)-1], r2r_time[1:], linewidth=0.5, label='R-R phase function' )
            plt.title('R-R phase (data set %g)' % u)
            plt.xlabel('R-R[n] (s)')
            plt.ylabel('R-R[n+1] (s)')
            plt.grid(True)
            plt.axis('square')
            plt.legend(loc='upper left')
            
        elif plot =='rtp':
            plt.figure(u+1)
            plt.clf()
            plt.plot(np.arange(0,1,0.1), np.arange(0,1, 0.1), linewidth=1, color = 'grey', linestyle = '--')
            plt.scatter(r2t_times_nan[:len(r2t_times_nan)-1], r2t_times_nan[1:], linewidth=0.5, label='R-T phase function' )
            plt.title('R-T phase (data set %g' % u)
            plt.xlabel('R-T[n] (s)')
            plt.ylabel('R-T[n+1] (s)')
            plt.grid(True)
            plt.axis('square')
            plt.legend(loc='upper left')
        else:
            pass
        return raw, fil, r_times, r2r_time, t_ind, r2t_times, r2t_times_corrected, heart_rate, r2r_skew, r2t_skew, r2r_dis, r2t_dis, std_r2r, std_r2t, p_ind, p2r_times, std_p2r
    else:
        #print('data set', u, 'is too  noisy')#', correlation =', correlation, ', ratio =', ratio)
        raw = fil = r_times = r2r_time = t_ind = r2t_times = r2t_times_corrected = heart_rate = r2r_skew = r2t_skew = r2r_dis = r2t_dis = std_r2r = std_r2t = p_ind = p2r_times = std_p2r = [float('Nan')] 
        return raw, fil, r_times, r2r_time, t_ind, r2t_times, r2t_times_corrected, heart_rate, r2r_skew, r2t_skew, r2r_dis, r2t_dis, std_r2r, std_r2t, p_ind, p2r_times, std_p2r



def patient_output(num,patient=True):
    #opens the data file and converts it to a list of lists
    # if num<10:
    #     with open('/data/t/smartWatch/patients/completeData/patData/record0{}/signal.csv'.format(num),newline='') as f:
    #         reader = csv.reader(f)
    #         data = list(reader)
    # else:
    if patient:
        with open(f'/data/t/smartWatch/patients/completeData/patData/record{num}/signal.csv',newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
    else:
        with open(f'/data/t/smartWatch/patients/volunteerData/{num}/signal.csv',newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
    rows=sort(data)

  
    
    
    #print ('Outputs for patient', num)   
    #calls the functions to extract and smooth the data for all lists in the csv
    n=0
    sam_fq=300
    raw_data=np.empty((9000,0))
    time_data=np.empty(())
    time=9000/300
    interval=1/300
    
    while n<len(rows):
        v_ex, t_ex= extract(rows, n)
        v=length(v_ex,9000)
        raw_data= np.c_[raw_data, v]
        time_data=np.append(time_data,t_ex)
        n=n+1
    time_data=time_data[1:]
    
    #orders the ECGs by time
    raw_data=np.transpose(raw_data)
    time_data1, raw_data= (np.array(t) for t in zip(*sorted(zip(time_data, raw_data))))
    
    time_data=[x.date() for x in time_data]
    time_data=pd.to_datetime(time_data)
    #time_data = time_data[~(time_data < '2022-03-03')]
    time_data = [x - time_data[0] for x in time_data]
    time_data = [x.days for x in time_data]
    time_dif=np.diff(time_data)
    if time_dif[0] > 1 :
        time_data=time_data[1:]
        time_data=np.array(time_data)-float(time_data[0])
    else:
        pass
    raw_data=raw_data[len(raw_data)-len(time_data):]
    time_data1=time_data1[len(time_data1)-len(time_data):]
    #print(raw_data)
    #print(len(time_data),len(raw_data))
    raw_data=np.transpose(raw_data)
    
    
    #empty arrays to put the output data in
    raw_out=np.empty((9000,0)) #raw out put data
    fil_out=np.empty((9000,0)) #filtered out put data
    fil_r_times=np.empty((100,0)) #R-peaks found from the filtered data
    fil_r2r= np.empty((100,0)) #R-R times found from the filtered data
    fil_t_ind=np.empty((100,0)) #the indicies of where the T wave peaks are in the filtered data
    fil_r2t_times=np.empty((100,0)) #the R-T interval for filtered data
    fil_r2t_times_corrected=np.empty((100,0)) #the corrected R-T interval for the filtered data
    fil_heart_rate=np.empty((100,0)) #heart rate worked out over time from the R-R times
    fil_r2r_skew=np.empty((1,0)) #skew on the R-R data, assuming normally distributed, tells us if it leans positive or negitive
    fil_r2t_skew = np.empty((1,0)) #skew on the R-t data, assuming normally distributed, tells us if it leans positive or negitive
    fil_r2r_dis=np.empty((100,0)) #distance from y=x on the R-R phase graph
    fil_r2t_dis=np.empty((100,0)) #distance from y=x on the R-T phase graph
    fil_std_r2r=np.empty((1,0)) #standard deviation of R-R times
    fil_std_r2t=np.empty((1,0)) #standard deviation of R-T times
    fil_p_ind=np.empty((100,0)) #the indicies of where the P wave peaks are in the filtered data
    fil_p2r_times=np.empty((100,0))
    fil_std_p2r=np.empty((1,0))
    fil_time_data=[]  
    
    
    
    #runs the filter function outputting the filtered data and the smoothed filtered data
    for u in range(0,len(raw_data[0,:])):
        
        fil_time_data=np.append(fil_time_data, time_data1[u])
        
        out1, out2, out4, out6, out14, out16, out18, out19, out20, out21, out24, out25, out30, out31, out34, out35, out36 = run(raw_data[:,u], u, plot='')
        
        #arary of the realigned raw data
        out1=length(out1,9000)
        raw_out= np.c_[raw_out, out1]
        
        #array of the realigned processed data
        out2=length(out2, 9000)
        fil_out= np.c_[fil_out, out2]
        
        #code identified aligned r-peaks
        out4=length(out4, 100)
        fil_r_times=np.c_[fil_r_times, out4]
        fil_r_times=np.where(fil_r_times!=0,fil_r_times,np.nan)
        
        #r-r times of the processed data
        out6= length(out6, 100)
        fil_r2r=np.c_[fil_r2r, out6]
        fil_r2r=np.where(fil_r2r!=0,fil_r2r,np.nan)
        
        out14=length(out14, 100)
        fil_t_ind=np.c_[fil_t_ind, out14]
        fil_t_ind=np.where(fil_t_ind!=0,fil_t_ind,np.nan)
        
        out16=length(out16, 100)
        fil_r2t_times=np.c_[fil_r2t_times, out16]
        fil_r2t_times=np.where(fil_r2t_times!=0,fil_r2t_times,np.nan)
        
        out18=length(out18, 100)
        fil_r2t_times_corrected=np.c_[fil_r2t_times_corrected, out18]
        fil_r2t_times_corrected=np.where(fil_r2t_times_corrected!=0,fil_r2t_times_corrected,np.nan)
        
        out19=length(out19, 100)
        fil_heart_rate=np.c_[fil_heart_rate, out19]
        fil_heart_rate=np.where(fil_heart_rate!=0,fil_heart_rate,np.nan)
        
        fil_r2r_skew=np.c_[fil_r2r_skew, out20]
        
        fil_r2t_skew = np.c_[fil_r2t_skew, out21]
        
        out24=length(out24, 100)
        fil_r2r_dis=np.c_[fil_r2r_dis, out24]
        fil_r2r_dis=np.where(fil_r2r_dis!=0,fil_r2r_dis,np.nan)
        
        out25=length(out25, 100)
        fil_r2t_dis=np.c_[fil_r2t_dis, out25]
        fil_r2t_dis=np.where(fil_r2t_dis!=0,fil_r2t_dis,np.nan)
        
        fil_std_r2r=np.append(fil_std_r2r, out30)
        
        fil_std_r2t=np.append(fil_std_r2t, out31)
        
        out34=length(out34, 100)
        fil_p_ind=np.c_[fil_p_ind, out34]
        fil_p_ind=np.where(fil_p_ind!=0,fil_p_ind,np.nan)
        
        out35=length(out35, 100)
        fil_p2r_times=np.c_[fil_p2r_times, out35]
        fil_p2r_times=np.where(fil_p2r_times!=0,fil_p2r_times,np.nan)
        
        fil_std_p2r=np.append(fil_std_p2r, out36)
        
    return fil_r2r,fil_time_data,fil_r_times
     
    print ('average R-T time:', np.nanmean(fil_r2t_times_corrected))  
    print ('Average heart rate:', np.nanmean(fil_heart_rate))
    print('Average r2r skew:', np.nanmean(fil_r2r_skew))
    print ('Average r2t skew:', np.nanmean(fil_r2t_skew))
    print('Average r2r distance from y=x:', np.nanmean(fil_r2r_dis))
    print(fil_r2r_dis)
    print ('Average r2t distance from y=x:', np.nanmean(fil_r2t_dis))
    print ('standard deviation of R-R interval:', np.nanmean(fil_std_r2r))
    print ('standard deviation of R-T interval:', np.nanmean(fil_std_r2t))   
    print ('standard deviation of P-R interval:', np.nanmean(fil_std_p2r)) 
      
    if  0.25 < np.nanmean(fil_r2t_times_corrected) < 0.36:
        print('QT interval in healthy range')
    else:
        print('QT interval not in healthy range')
    
    fil_av_heart_rate=[]
    fil_av_r2t_times=[]
    fil_av_p2r_times=[]
    for col in range(len(fil_heart_rate[0,:])):
        fil_av_heart_rate=np.append(fil_av_heart_rate, np.nanmean(fil_heart_rate[:,col]))
        fil_av_r2t_times=np.append(fil_av_r2t_times, np.nanmean(fil_r2t_times_corrected[:,col]))
        fil_av_p2r_times=np.append(fil_av_p2r_times, np.nanmean(fil_p2r_times[:,col]))
        
    print('max heart rate =', np.nanmax(fil_av_heart_rate))
    print('min heart rate =', np.nanmin(fil_av_heart_rate))     
    
    #Graphs over time
    ghr1, ghrsd, grt1, grtsd, gpr1, gprsd=time_graph(fil_time_data, fil_av_heart_rate, fil_std_r2r, fil_av_r2t_times, fil_std_r2t, fil_av_p2r_times, fil_std_p2r, u)
    lst=[ ghr1, ghrsd, grt1, grtsd, gpr1, gprsd]
    print('Altered PR =', np.nanmean(fil_av_p2r_times)-0.045+0.04)
    print('Altered QT =', np.nanmean(fil_av_r2t_times)+0.045+0.08)
      
    
    datasets_used=np.count_nonzero(~np.isnan(raw_out[0,:]))
    
    print('Number of data sets used =', datasets_used )
    print('')
    print('')
    return lst, datasets_used, len(raw_data)
# %% 
# global variables
dif_prom = 60#60 #prominance of the peak finding function
dif_dist = 120 #minimum seperation of peaks in the peak finding function   
table_data=[]
datasets_u=0
datasets_t=0
n_patients=0
table_col=[]
table_row=['HR (bpm)', 'HR sd (bpm)', 'RT (ms)', 'RT sd (ms)', 'PR (ms)', 'PR sd (ms)' ]

# %%
# calling the analysis 


# for s in list(range(24)) + list(range(25, 50)): #!!! 0-23,    !!!!
#     try:
#         lst, datasets_used, datasets_total = patient_output(s)
#         n_patients=n_patients+1
#         table_col.append('Patient %g' %s)
#         table_data.append(lst)
#         datasets_u=datasets_u+datasets_used
#         datasets_t=datasets_t+datasets_total
#     except FileNotFoundError:
#         #print(n)
#         pass

# # %%
# # printing/plotting the analysis
# table_data=np.array(table_data).T.tolist()
# table_col.append('Mean')
# for h in table_data:
#     val=[re.findall(r"[-+]?(?:\d*\.\d+|\d+)", z) for z in h]
#     if len(val[0]) > 1:
#         val1=[float(k[0]) for k in val]
#         err=[float(k[1]) for k in val]
#         #print(np.mean(val1))
#         meanval= "{:.2f}".format(np.mean(val1))
#         meanerr="{:.2f}".format(np.sqrt(sum(err*2))/n_patients)
#         mean=str(meanval)+'\u00B1'+ str(meanerr)
#     else:
#         val1=[float(k[0]) for k in val]
#         mean= "{:.2f}".format(np.mean(val1))
#     h.append(mean)

# print('percentage of datasets used =', (100*(datasets_u/datasets_t)))


# fig, ax = plt.subplots()
# fig.set_size_inches(18.5,10,5)
# fig.patch.set_visible(False)
# ax.axis('off')
# ax.axis('tight')
# the_tabel=ax.table(cellText=table_data, rowLabels=table_row, colLabels=table_col, loc='center', fontsize = 18) #rowLoc='right'
# fig.tight_layout(rect=(1,1,1,1)) #pad=2
# the_tabel.set_fontsize(18)
# the_tabel.scale(1,1.5)
# ax.set_title('Change in ECG features per day')
# plt.show()
# #plt.savefig("Gradients.png",bbox_inches="tight")#, pad_inches=0.5) #bbox_inches="tight"