

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import linregress
from MFDFA import MFDFA
from datetime import datetime
import csv
import sqlite3
import sys
from scipy.interpolate import UnivariateSpline

sys.path.append('/data/t/smartWatch/patients/completeData')
from patient_analysis3 import patient_output



def sortingHeartRate(number,
                     patient=True):
    #loads the heartrate data into an array from the file
    if patient:
        heartRate=(pd.read_csv(f'/data/t/smartWatch/patients/completeData/patData/record{number}/raw_hr_hr.csv',header=0))
    else:
        heartRate=(pd.read_csv(f'/data/t/smartWatch/patients/volunteerData/{number}/raw_hr_hr.csv',header=0))

    heartRate['start']=pd.to_datetime(heartRate['start'], format='ISO8601',utc=True) # converts the timestamps to datetime objects
    sortedData=heartRate.sort_values(by='start')
    return sortedData

def sortingActivityData(number,
                        patient=True):
    #loads activity data into array and returns the start and end times
    if patient:
        activities=(pd.read_csv(f'/data/t/smartWatch/patients/completeData/patData/record{number}/activities.csv',header=0))
    else:
        activities=(pd.read_csv(f'/data/t/smartWatch/patients/volunteerData/{number}/activities.csv',header=0))

    activities['from']=pd.to_datetime(activities['from'], format='ISO8601',utc=True) # converts the timestamps to datetime objects
    activities['to']=pd.to_datetime(activities['to'], format='ISO8601',utc=True)
    return activities['from'],activities['to']

def split_on_plus(data):
    return np.vstack(np.char.split(data,'+'))

def split_on_T(data):
    return np.vstack(np.char.split(data,'T'))

def split_on_dash(data):
    return np.vstack(np.char.split(data,'-'))

def split_on_colon(data):
    return np.vstack(np.char.split(data,':'))

def only_yearAndmonth(data):
    return np.vstack(np.array([d[:7] for d in data]))

def months_calc(data,number,time_index):
    avg_hr_per_month=[] # list to store the average heart rate for each month
    # finds the unique months in the data and returns them
    months=np.unique(time_index.month) # finds the unique months in the data
    for m in months:
        mask=time_index.month==m # creates a mask for the current month
        month_data=data[mask]
        month_x=month_data['start']
        month_y = month_data['value']
        plt.title('Heart rate for month {}'.format(m))
        plt.plot(month_x,month_y,label='HR data') # plots the heart rate data for this month
        plt.xlabel('Date')
        plt.ylabel('Heart rate [bpm]')
        plt.tick_params(axis='x',labelrotation=45,length=0.1)
        plt.tight_layout()
        
        plt.legend()
        plt.show()
        #Path("/data/t/smartWatch/patients/completeData/DamianInternshipFiles/heartRateRecord{}".format(number)).mkdir(exist_ok=True) # creating new directory
        plt.savefig('/data/t/smartWatch/patients/completeData/DamianInternshipFiles/heartRateRecord{}/month-{}'.format(number,m))
        plt.close()
        avg_hr_per_month.append(np.average(month_y)) # averages the hr for that month
    

    return avg_hr_per_month

def week_calc(data,number,time_index):
    # finds the unique weeks in the data
    avg_hr_weekly=[]
    weeks=np.unique(time_index.isocalendar().week) # finds the unique weeks in the data
    print(weeks)
    print(time_index.to_series().dt.strftime('%G-W%V').unique())
    for w in weeks:
        mask=(time_index.isocalendar().week==w).to_numpy() # creates a mask for the current week
        
        
        week_data=data[mask]
        week_x=week_data['start']
        print(w)
        week_y = week_data['value']
        avg_hr_weekly.append(np.average(week_y)) # averages the hr for that weeks
        plt.title('Heart rate for week {}'.format(w))
        plt.plot(week_x,week_y,label='HR data') # plots the heart rate data for this week
        plt.xlabel('Date')
        plt.ylabel('Heart rate [bpm]')
        plt.tick_params(axis='x',labelrotation=45,length=0.1)
        plt.tight_layout()
        
        plt.legend()
        plt.show()
        #Path("/data/t/smartWatch/patients/completeData/DamianInternshipFiles/heartRateRecord{}".format(number)).mkdir(exist_ok=True) # creating new directory
        plt.savefig('/data/t/smartWatch/patients/completeData/DamianInternshipFiles/heartRateRecord{}/week-{}'.format(number,w))
        plt.close()
    
    return time_index.to_series().dt.strftime('%G-W%V').unique(),avg_hr_weekly

def active_days_calc(data,number,time_index,patient):
    avg_hr_active_days=[] # list to store the average heart rate for each day with activity
    normalised_time_index=time_index.normalize() # normalises the time index to remove the time component
    start,end= sortingActivityData(number,patient=patient) # brings in activity data
    start_time_index=pd.DatetimeIndex(start).normalize() # ensures the activity start times are in datetime format
    active_dates= start_time_index.to_series().dt.strftime('%Y-%m-%d').unique() # finds all unique dates activities were done on
    for day in active_dates: # loops through the days activity was done on
        mask=normalised_time_index==day   
        day_data=data[mask]
        plt.title('Heart rate on  day with activity: {}'.format(day))
        day_x= day_data['start']
        day_y = day_data['value']
        plt.plot(day_x,day_y,label='HR data')
        plt.xlabel('Data')
        plt.ylabel('Heart rate [bpm]')
        avg_hr_active_days.append(np.average(day_y))
        day_mask=start_time_index==day # creates a mask for the current day
        active_starts=start[day_mask] # generates the datetime objects for the activities done on the current day
        active_ends=end[day_mask]
        for i in active_starts:
            plt.axvline(pd.to_datetime(i, format='ISO8601',utc=True),color='red')
        for j in active_ends:
            plt.axvline(pd.to_datetime(j, format='ISO8601',utc=True),color='red')
        plt.tick_params(axis='x',labelrotation=90,length=0.1)
        plt.tight_layout()
        plt.legend()
        plt.show()
        plt.savefig('/data/t/smartWatch/patients/completeData/DamianInternshipFiles/heartRateRecord{}/{}'.format(number,day))
        plt.close()
    return avg_hr_active_days,active_dates

def total_timespan(data,number):
    time_y = data['value']  # extracts the heart rate values from the data
    time_x=data['start']
    plt.title('Heart rate over study')
    plt.plot(time_x,time_y,label='HR data')
    plt.xlabel('Date')
    plt.ylabel('Heart rate [bpm]')
    plt.axhline(np.average(time_y),color='red')
    plt.tick_params(axis='x',labelrotation=90,length=0.1)
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.savefig('/data/t/smartWatch/patients/completeData/DamianInternshipFiles/heartRateRecord{}/Full'.format(number))
    plt.close()
    return time_y.to_numpy(dtype=np.float64)

def days_and_nights(data,number,time_index):
    night_mask=(time_index.hour>=20) | (time_index.hour<6) # creates a mask for the night time data
    day_mask=(time_index.hour<20) | (time_index.hour>=6) # creates a mask for the day time data
    night_data=data[night_mask] # generates heart rate data only between 10pm and 6am
    day_data= data[day_mask] # generates the other data for comparison
    night_y=night_data['value']  # extracts the heart rate values from the data
    day_y=day_data['value']  # extracts the heart rate values from the data
    night_x=night_data['start']
    day_x=day_data['start']
    plt.title('Heart rates over study - nights only')
    plt.plot(night_x,night_y,label='HR data')
    plt.xlabel('Date')
    plt.ylabel('Heart rate [bpm]')
    plt.axhline(np.average(night_y),color='red')
    plt.tick_params(axis='x',labelrotation=90,length=0.1)
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.savefig('/data/t/smartWatch/patients/completeData/DamianInternshipFiles/heartRateRecord{}/FullNight'.format(number))
    plt.close()
    df=resting_max_and_min(night_mask,day_mask,time_index,day_y,night_data)


    return np.average(night_y),df

def resting_max_and_min(night_mask,day_mask,time_index,day_data,night_data):
    results=[]
    resting_hr=pd.DataFrame({'date':[],
                             'resting_hr':[]})
    days_df=pd.DataFrame({'date':[],
                          'avg_day':[],
                          'min_day':[],
                          'max_day':[]})
    nights_df=pd.DataFrame({'date':[],
                          'avg_night':[],
                          'min_night':[],
                          'max_night':[]})
    days=np.unique(time_index.normalize()) # normalises the time index to remove the time component

    for i,day in enumerate(days):
        date_str = day.strftime('%Y-%m-%d')
        day_mask_i= time_index.normalize()[day_mask] == day # creates a mask for the current day
        day_vals=day_data[day_mask_i] # gets the heart rate data for the current day
        avg_day = np.mean(day_vals) if len(day_vals) > 0 else np.nan
        min_day = np.min(day_vals) if len(day_vals) > 0 else np.nan
        max_day = np.max(day_vals) if len(day_vals) > 0 else np.nan
        
        night_mask_i= time_index.normalize()[night_mask] == day # creates a mask for the current night

  
        night_vals=night_data[night_mask_i]['value'].to_numpy(dtype=np.float64) # gets the heart rate data for nights


        if len(night_vals)>0:
            avg_night = np.mean(night_vals)
            min_night = np.min(night_vals) 
            max_night = np.max(night_vals)
            min_indx=np.argmin(night_vals)
            resting_hr_val=np.inf
            for j in range(len(night_vals)):
                current_val=np.mean(night_vals[j:j+5])
                if current_val<resting_hr_val:
                    resting_hr_val=current_val
        
            
        else:
            resting_hr_val = np.nan
            avg_night = np.nan
            min_night = np.nan
            max_night = np.nan
        results.append({
        'date': date_str,
        'avg_day': avg_day,
        'min_day': min_day,
        'max_day': max_day,
        'avg_night': avg_night,
        'min_night': min_night,
        'max_night': max_night,
        'resting_hr': resting_hr_val
    })

    df = pd.DataFrame(results)

    return df


def plotting(data,number,p,months_on=True,weeks_on=True,active_on=True,total_on=True,day_and_night_on=True):
    Path("/data/t/smartWatch/patients/completeData/DamianInternshipFiles/heartRateRecord{}".format(number)).mkdir(exist_ok=True) # creating new directory
    ### time stamps have the form yr;mnth;dy;hr;min;sec;tz ### 
    data['value']=np.array([
            np.mean([int(x) for x in item.split(',')])  # Split and convert to integers, then average
            for item in np.char.strip(data['value'].to_numpy('str'),'[]')])
    time_index=pd.DatetimeIndex(data['start']) # ensures the timestamps are  in datetime format
    months=time_index.to_series().dt.strftime('%b %Y').unique() # finds the unique months in the data
    avg_hr_months,weeks,avg_week_hr, avg_hr_active_day,activities,time_y,avg_night,df=None,None,None,None,None,None,None,None
    if months_on:
        avg_hr_months=months_calc(data,number,time_index)
    if weeks_on:
        weeks,avg_week_hr=week_calc(data,number,time_index)
    if active_on:
        avg_hr_active_day,activities=active_days_calc(data,number,time_index,p)
    if total_on:
        time_y=total_timespan(data,number)
    if day_and_night_on :
        avg_night,df=days_and_nights(data,number,time_index)    
    print(len(weeks),len(avg_week_hr))
    return {"HRV":1/time_y,
            "avg_hr_months":avg_hr_months,
            "avg_hr_night":avg_night,
            "average_hr":np.average(time_y),
            "months":months,
            "avg_week_hr":avg_week_hr,
            "avg_active_hr":avg_hr_active_day,
            "weeks":weeks,
            "active_days":activities,
            "resting_plus_more":df}

def detecting_crossover(log_F,log_n):
    best_split=None
    best_score=-np.inf
    results=[]
    min_points=3
    for i in range(min_points, len(log_n) - min_points):
        x1, y1 = log_n[:i], log_F[:i]
        x2, y2 = log_n[i:], log_F[i:]
        slope1, _, r1, _, _ = linregress(x1, y1)
        slope2, _, r2, _, _ = linregress(x2, y2)
        total_r_squared=r1**2+r2**2
        balance_penalty = min(len(x1), len(x2)) / len(log_n)
        total_r_squared *= balance_penalty
        if total_r_squared>best_score:
            best_score=total_r_squared
            best_split=(i,slope1,slope2)
    return best_split

def detecting_outliers(ECG_RR):
    outliers=[]
    for i in range(6,len(ECG_RR)): # finds the outliers in the HRV graph where a value is greater than 120% of the average of the previous 5.
        last_5_mean=np.mean(ECG_RR[i-6:i-1])
        if abs(ECG_RR[i])>1.2*last_5_mean:
            outliers.append(i)
    outliers=np.array(outliers,dtype=int)
    mask = np.ones(len(ECG_RR), dtype=bool)
    mask[outliers] = False
    return mask

def DFA(RR):
    
    window_sizes=np.unique(np.logspace(0.5, np.log10(len(RR)), 100).astype(int)) # produces an array of varying window sizes scaled for logspace

    y=np.cumsum(RR - np.mean(RR)) # produces an inegration curve of each point - average
    F_s=[]
    for size in window_sizes: # loops through all the window sizes
        windows_num=len(y)//size # number of windows
        windows=y[:windows_num*size].reshape(windows_num,size) # readings in each window
        F_s_w=[]
        for w in windows:
            x=np.arange(size)
            vals=np.polyfit(x,w,deg=1,cov=False) # least square regression fit for each window
            local_trend=x*vals[0]+vals[1]
            F_s_w.append(np.sum((w-local_trend)**2)/size) # mean-squared of the RMS in each window = local fluctuations^2
        F_s_w=np.array(F_s_w)
        F_s.append(np.sqrt(np.sum(F_s_w)/windows_num)) # RMS of total fluctuations from each window/local fluctuation
    F_s=np.array(F_s)
    return F_s,window_sizes

def creating_or_updating_tales(con,table_name,columns,patient_num,value_matrix,column_matrix):
    cur=con.cursor()
    cur.execute(f"CREATE TABLE {table_name}(Number TEXT, PRIMARY KEY(Number))") # creates a new table with the specified columns
    adding_columns_command=";".join([f"ALTER TABLE {table_name} ADD COLUMN '{col}' TEXT" for col in columns]) # creates a command to add the columns to the table
    cur.executescript(adding_columns_command) # executes the command to add the columns

    for i,row in enumerate(column_matrix):
        for j,key in enumerate(row):
            try:
                cur.execute(f"INSERT INTO {table_name}('Number','{key}') VALUES(?,?)",(patient_num[i],value_matrix[i][j])) # inserts the first values into the table
            except:
                cur.execute(f"UPDATE {table_name} SET '{key}' = ? WHERE Number = ?",(value_matrix[i][j],patient_num[i])) # updates the subsequent values in the table

def DFA_analysis(RR,patientNum,type,plot=True):
    """ Performs Detrended Fluctuation Analysis (DFA) on the RR intervals and returns scaling exponents."""
    if len(RR)<1000:
        return (np.nan,np.nan,np.nan),[],[]
    F_s,window_sizes=DFA(RR) # performs DFA on the data
    log_n=np.log10(window_sizes)
    log_F=np.log10(F_s)
    params=detecting_crossover(log_F,log_n)
    cross_point=log_n[params[0]] # gets the crossover point from the params
    m1=params[1]
    m2=params[2]
    H_hat1 = np.polyfit(log_n[np.where(log_n<cross_point)],log_F[np.where(log_n<cross_point)],1,cov=False) # fits linear curve to the first section of the DFA plot
    regression_line_1=log_n[np.where(log_n<cross_point)]*H_hat1[0]+H_hat1[1]
    
    H_hat2 = np.polyfit(log_n[np.where(log_n>=cross_point)],log_F[np.where(log_n>=cross_point)],1,cov=False) # fits linear curve to the second section of the DFA plot
    regression_line_2=log_n[np.where(log_n>=cross_point)]*H_hat2[0]+H_hat2[1]
    if plot==True:
        ax=DFA_plot(params,log_n,log_F,H_hat1,H_hat2,patientNum,type) # plots the DFA results
    else:
        ax=None
    m,logn=plotting_scaling_pattern(log_n,log_F,patientNum,ax,type)

    H_hat=(m1,m2 ,cross_point) # returns the scaling exponents and crossover point for PPG data
    return H_hat,m,logn

def databasing(metrics,patient=True,months_on=True,weeks_on=True,active_on=True,total_on=True,day_and_night_on=True):
    db_name = 'patient_metrics.db' if patient else 'volunteer_metrics.db'
    con = sqlite3.connect(db_name)
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS Months")
    cur.execute("DROP TABLE IF EXISTS Weeks")
    cur.execute("DROP TABLE IF EXISTS Active")
    cur.execute("DROP TABLE IF EXISTS Patients")
    cur.execute("DROP TABLE IF EXISTS DayAndNight")

    patient_num=metrics['Patient_num'] # gets the patient number from the metrics dictionary

    if months_on:
        months=sorted(np.unique(np.concatenate(metrics['months'])),key=lambda x: datetime.strptime(x, '%b %Y')) #unique months in the metrics dictionary
        print(months)
        creating_or_updating_tales(con, 'Months', months, patient_num, metrics['avg_hr_per_month'],metrics['months']) # creates or updates the Months table with the average heart rate per month

    if weeks_on:
        weeks=sorted(np.unique(np.concatenate(metrics['weeks'])),key=lambda x: datetime.strptime(x, '%Y-W%W')) #unique weeks in the metrics dictionary
        print(weeks)
        creating_or_updating_tales(con, 'Weeks', weeks, patient_num, metrics['avg_hr_per_week'],metrics['weeks']) # creates or updates the Weeks table with the average heart rate per week
    if active_on:
        activities=sorted(np.unique(np.concatenate(metrics['activities'])),key=lambda x: datetime.strptime(x, '%Y-%m-%d')) #unique activities in the metrics dictionary
        print(activities)
        creating_or_updating_tales(con, 'Active', activities, patient_num, metrics['avg_hr_active'],metrics['activities']) # creates or updates the Active table with the average heart rate for each activity
    
    if total_on:
        
    
        cur.execute("CREATE TABLE Patients(Id INTEGER,Number TEXT,night_hr_avg FLOAT,overall_hr_avg FLOAT,scaling_exponent_noise FLOAT,scaling_exponent_linear FLOAT, ECG_scaling_exponent_noise FLOAT, ECG_scaling_exponent_linear FLOAT,crossover_PPG FLOAT, crossover_ECG FLOAT, PRIMARY KEY(Id AUTOINCREMENT), FOREIGN KEY(Number) REFERENCES Months(Number))")
        for i in range(len(patient_num)):
            print(metrics['ECG_scaling_exponent_noise'][i])
            cur.execute("INSERT INTO Patients(Number,night_hr_avg,overall_hr_avg,scaling_exponent_noise,scaling_exponent_linear,ECG_scaling_exponent_noise,ECG_scaling_exponent_linear,crossover_PPG,crossover_ECG) VALUES (?,?,?,?,?,?,?,?,?)",(metrics['Patient_num'][i],metrics['avg_hr_night'][i],metrics['avg_hr_overall'][i],metrics['scaling_exponent_noise'][i],metrics['scaling_exponent_linear'][i],metrics['ECG_scaling_exponent_noise'][i],metrics['ECG_scaling_exponent_linear'][i],metrics['crossover_PPG'][i],metrics['crossover_ECG'][i]))
        
    if day_and_night_on:
        # creates a table to store the rest of the patient data
        cur.execute("CREATE TABLE DayAndNight(Id INTEGER, Number TEXT, date TEXT, day_avg FLOAT, night_avg FLOAT, day_min FLOAT, night_min FLOAT, day_max FLOAT, night_max FLOAT,resting_hr FLOAT, PRIMARY KEY(Id AUTOINCREMENT), FOREIGN KEY(Number) REFERENCES Patients(Number))")
        for i in range(len(patient_num)):
            for j in range(len(metrics['days'][i])):
                cur.execute("""INSERT INTO DayAndNight(Number,date,day_avg,night_avg,day_min,night_min,day_max,night_max,resting_hr) VALUES (?,?,?,?,?,?,?,?,?)""",(metrics['Patient_num'][i],metrics['days'][i][j],metrics['day_avg'][i][j],metrics['night_avg'][i][j],metrics['day_min'][i][j],metrics['night_min'][i][j],metrics['day_max'][i][j],metrics['night_max'][i][j],metrics['resting_hr'][i][j]))

        




    con.commit() # commits and closes the database
    con.close()


def surrogate_databasing(surrogate_dictionary,type):
    con=sqlite3.connect('patient_metrics.db') # connects to database
    cur=con.cursor()
    #cur.execute("DROP TABLE Surrogates")
    #cur.execute("CREATE TABLE Surrogates(Number TEXT,'{ftype}_linear' FLOAT,'{ftype}_noise' FLOAT,PRIMARY KEY(Number))".format(ftype=type))
    # sql_commands = "ALTER TABLE Surrogates ADD COLUMN '{ftype}_linear' FLOAT".format(ftype=type)
    # cur.executescript(sql_commands)
    # sql_commands = "ALTER TABLE Surrogates ADD COLUMN '{ftype}_noise' FLOAT".format(ftype=type)
    # cur.executescript(sql_commands)
    for i in range(len(surrogate_dictionary['Patient_num'])):
        flinear=surrogate_dictionary['Surrogate_data_linear'][i]
        fnoise=surrogate_dictionary['Surrogate_data_noise'][i]
        fnum=surrogate_dictionary['Patient_num'][i]
        sql="UPDATE Surrogates SET {ftype}_linear=?,{ftype}_noise=? WHERE Number=?".format(ftype=type)
        cur.execute(sql,(flinear,fnoise,fnum))
    con.commit()
    con.close()

def surrogate(data):
    #data=np.random.permutation(data)
    n_iterations=1000
    tol=1e-8
    x = np.asarray(data)
    sorted_x = np.sort(x)
    phases = np.angle(np.fft.fft(np.random.randn(len(x))))
    surr = np.copy(x)

    for _ in range(n_iterations):
        # 1. Match power spectrum
        surr_fft = np.fft.fft(surr)
        surr_phases = np.exp(1j * phases)
        new_spectrum = np.abs(surr_fft) * surr_phases
        surr = np.real(np.fft.ifft(new_spectrum))

        # 2. Match amplitude distribution
        surr = sorted_x[np.argsort(np.argsort(surr))]

        # 3. Update phases
        new_phases = np.angle(np.fft.fft(surr))
        if np.allclose(phases, new_phases, atol=tol):
            break
        phases = new_phases

    return surr

def DFA_plot(params,log_n,log_F,H_hat1,H_hat2,patientNum,type): 
    """
    Plots the Detrended Fluctuation Analysis (DFA) results and computes scaling exponents.

    This function creates a log-log plot of DFA fluctuation values against segment sizes (`lag`),
    fits piecewise linear regressions to detect scaling regions, identifies the crossover point
    (if any), and plots the scaling pattern. It also saves the resulting plot to a PNG file.

    Parameters:
        lag (array-like): Window sizes used in the DFA calculation.
        dfa (array-like): DFA fluctuation function corresponding to the `lag` values.
        patientNum (str or int): Identifier for the patient whose data is being analyzed.
        RR (array-like): RR interval data used for determining whether DFA should proceed.
        type (str): Description of the dataset type ('' for PPG and 'ECG').

    Returns:
        tuple:
            (a1, a2, crossover_log_n) (tuple): DFA scaling exponents before and after the crossover point,
                                               and the log of the crossover lag value.
            m (float): Slope of the overall scaling pattern.
            logn (array-like): Log-transformed lag values used in scaling pattern analysis.

    Notes:
        - If the RR interval data has fewer than 1000 points,
          the function returns NaNs to indicate insufficient data.
        - The function saves the scaling pattern plot to a fixed directory.
    """
     
    cross_indx,a1,a2=params
    cross_point=log_n[cross_indx]
    ax,fig=plt.subplots(2,1,figsize=(12, 8))
    plt.subplot(2,1,1)
    # plots log-log of DFA analysed data against n, cutting off the end where linearality is lost
    # plt.plot(np.log10(lag)[np.where(np.log10(lag)<3)], log_F[np.where(np.log10(lag)<3)], 'o',label='DFA data')
    plt.plot(log_n, log_F, 'o',label='DFA data',color='g')
    plt.xlabel('logged size of integral slices')
    plt.ylabel('log of fluctuations from fit for each slice size')
    print(type)
    plt.title('DFA - Measure of Randomness in HRV {} Patient {}'.format(type,patientNum))
    plt.axvline(log_n[cross_indx], color='r', linestyle='--', label="Crossover point")
    # closest=(np.abs(lag - len(RR)/10)).argmin()
    # A_L=10**(np.log10(dfa[closest])-2*np.log10(lag[closest]))
    # Linear_trend=(A_L)*lag**2
    # plt.plot(log_n,np.log10(Linear_trend))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # fits a linear line to the log-log graph
    # H_hat = np.polyfit(log_n[np.where(log_n<3)],np.log10(dfa)[np.where(log_n<3)],1,cov=False)
    # plt.plot(log_n[np.where(log_n<3)],log_n[np.where(log_n<3)]*H_hat[0]+H_hat[1],color='red')
    
    regression_line_1=log_n[np.where(log_n<cross_point)]*H_hat1[0]+H_hat1[1]
    plt.plot(log_n[np.where(log_n<cross_point)],regression_line_1,color='red')
    regression_line_2=log_n[np.where(log_n>=cross_point)]*H_hat2[0]+H_hat2[1]
    plt.plot(log_n[np.where(log_n>=cross_point)],regression_line_2,color='blue')
    plt.subplot(2,1,2)
    plt.axvline(log_n[cross_indx], color='r', linestyle='--')
    
    return (a1,a2,log_n[cross_indx])

def adding_to_dictionary(metrics,patientNum,RR,H_hat,H_hat_ECG):

    metrics['Patient_num'].append(patientNum)
    metrics['avg_hr_per_month'].append(RR['avg_hr_months'])
    metrics['avg_hr_night'].append(RR['avg_hr_night'])
    metrics['avg_hr_overall'].append(RR['average_hr'])
    metrics['avg_hr_active'].append(RR['avg_active_hr'])
    metrics['scaling_exponent_noise'].append(H_hat[0])
    metrics['scaling_exponent_linear'].append(H_hat[1])
    metrics['ECG_scaling_exponent_noise'].append(H_hat_ECG[0])
    metrics['ECG_scaling_exponent_linear'].append(H_hat_ECG[1])
    metrics['crossover_PPG'].append(H_hat[2])
    metrics['crossover_ECG'].append(H_hat_ECG[2])
    metrics['avg_hr_per_week'].append(RR['avg_week_hr'])
    metrics['months'].append(RR['months'])
    metrics['weeks'].append(RR['weeks'])
    metrics['activities'].append(RR['active_days'])
    df=RR['resting_plus_more']
    try:
        metrics['days'].append(df['date'].to_numpy())
        metrics['day_avg'].append(df['avg_day'].to_numpy())
        metrics['night_avg'].append(df['avg_night'].to_numpy())
        metrics['day_min'].append(df['min_day'].to_numpy())
        metrics['night_min'].append(df['min_night'].to_numpy())
        metrics['day_max'].append(df['max_day'].to_numpy())
        metrics['night_max'].append(df['max_night'].to_numpy())
        metrics['resting_hr'].append(df['resting_hr'].to_numpy())
    except:
        pass
    return metrics

def alpha_beta_filter(x,y,Q=500):
    d=x[1]-x[0]
    N=len(x)
    m_est=np.zeros(N)
    G_est=np.zeros(N)
    G_est[0]=y[0]
    m_est[0]=0
    for k in range(1,N):
        a_k = 2*(2*k - 1) / (k*(k+1))
        b_k = 6 / (k*(k+1))
        if k > Q:
            a_k = 2*(2*Q - 1) / (Q*(Q+1))
            b_k = 6 / (Q*(Q+1))
        G_pred=G_est[k-1]+m_est[k-1]*d # next point as previous + gradient at last point * distance between points
        G_est[k]=(1-a_k)*G_pred + a_k*y[k]
        m_est[k]=m_est[k-1]+b_k/d*(y[k]-G_pred)
    return m_est,G_est

def interpolating_for_uniform(logn,log_f):
    uniform_log_n=np.linspace(logn.min(),logn.max(),100)
    spline=UnivariateSpline(logn,log_f,s=0)
    smoothed_log_f=spline(uniform_log_n)
    return uniform_log_n,smoothed_log_f

def plotting_scaling_pattern(log_n,log_f,patient_num,ax,type):
    
    interpolated=interpolating_for_uniform(log_n,log_f)
    mask=np.where(interpolated[0]>0.55)
    m,log_f=alpha_beta_filter(*interpolated)
    print(len(m))
    if ax is None:
        return m,interpolated[0]
    plt.subplot(2,1,2)
    if np.max(m)>2:
        plt.ylim(0,np.max(m)+0.5)
    plt.ylim(0,2)
    plt.plot(interpolated[0][mask],m[mask])
    plt.axhline(1,linestyle='dashed',color='k')
    plt.axhline(0.5,linestyle='dashed',color='k')
    plt.axhline(1.5,linestyle='dashed',color='k')
    plt.xlabel('logged size of integral slices')
    plt.ylabel('gradient at each value of n - $m_{e}(n)$')
    plt.title('continuous gradient over the DFA plot')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'/data/t/smartWatch/patients/completeData/DamianInternshipFiles/Graphs/scaling_patterns/{patient_num}-{type}.png')
    plt.close()
    return m,interpolated[0]
    #plt.savefig(f'/data/t/smartWatch/patients/completeData/DamianInternshipFiles/Graphs/scaling_patterns/{patient_num}.png')

def ECG_HRV(ECG_RR,patientNum):
    ECG_RR=(ECG_RR[:,:len(ECG_RR[0])-1].T).flatten()
    ECG_RR = ECG_RR[~np.isnan(ECG_RR)] # removes nans
    plt.plot(np.arange(0,len(ECG_RR+1)),ECG_RR)
    plt.title('ECG HRV')
    plt.ylabel('RR interval (s)')
    plt.xlabel('Beats')
    plt.savefig("/data/t/smartWatch/patients/completeData/DamianInternshipFiles/heartRateRecord{}/ECG_HRV".format(patientNum))
    plt.close()
    mask=detecting_outliers(ECG_RR)
    ECG_RR=ECG_RR[mask] # removes outliers
    
    plt.plot(np.arange(0,len(ECG_RR+1)),ECG_RR)
    plt.title('ECG HRV Filtered')
    plt.ylabel('RR interval (s)')
    plt.xlabel('Beats')
    plt.savefig("/data/t/smartWatch/patients/completeData/DamianInternshipFiles/heartRateRecord{}/ECG_HRV_Filtered".format(patientNum))
    plt.close()
    return ECG_RR # returns the RR intervals for the ECG data

def avg_scaling_pattern(scaling_patterns,type):
    """
    Computes the average scaling pattern from a DataFrame of scaling patterns.

    Parameters:
        scaling_patterns (DataFrame): DataFrame containing 'gradient' and 'log_n' columns.

    Returns:
        tuple: Average gradient and log_n values.
    """
    avg_gradient = np.mean([val for sublist in scaling_patterns['gradient'] for val in sublist])
    avg_log_n = np.mean([val for sublist in scaling_patterns['log_n'] for val in sublist])
    return avg_gradient, avg_log_n

def plotting_average_scaling_pattern(scaling_patterns,type):
    """
    Plots the average scaling pattern from a DataFrame of scaling patterns.

    Parameters:
        scaling_patterns (DataFrame): DataFrame containing 'gradient' and 'log_n' columns.
        patientNum (str): Patient number for labeling the plot.
        type (str): Type of data (e.g., 'PPG', 'ECG') for labeling the plot.

    Returns:
        None
    """
    avg_gradient, avg_log_n = avg_scaling_pattern(scaling_patterns,type)
    plt.plot(avg_log_n, avg_gradient)
    plt.title(f'Average Scaling Pattern for {type}')
    plt.xlabel('Log n')
    plt.ylabel('Average Gradient')
    plt.savefig(f'/data/t/smartWatch/patients/completeData/DamianInternshipFiles/Graphs/scaling_patterns/{type}-average.png')
    plt.close()
    return avg_gradient, avg_log_n

# %%
def main():
    from scipy.fft import fft, ifft, fftshift, ifftshift
    from scipy.interpolate import interp1d
    months_on,weeks_on,active_on,total_on,day_and_night_on=True,True,True,True,True
    # dictionary storing all patient data calcualted in the code to be outputted to db
    metrics={'Patient_num':[],
                'avg_hr_per_month':[],
                'avg_hr_night':[],
                'avg_hr_overall':[],
                'avg_hr_active':[],
                'scaling_exponent_noise':[],
                'scaling_exponent_linear':[],
                'ECG_scaling_exponent_noise':[],
                'ECG_scaling_exponent_linear':[],
                'crossover_PPG':[],
                'crossover_ECG':[],
                'avg_hr_per_week':[],
                'months':[],
                'weeks':[],
                'activities':[],
                'days':[],
                'day_avg':[],
                'night_avg':[],
                'day_min':[],
                'night_min':[],
                'day_max':[],
                'night_max':[],
                'resting_hr':[]}
    surrogate_dictionary={'Patient_num':[],
                          'Surrogate_data_linear':[],
                          'Surrogate_data_noise':[]}
    counter=0
    data=pd.read_excel('/data/t/smartWatch/patients/completeData/dataCollection_wPatch Starts.xlsx','Sheet1')
    scaling_patterns_PPG=pd.DataFrame({'gradient':[],'log_n':[]})
    scaling_patterns_ECG=pd.DataFrame({'gradient':[],'log_n':[]})
    for i in range(2,10):
        print(i)
        if i==42 or i==24:
            continue
        if i<10:
            patientNum='0{}'.format(str(i))
        else:
            patientNum='{}'.format(str(i))
        
        #patientNum='data_AMC_1633769065'
        patient_analysis=True

          
        try:
            ECG_RR,ECG_R_times=patient_output(patientNum,patient=patient_analysis)
            heartRateDataByMonth=sortingHeartRate(patientNum,patient=patient_analysis)
            RR=plotting(heartRateDataByMonth,patientNum,p=patient_analysis,months_on=months_on,weeks_on=weeks_on,active_on=active_on,total_on=total_on,day_and_night_on=day_and_night_on)
        
            print(patientNum)
        except:
            continue
        ECG_RR=ECG_HRV(ECG_RR,patientNum)
        #surrogate_data=surrogate(RR[0])

        H_hat,m,log_n=DFA_analysis(RR['HRV'],patientNum,'PPG')
        scaling_patterns_PPG.loc[i]=[m,log_n]
        

        H_hat_ECG,m,log_n=DFA_analysis(ECG_RR,patientNum,'ECG')
        scaling_patterns_ECG.loc[i]=[m,log_n]
        

        print('H_hat',H_hat)
        print('H_hat_ECG',H_hat_ECG)
        metrics=adding_to_dictionary(metrics,patientNum,RR,H_hat,H_hat_ECG)
    print(scaling_patterns_ECG)
    plotting_average_scaling_pattern(scaling_patterns_PPG,'PPG')
    plotting_average_scaling_pattern(scaling_patterns_ECG,'ECG') 
    #print(surrogate_dictionary)
    #surrogate_databasing(surrogate_dictionary,'IAAFT')
    databasing(metrics,patient=patient_analysis,months_on=months_on,weeks_on=weeks_on,active_on=active_on,total_on=total_on,day_and_night_on=day_and_night_on)
    



def ECG_data(Patient_num):
    print('hello')


if __name__=="__main__":
    main()
    

"""
inputs are /data/t/smartWatch
outputs write to same place
"""
# %%
