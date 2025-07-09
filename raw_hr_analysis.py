

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
        plt.show()
        plt.legend()
        #Path("/data/t/smartWatch/patients/completeData/DamianInternshipFiles/heartRateRecord{}".format(number)).mkdir(exist_ok=True) # creating new directory
        plt.savefig('/data/t/smartWatch/patients/completeData/DamianInternshipFiles/heartRateRecord{}/month-{}'.format(number,m))
        plt.close()
        avg_hr_per_month.append(np.average(month_y)) # averages the hr for that month
    

    return avg_hr_per_month

def week_calc(data,number,time_index):
    # finds the unique weeks in the data
    avg_hr_weekly=[]
    weeks=np.unique(time_index.isocalendar().week) # finds the unique weeks in the data
    for w in weeks:
        mask=(time_index.isocalendar().week==w).to_numpy() # creates a mask for the current week
        
        
        week_data=data[mask]
        week_x=week_data['start']
        week_y = week_data['value']
        avg_hr_weekly.append(np.average(week_y)) # averages the hr for that weeks
        plt.title('Heart rate for week {}'.format(w))
        plt.plot(week_x,week_y,label='HR data') # plots the heart rate data for this week
        plt.xlabel('Date')
        plt.ylabel('Heart rate [bpm]')
        plt.tick_params(axis='x',labelrotation=45,length=0.1)
        plt.tight_layout()
        plt.show()
        plt.legend()
        #Path("/data/t/smartWatch/patients/completeData/DamianInternshipFiles/heartRateRecord{}".format(number)).mkdir(exist_ok=True) # creating new directory
        plt.savefig('/data/t/smartWatch/patients/completeData/DamianInternshipFiles/heartRateRecord{}/week-{}'.format(number,w))
        plt.close()
    
    return weeks,avg_hr_weekly

def active_days_calc(data,number,time_index,start,end,patient):
    avg_hr_active_days=[] # list to store the average heart rate for each day with activity
    normalised_time_index=time_index.normalize() # normalises the time index to remove the time component
    start_time_index=pd.DatetimeIndex(start).normalize() # ensures the activity start times are in datetime format
    active_dates= np.unique(start_time_index) # finds all unique dates activities were done on
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
        plt.savefig('/data/t/smartWatch/patients/completeData/DamianInternshipFiles/heartRateRecord{}/{}'.format(number,day.strftime('%Y-%m-%d')))
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
    plt.savefig('/data/t/smartWatch/patients/completeData/DamianInternshipFiles/heartRateRecord{}/Full'.format(number))
    plt.close()
    return time_y

def days_and_nights(data,number,time_index,start,end):
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
    plt.savefig('/data/t/smartWatch/patients/completeData/DamianInternshipFiles/heartRateRecord{}/FullNight'.format(number))
    plt.close()
    df=resting_max_and_min(night_mask,day_mask,time_index,day_y,night_data)


    return np.average(night_y),df

def resting_max_and_min(night_mask,day_mask,time_index,day_data,night_data):
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
        print(day)
        night_mask_i= time_index.normalize()[night_mask] == day # creates a mask for the current night
        night_hr_data=night_data['value'] # gets the heart rate data for the current night
        night_vals=night_hr_data[night_mask_i] # gets the heart rate data for the current night
        if len(night_vals)>0:
            avg_night = np.mean(night_vals)
            min_night = np.min(night_vals) 
            max_night = np.max(night_vals)
            
        else:
            resting_hr_val = np.nan
            avg_night = np.nan
            min_night = np.nan
            max_night = np.nan
        print(resting_hr)




    for i in range(len(days)):
        mask=time_index.normalize()[day_mask]==days[i]
        day_vals=day_data[mask]
        days_df.loc[i]=[days[i].strftime('%Y-%m-%d'),np.mean(day_vals),day_vals.min(),day_vals.max()]

        mask=time_index.normalize()[night_mask]==days[i]
        night_vals=night_data[mask]
        try:
            nights_df.loc[i]=[days[i].strftime('%Y-%m-%d'),np.mean(night_vals),night_vals.min(),night_vals.max()]
            print([pd.DataFrame(night_vals).rolling(window=300,min_periods=1).mean().min().to_numpy(dtype=np.float64)])
            resting_hr.loc[i]=[days[i].strftime('%Y-%m-%d'),pd.DataFrame(night_vals).rolling(window=300,min_periods=1).mean().min().to_numpy(dtype=np.float64)] # rolling average of the last 5 minutes of data
        except:
            nights_df.loc[i]=[days[i].strftime('%Y-%m-%d'),np.nan,np.nan,np.nan]
    print(resting_hr)

    df=pd.merge(days_df,nights_df, on='date')
    df=pd.merge(df,resting_hr, on='date') # merges the dataframes together
    print(df)
    return df


def plotting(data,number,p,months_on=True,weeks_on=True,active_on=True,total_on=True,day_and_night_on=True):
    Path("/data/t/smartWatch/patients/completeData/DamianInternshipFiles/heartRateRecord{}".format(number)).mkdir(exist_ok=True) # creating new directory
    ### time stamps have the form yr;mnth;dy;hr;min;sec;tz ### 
    data['value']=np.array([
            np.mean([int(x) for x in item.split(',')])  # Split and convert to integers, then average
            for item in np.char.strip(data['value'].to_numpy('str'),'[]')])
    print(data)
    time_index=pd.DatetimeIndex(data['start']) # ensures the timestamps are  in datetime format
    months=np.unique(time_index.month) # finds the unique months in the data
    start,end= sortingActivityData(number,patient=p) # brings in activity data
    if months_on:
        avg_hr_months=months_calc(data,number,time_index)
    if weeks_on:
        weeks,avg_week_hr=week_calc(data,number,time_index)
    if active_on:
        avg_hr_active_day,activities=active_days_calc(data,number,time_index,start,end,p)
    if total_on:
        time_y=total_timespan(data,number)
    if day_and_night_on:
        avg_night,df=days_and_nights(data,number,time_index,start,end)    
        
    return 1/time_y,avg_hr_months,avg_night,np.average(time_y),months,avg_week_hr,avg_hr_active_day,weeks,activities,df

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
    return best_split,log_n[best_split[0]]

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

def DFA(RR,number):
    
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

def databasing(metrics,patient=True):
    if patient:
        con=sqlite3.connect('patient_metrics.db') # connects to database
    else:
        con=sqlite3.connect('volunteer_metrics.db')
    cur=con.cursor()
    
    cur.execute("DROP TABLE Months")
    cur.execute("CREATE TABLE Months(Number TEXT, PRIMARY KEY(Number))") # creates a table to store the Avg heart rate for each month
    # months_hr=months_hr[np.argsort(months)]
    # months=np.sort(months)
    months=np.unique(np.concatenate(metrics['months']))
    weeks=np.unique(np.concatenate(metrics['weeks']))
    activities=np.unique(np.concatenate(metrics['activities']))
    sql_commands = "; ".join([f"ALTER TABLE Months ADD COLUMN '{month}' TEXT" for month in months])
    cur.executescript(sql_commands)
    for i in range(len(metrics['months'])):
        # months_hr=months_hr[np.argsort(months)]
        # months=np.sort(months)
        for j in range(len(metrics['months'][i])): # loops through each patient and then each month they took readings for
            # try:
            #     cur.execute("ALTER TABLE Months ADD COLUMN '{}' TEXT".format(months[j])) # adds a column to the table for each new month
            # except:
            #     pass
            try:
                # for the first entry for each patient
                cur.execute("INSERT INTO Months('Number','{fmonth}') VALUES('{fnumval}','{fmonthval}')".format(fmonth=metrics['months'][i][j],fnumval=metrics['Patient_num'][i],fmonthval=metrics['avg_hr_per_month'][i][j]))
            except:
                # for susequent entries for each patient
                sql = f'UPDATE Months SET "{metrics['months'][i][j]}" = ? WHERE Number = ?'
                cur.execute(sql, (metrics['avg_hr_per_month'][i][j], metrics['Patient_num'][i]))

                
    cur.execute("DROP TABLE Weeks")
    # exactly the same for Weeks as for Months
    cur.execute("CREATE TABLE Weeks(Number TEXT, PRIMARY KEY(Number))")
    sql_commands = "; ".join([f"ALTER TABLE Weeks ADD COLUMN '{week}' TEXT" for week in weeks])
    cur.executescript(sql_commands)
    for i in range(len(metrics['weeks'])):
        for j in range(len(metrics['weeks'][i])):
            try:
                cur.execute("INSERT INTO Weeks('Number','{fweek}') VALUES('{fnumval}','{fweekval}')".format(fweek=metrics['weeks'][i][j],fnumval=metrics['Patient_num'][i],fweekval=metrics['avg_hr_per_week'][i][j]))
            except:
                sql = f'UPDATE Weeks SET "w/c {metrics["weeks"][i][j]}" = ? WHERE Number = ?'
                cur.execute(sql, (metrics['avg_hr_per_week'][i][j], metrics['Patient_num'][i]))
    cur.execute("DROP TABLE Active")
    # Almost exactly the same for Active as for Months and Weeks
    cur.execute("CREATE TABLE Active(Number TEXT, PRIMARY KEY(Number))")
    sql_commands = "; ".join([f"ALTER TABLE Active ADD COLUMN '{activity}' TEXT" for activity in activities])
    cur.executescript(sql_commands)
    for i in range(len(metrics['activities'])):
        for j in range(len(metrics['activities'][i])):
            try:
                cur.execute("INSERT INTO Active('Number','{factivestart}') VALUES('{fnumval}','{factiveval}')".format(factivestart=metrics['activities'][i][j],fnumval=metrics['Patient_num'][i],factiveval=metrics['avg_hr_active'][i][j]))
            except:
                sql = f'UPDATE Active SET "{metrics["activities"][i][j]}" = ? WHERE Number = ?'
                cur.execute(sql, (metrics['avg_hr_active'][i][j], metrics['Patient_num'][i]))

    
    cur.execute("DROP TABLE Patients")
    cur.execute("DROP tABLE DayAndNight")
    # creates a table to store the rest of the patient data
    cur.execute("CREATE TABLE Patients(Id INTEGER,Number TEXT,night_hr_avg FLOAT,overall_hr_avg FLOAT,scaling_exponent_noise FLOAT,scaling_exponent_linear FLOAT, ECG_scaling_exponent_noise FLOAT, ECG_scaling_exponent_linear FLOAT,crossover_PPG FLOAT, crossover_ECG FLOAT, PRIMARY KEY(Id AUTOINCREMENT), FOREIGN KEY(Number) REFERENCES Months(Number))")
    cur.execute("CREATE TABLE DayAndNight(Id INTEGER, Number TEXT, date TEXT, day_avg FLOAT, night_avg FLOAT, day_min FLOAT, night_min FLOAT, day_max FLOAT, night_max FLOAT, PRIMARY KEY(Id AUTOINCREMENT), FOREIGN KEY(Number) REFERENCES Patients(Number))")
    for i in range(len(metrics['Patient_num'])):
        cur.execute("INSERT INTO Patients(Number,night_hr_avg,overall_hr_avg,scaling_exponent_noise,scaling_exponent_linear,ECG_scaling_exponent_noise,ECG_scaling_exponent_linear,crossover_PPG,crossover_ECG) VALUES ('{fnum}','{fnight}','{foverall}','{fscalingN}','{fscalingL}','{fECGscalingN}','{fECGscalingL}','{fcPPG}','{fcECG}')".format(fnum=metrics['Patient_num'][i],fnight=metrics['avg_hr_night'][i],foverall=metrics['avg_hr_overall'][i],fscalingN=metrics['scaling_exponent_noise'][i],fscalingL=metrics['scaling_exponent_linear'][i],fECGscalingN=metrics['ECG_scaling_exponent_noise'][i],fECGscalingL=metrics['ECG_scaling_exponent_linear'][i],fcPPG=metrics['crossover_PPG'][i],fcECG=metrics['crossover_ECG'][i]))
        for j in range(len(metrics['days'][i])):
            cur.execute("""INSERT INTO DayAndNight(Number,date,day_avg,night_avg,day_min,night_min,day_max,night_max) VALUES (?,?,?,?,?,?,?,?)""",(metrics['Patient_num'][i],metrics['days'][i][j],metrics['day_avg'][i][j],metrics['night_avg'][i][j],metrics['day_min'][i][j],metrics['night_min'][i][j],metrics['day_max'][i][j],metrics['night_max'][i][j]))

    




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

def DFA_plot(lag,dfa,patientNum,RR,type):
    if len(RR)<1000 and len(type)!=0:
        return [np.nan,np.nan,np.nan]
    ax,fig=plt.subplots(2,1,figsize=(12, 8))
    plt.subplot(2,1,1)
    # plots log-log of DFA analysed data against n, cutting off the end where linearality is lost
    # plt.plot(np.log10(lag)[np.where(np.log10(lag)<3)], log_F[np.where(np.log10(lag)<3)], 'o',label='DFA data')
    log_n=np.log10(lag)
    log_F=np.log10(dfa)
    plt.plot(log_n, log_F, 'o',label='DFA data',color='g')
    plt.xlabel('logged size of integral slices')
    plt.ylabel('log of fluctuations from fit for each slice size')
    plt.title('DFA - Measure of Randomness in HRV {} Patient {}'.format(type,patientNum))
    (cross_indx,a1,a2),cross_point=detecting_crossover(log_F,log_n)
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
    
    H_hat1 = np.polyfit(log_n[np.where(log_n<cross_point)],log_F[np.where(log_n<cross_point)],1,cov=False) # fits linear curve to the first section of the DFA plot
    regression_line_1=log_n[np.where(log_n<cross_point)]*H_hat1[0]+H_hat1[1]
    plt.plot(log_n[np.where(log_n<cross_point)],regression_line_1,color='red')
    H_hat2 = np.polyfit(log_n[np.where(log_n>=cross_point)],log_F[np.where(log_n>=cross_point)],1,cov=False) # fits linear curve to the second section of the DFA plot
    regression_line_2=log_n[np.where(log_n>=cross_point)]*H_hat2[0]+H_hat2[1]
    plt.plot(log_n[np.where(log_n>=cross_point)],regression_line_2,color='blue')
    ax,m,logn=plotting_scaling_pattern(log_n,log_F,patientNum,ax)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'/data/t/smartWatch/patients/completeData/DamianInternshipFiles/Graphs/scaling_patterns/{patientNum}.png')
    plt.close()
    return (a1,a2,log_n[cross_indx]),m,logn

def adding_to_dictionary(metrics,patientNum,RR,H_hat,H_hat_ECG):

    metrics['Patient_num'].append(patientNum)
    metrics['avg_hr_per_month'].append(RR[1])
    metrics['avg_hr_night'].append(RR[2])
    metrics['avg_hr_overall'].append(RR[3])
    metrics['avg_hr_active'].append(RR[6])
    metrics['scaling_exponent_noise'].append(H_hat[0])
    metrics['scaling_exponent_linear'].append(H_hat[1])
    metrics['ECG_scaling_exponent_noise'].append(H_hat_ECG[0])
    metrics['ECG_scaling_exponent_linear'].append(H_hat_ECG[1])
    metrics['crossover_PPG'].append(H_hat[2])
    metrics['crossover_ECG'].append(H_hat_ECG[2])
    metrics['avg_hr_per_week'].append(RR[5])
    metrics['months'].append(RR[4])
    metrics['weeks'].append(RR[7])
    metrics['activities'].append(RR[8])
    df=RR[9]
    metrics['days'].append(df['date'].to_numpy())
    metrics['day_avg'].append(df['avg_day'].to_numpy())
    metrics['night_avg'].append(df['avg_night'].to_numpy())
    metrics['day_min'].append(df['min_day'].to_numpy())
    metrics['night_min'].append(df['min_night'].to_numpy())
    metrics['day_max'].append(df['max_day'].to_numpy())
    metrics['night_max'].append(df['max_night'].to_numpy())
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

def plotting_scaling_pattern(log_n,log_f,patient_num,ax):
    plt.subplot(2,1,2)
    interpolated=interpolating_for_uniform(log_n,log_f)
    mask=np.where(interpolated[0]>0.55)
    m,log_f=alpha_beta_filter(*interpolated)
    print(len(m))
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
    return ax,m,interpolated[0]
    #plt.savefig(f'/data/t/smartWatch/patients/completeData/DamianInternshipFiles/Graphs/scaling_patterns/{patient_num}.png')

# %%
def main():
    from scipy.fft import fft, ifft, fftshift, ifftshift
    from scipy.interpolate import interp1d
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
                'night_max':[]}
    surrogate_dictionary={'Patient_num':[],
                          'Surrogate_data_linear':[],
                          'Surrogate_data_noise':[]}
    counter=0
    data=pd.read_excel('/data/t/smartWatch/patients/completeData/dataCollection_wPatch Starts.xlsx','Sheet1')
    scaling_patterns=pd.DataFrame({'gradient':[],
                                   'log_n':[]})
    for i in range(2,50):
        print(i)
        if i==42 or i==24:
            continue
        if i<10:
            patientNum='0{}'.format(str(i))
        else:
            patientNum='{}'.format(str(i))
        
        #patientNum='data_AMC_1633769065'
        patient_analysis=True
        heartRateDataByMonth=sortingHeartRate(patientNum,patient=patient_analysis)
        RR=plotting(heartRateDataByMonth,patientNum,p=patient_analysis)

        try:
            ECG_RR,ECG_R_times=patient_output(patientNum,patient=patient_analysis)
            individual_length=len(ECG_RR[:,0])
            heartRateDataByMonth=sortingHeartRate(patientNum,patient=patient_analysis)
            RR=plotting(heartRateDataByMonth,patientNum,p=patient_analysis)
        
            print(patientNum)
        except:
            continue
        ECG_RR=(ECG_RR[:,:len(ECG_RR[0])-1].T).flatten()
        num_of_ECG=len(ECG_RR)/individual_length
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
        #surrogate_data=surrogate(RR[0])

        dfa,lag=DFA(RR[0],patientNum)
        plt.plot(lag,dfa)
        plt.savefig(f'/data/t/smartWatch/patients/completeData/DamianInternshipFiles/heartRateRecord{patientNum}/unlogged_DFA.png')
        plt.close()
        H_hat,m,log_n=DFA_plot(lag,dfa,patientNum,RR[0],'')
        scaling_patterns.loc[i]=[m,log_n]
        dfa,lag=DFA(ECG_RR,patientNum)
        H_hat_ECG=DFA_plot(lag,dfa,patientNum,ECG_RR,'ECG')
        metrics=adding_to_dictionary(metrics,patientNum,RR,H_hat,H_hat_ECG)
    #print(surrogate_dictionary)
    #surrogate_databasing(surrogate_dictionary,'IAAFT')
    avg_gradient=np.mean(np.vstack(scaling_patterns['gradient'].to_numpy()),axis=0)
    avg_log_n=np.mean(np.vstack(scaling_patterns['log_n'].to_numpy()),axis=0)
    plt.plot(avg_log_n,avg_gradient)
    plt.savefig(f'/data/t/smartWatch/patients/completeData/DamianInternshipFiles/Graphs/scaling_patterns/average.png')
    databasing(metrics,patient=patient_analysis)
    



def ECG_data(Patient_num):
    print('hello')


if __name__=="__main__":
    main()
    

"""
inputs are /data/t/smartWatch
outputs write to same place
"""
# %%
