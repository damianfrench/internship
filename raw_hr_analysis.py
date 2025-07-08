

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

    heartRateData = heartRate[['start','duration','value']].to_numpy(dtype='str')
    #splits the timestamps to seperate out the month, day, hour and minute and then orders them 
    brokenData=split_on_dash(heartRateData[:,0])
    brokenData2=split_on_T(brokenData[:,2])
    brokenData3=split_on_colon(brokenData2[:,1])
    sortedData=heartRateData[np.lexsort((brokenData3[:,2],brokenData3[:,1],brokenData3[:,0],brokenData2[:,0],brokenData[:,1],brokenData[:,0]))] # heart rate data sorted by time
    sortedData[:,0]=split_on_plus(sortedData[:,0])[:,0]
    return sortedData

def sortingActivityData(number,
                        patient=True):
    #loads activity data into array and returns the start and end times
    if patient:
        activities=(pd.read_csv(f'/data/t/smartWatch/patients/completeData/patData/record{number}/activities.csv',header=0))
    else:
        activities=(pd.read_csv(f'/data/t/smartWatch/patients/volunteerData/{number}/activities.csv',header=0))

    activitiesData = activities[['from','to','from (manual)','to (manual)','Timezone','Activity type','Data','GPS','Modified']].to_numpy(dtype='str')
    starts=activitiesData[:,0]
    ends=activitiesData[:,1]
    return starts,ends

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

def plotting(data,number,p):

    ### time stamps have the form yr;mnth;dy;hr;min;sec;tz ### 
    
    all_months=split_on_dash(data[:,0]) # splits time stamps up into year, month, rest
    year_months = only_yearAndmonth(data[:,0])
    months=np.unique(year_months) # months data was taken for
    Weeks=[]
    avg_hr_months=[]
    avg_hr_active_days=[]
    avg_hr_weekly=[]

    for m in months:
        month_data=data[np.where(year_months[:,0]==m)] # generates only the heart rate data for the current month
        
        """individiual months""" 

        month_x=pd.to_datetime(month_data[:,0], format='ISO8601',utc=True) # turns the timestamps for this months data into datetime objects that can be interpretted my matplotlib
        month_y = np.array([
            np.mean([int(x) for x in item.split(',')])  # Split and convert the BPM data to integers, then average if there are multiple readings
            for item in np.char.strip(month_data[:,2],'[]')])
        avg_hr_months.append(np.average(month_y))
        print(m)
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


        """ individual weeks"""

        days=split_on_T(split_on_dash(month_data[:,0])[:,2]) # splits up the month into the days in the month
        int_days=np.array([int(numeric_string) for numeric_string in days[:,0]]) # turns the day numbers into integers
        weeks=np.unique(days[:,0][np.where(np.mod(int_days,7)==0)]) # determins how many week beginnings there are
        int_weeks=np.array([int(numeric_string) for numeric_string in weeks]) # turns the week beginnings into integers
        useable_days=split_on_T(month_data[:,0]) # splits the month data on the time part
        useable_weeks=np.unique(useable_days[:,0][np.where(np.mod(int_days,7)==0)]) # finds the year/month/days that correspond to the start of weeks
        for w in range(len(int_weeks)):
            Weeks.append(useable_weeks[w]) # adds start date of week to array
            if w==0:
                # finds the days in each week
                week_data=month_data[np.where(int_days<=int_weeks[w])]
            else:
                week_data=month_data[np.where((int_days>int_weeks[w-1]) & (int_days<=int_weeks[w]))]
            week_y=np.array([
            np.mean([int(x) for x in item.split(',')]) # Split and convert the BPM data to integers, then average if there are multiple readings
            for item in np.char.strip(week_data[:,2],'[]')])
            avg_hr_weekly.append(np.average(week_y)) # averages the hr for that weeks
        if len(int_weeks)==0:
            week_data=month_data
            week_y=np.array([
            np.mean([int(x) for x in item.split(',')]) # Split and convert the BPM data to integers, then average if there are multiple readings
            for item in np.char.strip(week_data[:,2],'[]')])
            avg_hr_weekly.append(np.average(week_y)) # averages the hr for that weeks
            Weeks.append(np.unique(useable_days[:,0])[0])



        """individual active days"""

        
        activityStarts,activityEnds=sortingActivityData(number,patient=p) # brings in activity data
        splitActivityStarts=split_on_T(activityStarts) # seperates activity start times into date,rest
        splitActivityEnds=split_on_T(activityEnds) # seperates activity end times into date,rest
        Active_months_starts=only_yearAndmonth(splitActivityStarts[:,0])
        activities=[]
        active_dates=np.unique(split_on_T(activityStarts)[:,0]) # finds all unique dates activities were done on
        active_year_months=only_yearAndmonth(active_dates)
        day_data=split_on_T(all_months[np.where(year_months[:,0]==m)][:,2]) # seperates out this months time stamps into days,rest
        activities.append(active_dates[np.where(active_year_months[:,0]==m)])
        monthsActivitiesStarts=activityStarts[np.where(Active_months_starts[:,0]==m)] # generates only the activity start and end times for ones occurring this month
        monthsActivitiesEnds=activityEnds[np.where(Active_months_starts[:,0]==m)]
        monthsActivitiesStartsts=pd.to_datetime(monthsActivitiesStarts, format='ISO8601',utc=True) # converting the activity start and end timestamps to datetime objects
        monthsActivitiesEndsts=pd.to_datetime(monthsActivitiesEnds, format='ISO8601',utc=True)

        if len(monthsActivitiesStartsts)!=0: # check if any activity occured on this day
            day_activity_start=split_on_dash(splitActivityStarts[np.where(Active_months_starts[:,0]==m)][:,0])[:,2]
            day_activity_end=split_on_dash(splitActivityEnds[np.where(Active_months_starts[:,0]==m)][:,0])[:,2] # seperates out the day from the timestamps of this months activities
        else:
            continue
        for day in np.unique(day_activity_start): # loops through the days activity was done on
            plt.title('Heart rate on day with activity: {}-{}'.format(m,day))
            plt.plot(month_x[np.where(day_data[:,0]==day)],month_y[np.where(day_data[:,0]==day)],label='HR data')
            plt.xlabel('Data')
            plt.ylabel('Heart rate [bpm]')
            avg_hr_active_days.append(np.average(month_y[np.where(day_data[:,0]==day)]))
            active_starts=monthsActivitiesStartsts[np.where(day_activity_start==day)] # generates the datetime objects for the activities done on the current day
            active_ends=monthsActivitiesEndsts[np.where(day_activity_end==day)]
            for i in active_starts:
                plt.axvline(pd.to_datetime(i, format='ISO8601',utc=True),color='red')
            for j in active_ends:
                plt.axvline(pd.to_datetime(j, format='ISO8601',utc=True),color='red')
            plt.tick_params(axis='x',labelrotation=90,length=0.1)
            plt.tight_layout()
            plt.legend()
            plt.savefig('/data/t/smartWatch/patients/completeData/DamianInternshipFiles/heartRateRecord{}/month-{}-day-{}'.format(number,m,day))
            plt.close()

        
    
    """plotting total timespan"""

    time_y = np.array([
            np.mean([int(x) for x in item.split(',')]) # Split and convert the BPM data to integers, then average if there are multiple readings
            for item in np.char.strip(data[:,2],'[]')])
    time_x=pd.to_datetime(data[:,0], format='ISO8601',utc=True)
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
    
    
    """only nights"""

    split_only_time=split_on_colon(split_on_T(data[:,0])[:,1]) # splits time stamps up into hour, minute,seconds+tz, other(not important)
    night_data=data[np.where((20<split_only_time[:,0].astype(int)) | (split_only_time[:,0].astype(int)<6))] # generates heart rate data only between 10pm and 6am
    day_data=data[np.where((20>split_only_time[:,0].astype(int)) | (split_only_time[:,0].astype(int)>6))] # generates the other data for comparison
    night_data_ts=np.vstack(np.char.split(night_data[:,0],'+'))[:,0]
    night_x=pd.to_datetime(night_data_ts, format='ISO8601')
    night_y=np.array([
            np.mean([int(x) for x in item.split(',')]) # Split and convert the BPM data to integers, then average if there are multiple readings
            for item in np.char.strip(night_data[:,2],'[]')])
    day_y=np.array([
            np.mean([int(x) for x in item.split(',')])  # Split and convert to integers, then average
            for item in np.char.strip(day_data[:,2],'[]')])
    
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

    """resting and avg. hr day and night"""
    days=split_on_T(day_data[:,0])
    df=pd.DataFrame({'date': days[:,0], 'day_Heart_rate':day_y})
    #print(df.groupby('date')['Heart_rate'].apply(list).to_list())
    days_df=pd.DataFrame({'date':[],
                          'avg_day':[],
                          'min_day':[],
                          'max_day':[]})
    for i in range(len(np.unique(days[:,0]))):
        mask=df['date']==np.unique(days[:,0])[i]
        day_vals=df['day_Heart_rate'][mask]
        days_df.loc[i]=[np.unique(days[:,0])[i],np.mean(day_vals),day_vals.min(),day_vals.max()]
    nights=split_on_T(night_data[:,0])
    df=pd.DataFrame({'date': nights[:,0], 'night_Heart_rate':night_y})
    nights_df=pd.DataFrame({'date':[],
                          'avg_night':[],
                          'min_night':[],
                          'max_night':[]})
    
    for i in range(len(np.unique(nights[:,0]))):
        mask=df['date']==np.unique(nights[:,0])[i]
        night_vals=df['night_Heart_rate'][mask]
        nights_df.loc[i]=[np.unique(nights[:,0])[i],np.mean(night_vals),night_vals.min(),night_vals.max()]

    df=pd.merge(days_df,nights_df, on='date')

    activities=np.concatenate(activities).tolist()
    return 1/time_y,avg_hr_months,np.average(night_y),np.average(time_y),months,avg_hr_weekly,avg_hr_active_days,Weeks,activities,df

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
    sql_commands = "; ".join([f"ALTER TABLE Weeks ADD COLUMN 'w/c {week}' TEXT" for week in weeks])
    cur.executescript(sql_commands)
    for i in range(len(metrics['weeks'])):
        for j in range(len(metrics['weeks'][i])):
            try:
                cur.execute("INSERT INTO Weeks('Number','w/c {fweek}') VALUES('{fnumval}','{fweekval}')".format(fweek=metrics['weeks'][i][j],fnumval=metrics['Patient_num'][i],fweekval=metrics['avg_hr_per_week'][i][j]))
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
