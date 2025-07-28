

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import linregress
from scipy.stats import spearmanr
from MFDFA import MFDFA
from datetime import datetime
import csv
import sqlite3
import sys
from scipy.interpolate import UnivariateSpline
import traceback
from collections import namedtuple
from matplotlib import colormaps
import statsmodels.api as sm
import pyhrv.nonlinear as nl

sys.path.append('/data/t/smartWatch/patients/completeData')
from patient_analysis3 import patient_output



def sortingHeartRate(number,data_path,
                     patient=True):
    """
    Loads and sorts heart rate data by timestamp.

    This function reads raw heart rate data from a CSV file corresponding to either
    a patient or a volunteer, converts the timestamp column to datetime objects,
    and returns the data sorted by time.

    Parameters:
        number (int or str): Identifier used to locate the patient's or volunteer's data file.
        data_path: Path to access the Heart Rate data
        patient (bool, optional): If True, loads data from the patient directory;
                                  if False, loads from the volunteer directory. Default is True.

    Returns:
        pandas.DataFrame: Heart rate data sorted by start time, with timestamps parsed as UTC.

    Notes:
        - Expects the CSV file to contain a 'start' column with ISO8601-formatted timestamps.
    """
    if patient:
        heartRate=(pd.read_csv(f'{data_path}/record{number}/raw_hr_hr.csv',header=0))
    else:
        heartRate=(pd.read_csv(f'{data_path}/{number}/raw_hr_hr.csv',header=0))

    heartRate['start']=pd.to_datetime(heartRate['start'], format='ISO8601',utc=True) # converts the timestamps to datetime objects
    sortedData=heartRate.sort_values(by='start')
    return sortedData

def sortingActivityData(number,data_path,
                        patient=True):
    """
    Loads and parses activity data, returning start and end times.

    This function reads activity time intervals from a CSV file corresponding to either
    a patient or a volunteer, converts the 'from' and 'to' timestamp columns to datetime objects,
    and returns them.

    Parameters:
        number (int or str): Identifier used to locate the patient's or volunteer's activity file.
        data_path: Path to access the Heart Rate data
        patient (bool, optional): If True, loads data from the patient directory;
                                  if False, loads from the volunteer directory. Default is True.

    Returns:
        tuple of pandas.Series:
            - activities['from']: Series of activity start times as UTC datetime objects.
            - activities['to']: Series of activity end times as UTC datetime objects.

    Notes:
        - Assumes activity CSV contains 'from' and 'to' columns in ISO8601 format.
    """
    #loads activity data into array and returns the start and end times
    if patient:
        activities=(pd.read_csv(f'{data_path}/record{number}/activities.csv',header=0))
    else:
        activities=(pd.read_csv(f'{data_path}/{number}/activities.csv',header=0))

    activities['from']=pd.to_datetime(activities['from'], format='ISO8601',utc=True) # converts the timestamps to datetime objects
    activities['to']=pd.to_datetime(activities['to'], format='ISO8601',utc=True)
    return activities['from'],activities['to']

def reading_sleep_Data(number,data_path,patient=True):
    """
    Loads and parses sleep data.

    Parameters:
        number (int or str): Identifier used to locate the patient's or volunteer's activity file.
        data_path: Path to access the Heart Rate data
        patient (bool, optional): If True, loads data from the patient directory;
                                  if False, loads from the volunteer directory. Default is True.
    Returns:
         pandas.DataFrame: Sleep data with timestamps parsed as UTC.
    
    """

    if patient:
        sleep_data=(pd.read_csv(f'{data_path}/record{number}/sleep.csv',header=0))
    else:
        sleep_data=(pd.read_csv(f'{data_path}/{number}/sleep.csv',header=0))
    
    sleep_data['from']=pd.to_datetime(sleep_data['from'],format='ISO8601',utc=True)
    sleep_data['to']=pd.to_datetime(sleep_data['to'],format='ISO8601',utc=True)

    sleep_data["Average heart rate"]=np.array([
            np.mean([int(x) for x in item.split(',')])  # Split and convert to integers, then average
            for item in sleep_data["Average heart rate"].to_numpy('str')])



    return sleep_data

def split_on_plus(data):
    return np.vstack(np.char.split(data,'+'))

def split_on_T(data):
    return np.vstack(np.char.split(data,'T'))

def split_on_dash(data):
    return np.vstack(np.char.split(data,'-'))

def split_on_colon(data):
    data = np.array(data, dtype=str)
    return np.char.partition(data, ':')[:, [0, 2]]
        

def only_yearAndmonth(data):
    return np.vstack(np.array([d[:7] for d in data]))

def months_calc(data,number,saving_path,Flag):
    """
    Calculate and plot average heart rate for each unique month in the dataset.

    This function groups heart rate data by the month extracted from the 'start' datetime column,
    plots the heart rate values over time for each month, saves each plot, and computes
    the average heart rate for each month.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing at least the following columns:
        - 'start': datetime64[ns, tz] type with timestamps of heart rate measurements.
        - 'value': numeric heart rate values corresponding to each timestamp.
    number : int or str
        Identifier (e.g., patient number) used for saving plot files to a directory.
    saving_path : string
        Directory path to save plots to
    Flag : bool
        Boolean Flag determining if plots should be produced

    Returns
    -------
    avg_hr_per_month : list of float
        List containing the average heart rate for each month in the data.

    Side Effects
    ------------
    - Displays a plot of heart rate vs. date for each month.
    - Saves each monthly heart rate plot as a PNG file in a directory path
      based on the given `number` and 'saving_path'
    """
    avg_hr_per_month=[] # list to store the average heart rate for each month
    # finds the unique months in the data and returns them
    months=data['start'].dt.strftime('%b %Y').unique() # finds the unique months in the data
    for m in months:
        mask=data['start'].dt.strftime('%b %Y')==m # creates a mask for the current month
        month_data=data[mask]
        month_x=month_data['start']
        month_y = month_data['value']
        if Flag:
            fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
            ax.set_title('Heart rate for month {}'.format(m))
            ax.plot(month_x,month_y,label='HR data') # plots the heart rate data for this month
            ax.set_xlabel('Date')
            ax.set_ylabel('Heart rate [bpm]')
            ax.tick_params(axis='x',labelrotation=45,length=0.1)
            
            ax.legend()
            plt.show()
            #Path("/data/t/smartWatch/patients/completeData/DamianInternshipFiles/heartRateRecord{}".format(number)).mkdir(exist_ok=True) # creating new directory
            fig.savefig(f'{saving_path}/heartRateRecord{number}/month-{m}')
            plt.close()
        else:
            pass
        avg_hr_per_month.append(np.average(month_y)) # averages the hr for that month
    

    return months,avg_hr_per_month

def week_calc(data,number,saving_path,Flag):
    """
    Calculate and plot average heart rate for each unique ISO week in the dataset.

    This function extracts the ISO week number from the 'start' datetime column, groups heart rate data by week,
    plots the heart rate values over time for each week, saves each plot to disk, and computes
    the average heart rate for each week.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing at least the following columns:
        - 'start': datetime64[ns, tz] type with timestamps of heart rate measurements.
        - 'value': numeric heart rate values corresponding to each timestamp.
    number : int or str
        Identifier (e.g., patient number) used for saving plot files to a directory.
    saving_path : string
        Directory path to save plots to
    Flag : bool
        Boolean Flag determining if plots should be produced

    Returns
    -------
    weeks : numpy.ndarray of str
        Array of unique week labels in ISO year-week format (e.g., '2025-W29').
    avg_hr_weekly : list of float
        List containing the average heart rate for each week in the data.

    Side Effects
    ------------
    - Displays a plot of heart rate vs. date for each week.
    - Saves each weekly heart rate plot as a PNG file in a directory path
      based on the given `number` and 'saving_path'.
    """
    avg_hr_weekly=[]
    weeks=data['start'].dt.strftime('%G-W%V').unique() # finds the unique weeks in the data
    for w in weeks:
        mask=(data['start'].dt.strftime('%G-W%V')==w).to_numpy() # creates a mask for the current week
        week_data=data[mask]
        week_x=week_data['start']
        week_y = week_data['value']
        avg_hr_weekly.append(np.average(week_y)) # averages the hr for that weeks
        if Flag:
            fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
            ax.set_title('Heart rate for week {}'.format(w))
            ax.plot(week_x,week_y,label='HR data') # plots the heart rate data for this week
            ax.set_xlabel('Date')
            ax.set_ylabel('Heart rate [bpm]')
            ax.tick_params(axis='x',labelrotation=45,length=0.1)
            
            ax.legend()
            plt.show()
            #Path("/data/t/smartWatch/patients/completeData/DamianInternshipFiles/heartRateRecord{}".format(number)).mkdir(exist_ok=True) # creating new directory
            fig.savefig(f'{saving_path}/heartRateRecord{number}/week-{w}')
            plt.close()
        else:
            pass
        
    return weeks,avg_hr_weekly

def active_days_calc(data,number,patient,data_path,saving_path,Flag):
    """
    Calculate and plot average heart rate for each day with recorded activity.

    This function identifies the days on which activity occurred by loading activity data,
    normalizes the timestamps in the heart rate data to just dates (removing the time component),
    then calculates and plots heart rate values for each active day. It also marks the start and
    end times of activities on the plots using vertical red lines. Each plot is displayed and saved to disk.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing heart rate data with at least the following columns:
        - 'start': datetime64 timestamps of heart rate measurements.
        - 'value': numeric heart rate values.
    number : int or str
        Identifier for the patient or volunteer, used to load activity data and save plots.
    patient : bool
        If True, load patient activity data; if False, load volunteer activity data.
    data_path : string
        Directory path to access activity data
    saving_path : string
        Directory path to save plots to
    Flag : bool
        Boolean Flag determining if plots should be produced

    Returns
    -------
    avg_hr_active_days : list of float
        List of average heart rates computed for each active day.
    active_dates : numpy.ndarray of str
        Array of unique active dates as strings in 'YYYY-MM-DD' format.

    Side Effects
    ------------
    - Displays a plot for each active day showing heart rate over time with activity start/end times marked.
    - Saves plots as PNG files in a directory path based on the provided `number` and 'saving_path'.
    """

    avg_hr_active_days=[] # list to store the average heart rate for each day with activity
    normalised_time_index=data['start'].dt.normalize() # normalises  to remove the time component
    start,end= sortingActivityData(number,data_path,patient=patient) # brings in activity data
    start_time_index=pd.DatetimeIndex(start).normalize() # ensures the activity start times are in datetime format
    active_dates= start_time_index.to_series().dt.strftime('%Y-%m-%d').unique() # finds all unique dates activities were done on
    for day in active_dates: # loops through the days activity was done on
        mask=normalised_time_index==day   
        day_data=data[mask]
        day_x= day_data['start']
        day_y = day_data['value']
        avg_hr_active_days.append(np.average(day_y))
        if Flag:
            fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
            ax.set_title('Heart rate on  day with activity: {}'.format(day))
            ax.plot(day_x,day_y,label='HR data')
            ax.set_xlabel('Dates')
            ax.set_ylabel('Heart rate [bpm]')
            day_mask=start_time_index==day # creates a mask for the current day
            active_starts=start[day_mask] # generates the datetime objects for the activities done on the current day
            active_ends=end[day_mask]
            for i in active_starts:
                ax.axvline(pd.to_datetime(i, format='ISO8601',utc=True),color='red')
            for j in active_ends:
                ax.axvline(pd.to_datetime(j, format='ISO8601',utc=True),color='red')
            ax.tick_params(axis='x',labelrotation=90,length=0.1)
            ax.legend()
            plt.show()
            fig.savefig(f'{saving_path}/heartRateRecord{number}/{day}')
            plt.close()
        else:
            pass
    return avg_hr_active_days,active_dates

def total_timespan(data,number,saving_path,Flag):
    """
    Plots heart rate over the entire timespan of the dataset and return heart rate values as a NumPy array.

    This function generates a plot showing the heart rate values across the full duration of the dataset.
    A horizontal red line is drawn to represent the average heart rate. The plot is displayed and saved
    to a specified directory. The heart rate values are also returned as a NumPy array.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing heart rate data with at least the following columns:
        - 'start': datetime64 timestamps of heart rate measurements.
        - 'value': numeric heart rate values.
    number : int or str
        Identifier used to determine the directory path for saving the plot.
    saving_path : string
        Directory path to save plots to
    Flag : bool
        Boolean Flag determining if plots should be produced

    Returns
    -------
    numpy.ndarray
        A NumPy array of heart rate values (dtype=float64).

    Side Effects
    ------------
    - Displays a plot of heart rate over the study duration.
    """
    time_y = data['value']  # extracts the heart rate values from the data
    time_x=data['start']
    if Flag:
        fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
        ax.set_title('Heart rate over study')
        ax.plot(time_x,time_y,label='HR data')
        ax.set_xlabel('Date')
        ax.set_ylabel('Heart rate [bpm]')
        ax.axhline(np.average(time_y),color='red')
        ax.tick_params(axis='x',labelrotation=90,length=0.1)
        ax.legend()
        plt.show()
        fig.savefig(f'{saving_path}/heartRateRecord{number}/Full')
        plt.close()
    else:
        pass
    return time_y.to_numpy(dtype=np.float64)

def days_and_nights(data,number,patient,data_path,saving_path,Flag):
    """
    Analyze and plot night-time heart rate data, then calculate daily heart rate statistics.

    This function separates the input heart rate data into day and night segments based on the
    hour of each timestamp. Night is defined as 8 PM to 6 AM. It plots heart rate values during 
    night-time across the entire study duration and saves the figure. It then computes summary 
    statistics and estimated resting heart rates per day using the `resting_max_and_min` function.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing heart rate data. Must have a 'start' datetime column and a 'value' column for heart rate.
    number : int or str
        Identifier used to construct the save path for output plots.
    data_path : string
        Directory path to access activity data
    saving_path : string
        Directory path to save plots to
    Flag : bool
        Boolean Flag determining if plots should be produced

    Returns
    -------
    tuple
        - float : The average heart rate during night-time.
        - pandas.DataFrame : A DataFrame with daily heart rate statistics and estimated resting HR, returned from `resting_max_and_min`.
    """
    try:
        sleep_data=reading_sleep_Data(number,data_path,patient)
    except:
        print(f'patient number {number} doesnt have any sleep data')
        return pd.DataFrame(),pd.DataFrame()
    average_bedtime=circular_mean(sleep_data,'from','to')
    average_wakeup=circular_mean(sleep_data,'from','to',first=False)
    # print(f'average wake up ={average_wakeup}\n average bedtime={average_bedtime}')
    night_mask=(data['start'].dt.hour>=average_bedtime) | (data['start'].dt.hour<average_wakeup) # creates a mask for the night time data
    day_mask=(data['start'].dt.hour<average_bedtime) | (data['start'].dt.hour>=average_wakeup) # creates a mask for the day time data
    night_data=data[night_mask] # generates heart rate data for night
    day_data= data[day_mask] # generates the other data for comparison
    night_y=night_data['value']  # extracts the heart rate values from the data
    day_y=day_data['value']  # extracts the heart rate values from the data
    night_x=night_data['start']
    day_x=day_data['start']
    if Flag:
        fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
        ax.set_title('Heart rates over study - nights only')
        ax.plot(night_x,night_y,label='HR data')
        ax.set_xlabel('Date')
        ax.set_ylabel('Heart rate [bpm]')
        ax.axhline(np.average(night_y),color='red')
        ax.tick_params(axis='x',labelrotation=90,length=0.1)
        ax.legend()
        plt.show()
        fig.savefig(f'{saving_path}/heartRateRecord{number}/FullNight')
        plt.close()
    else:
        pass
    day_df,night_df=resting_max_and_min(night_mask,day_mask,data['start'].dt,day_y,night_y,sleep_data)

    return day_df,night_df

def resting_max_and_min(night_mask,day_mask,time_index,day_data,night_data,sleep_data):
    """
    Calculate daily statistics for day and night heart rate, including an estimate of resting heart rate.

    This function calculates daily averages, minimums, and maximums for heart rate during both day and 
    night periods. It also estimates the resting heart rate each night by finding the lowest 5-point 
    moving average from the night-time heart rate data.

    Parameters
    ----------
    night_mask : pandas.Series or np.ndarray
        Boolean mask for filtering night-time entries from the original dataset.
    day_mask : pandas.Series or np.ndarray
        Boolean mask for filtering day-time entries from the original dataset.
    time_index : pandas.Series
        Datetime values corresponding to the original dataset's timestamps (typically `data['start'].dt`).
    day_data : pandas.Series
        Heart rate values corresponding to the day-time mask.
    night_data : pandas.Series
        Heart rate values corresponding to the night-time mask.
    sleep_data : dataframe containing start, end, and various heart rate metrics during sleep.

    Returns
    -------
    day_df
        A DataFrame containing the following columns for each date:
        - 'day_date'
        - 'day_avg', 'day_min', 'day_max'
        - 'resting_hr' : minimum 5-point rolling average during night-time
        - 'avg_PPG_HRV_day'
        - 'std_PPG_HRV_day'
    night_df
        A DataFrame containing the following columns for each date:
        - 'night_data'
        - 'night_avg', 'night_min', 'night_max'
        - 'avg_PPG_HRV_night'
        - 'std_PPG_HRV_night'
    """
    day_results=[]
    night_results=[]
    nights_HRV_info=[]

    days=np.unique(time_index.normalize()) # normalises the time index to remove the time component

    for i,day in enumerate(days):
        date_str = day.strftime('%Y-%m-%d')
        day_mask_i= time_index.normalize()[day_mask] == day # creates a mask for the current day
        day_vals=day_data[day_mask_i] # gets the heart rate data for the current day
        avg_day = np.mean(day_vals) if len(day_vals) > 0 else np.nan
        min_day = np.min(day_vals) if len(day_vals) > 0 else np.nan
        max_day = np.max(day_vals) if len(day_vals) > 0 else np.nan
        
        night_mask_i= time_index.normalize()[night_mask] == day # creates a mask for the current night

  
        night_vals=night_data[night_mask_i] # gets the heart rate data for nights


        if len(night_vals)>3:
            min_indx=np.argmin(night_vals)
            resting_hr_val=np.inf
            for j in range(len(night_vals)):
                current_val=np.mean(night_vals[j:j+5])
                if current_val<resting_hr_val:
                    resting_hr_val=current_val
 
            HRV_night=1/night_vals

        else:
            resting_hr_val = np.nan
            HRV_night=np.nan
        avg_HRV_night = np.mean(HRV_night) if len(night_vals) > 0 else np.nan
        std_HRV_night = np.std(HRV_night) if len(night_vals) > 0 else np.nan

        nights_HRV_info.append({
            'night_dates':date_str,
            'avg_PPG_HRV_night': avg_HRV_night,
            'std_PPG_HRV_night':std_HRV_night
        })
        
        
        HRV_day=1/day_vals
        avg_HRV_day = np.mean(HRV_day) if len(day_vals) > 0 else np.nan
        std_HRV_day= np.std(HRV_day) if len(day_vals) > 0 else np.nan

        day_results.append({
        'day_dates': date_str,
        'day_avg': avg_day,
        'day_min': min_day,
        'day_max': max_day,
        'resting_hr': resting_hr_val,
        'avg_PPG_HRV_day': avg_HRV_day,
        'std_PPG_HRV_day':std_HRV_day
    })
        
        

    
    nights=sleep_data['from'].dt.normalize().unique()
    for i, night in enumerate(nights):
        date=night.strftime('%Y-%m-%d')
        filtered=sleep_data.loc[sleep_data['from'].dt.normalize()==night]
        # print(f'current night is:{night}, index ={i}')
        # print(filtered)
        night_avg=float(filtered["Average heart rate"].iloc[0])
        night_min=float(filtered["Heart rate (min)"].iloc[0])
        night_max=float(filtered["Heart rate (max)"].iloc[0])
        # print('night avg=',night_avg)
        # print('night min=',night_min)
        # print('night max=',night_max)


        

        night_results.append({'night_dates':date,
                              'night_avg':night_avg,
                              'night_min':night_min,
                            'night_max':night_max})
        

    
    day_df = pd.DataFrame(day_results)
    night_df=pd.DataFrame(night_results)
    nights_HRV_df=pd.DataFrame(nights_HRV_info)
    night_df=pd.merge(night_df,nights_HRV_df,on='night_dates')


    return day_df,night_df


def plotting(data,number,data_path,saving_path,Flags=None):
    """
    Generates and saves various heart rate analysis plots and statistics for a given patient.

    This function performs multiple analyses on heart rate time series data and generates
    corresponding plots. Depending on the flags provided, it can compute statistics for:
    - Monthly average heart rate
    - Weekly average heart rate
    - Days with physical activity
    - Overall heart rate trend over the study
    - Day vs. night heart rate and estimated resting HR

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the heart rate data with at least two columns:
        - 'start': datetime64[ns] timestamps of HR recordings
        - 'value': heart rate readings (either stringified arrays or numerical values)
    number : int or str
        Patient identifier used to create output directory and file names.
    Flags : Dataclass instance containing boolean flags that control which plots are generated:
            - Flags.months: Plot average heart rate per month.
            - Flags.weeks: Plot average heart rate per week.
            - Flags.active: Plot active-day heart rate patterns.
            - Flags.total: Plot overall average HR trends.
            - Flags.day_and_night: Plot day/night averages and min/max HR.


    Returns
    -------
    dict
        Dictionary containing the following keys and their computed values:
        - "HRV": numpy.ndarray, inverse of heart rate values (if total_on is True)
        - "avg_hr_per_months": list of monthly average heart rates (or None)
        - "avg_hr_overall": float, overall average HR across the study
        - "months": list of month labels (e.g., ['Jan 2024', 'Feb 2024', ...])
        - "avg_hr_per_week": list of weekly average heart rates (or None)
        - "avg_hr_active": list of HR values on active days (or None)
        - "weeks": list of ISO weeks corresponding to the data (or None)
        - "activities": list of active dates (or None)
        - "resting_and_days": Dataframe storing the max, min and average heartrates from each day and the resting heartrate
        - "nights": Dataframe storing the max, min and average heartrates from each night
    """
    Path(f"{saving_path}/heartRateRecord{number}").mkdir(exist_ok=True) # creating new directory
    ### time stamps have the form yr;mnth;dy;hr;min;sec;tz ### 
    data['value']=np.array([
            np.mean([int(x) for x in item.split(',')])  # Split and convert to integers, then average
            for item in np.char.strip(data['value'].to_numpy('str'),'[]')])
    avg_hr_months,weeks,avg_week_hr,avg_hr_active_day,activities,time_y,months,day_df,night_df=None,None,None,None,None,np.array([]),None,None,None
    months,avg_hr_months=months_calc(data,number,saving_path,Flags.months)
    weeks,avg_week_hr=week_calc(data,number,saving_path,Flags.weeks)
    avg_hr_active_day,activities=active_days_calc(data,number,Flags.patient_analysis,data_path,saving_path,Flags.activities)
    time_y=total_timespan(data,number,saving_path,Flags.total)
    day_df,night_df=days_and_nights(data,number,Flags.patient_analysis,data_path,saving_path,Flags.day_night)
    poincare_df=pd.DataFrame([poincare_plot(1/time_y,input_type='PPG',patient_number=number,plot_on=Flags.poincare_plot_on)])
    return {"HRV":1/time_y,
            "avg_hr_per_month":avg_hr_months,
            "avg_hr_overall":np.average(time_y),
            "months":months,
            "avg_hr_per_week":avg_week_hr,
            "avg_hr_active":avg_hr_active_day,
            "weeks":weeks,
            "activities":activities,
            "resting_and_days":day_df,
            "nights":night_df,
            'poincare':poincare_df}


def poincare_plot(RR_interval,input_type='',patient_number=None,plot_on=True):
    result=nl.poincare(nni=RR_interval,mode='normal' if plot_on else 'dev')
    print(result)
    if plot_on:
        fig=result['poincare_plot']
        ax=fig.axes[0]
        ax.set_title(f'poincare plot for patient {patient_number} {input_type}')
        ax.set_xlabel('$RRI_{i}$ [ms]')
        ax.set_ylabel('$RRI_{i+1}$ [ms]')
        fig, sd1, sd2, sd_ratio, area = result
    else:
        sd1, sd2, sd_ratio, area = result

    dic = {
        f'{input_type}_sd1': float(sd1),
        f'{input_type}_sd2': float(sd2),
        f'{input_type}_sd_ratio': float(sd_ratio),
        f'{input_type}_ellipse_area':float(area)
    }
    return dic
   
    # plt.show()


def circular_mean(df,key_from,key_to,first=True):
    """
    Computes the weighted circular mean of time-of-day values (in hours), useful for periodic data like sleep times.

    This function calculates the circular average of timestamps (e.g., sleep start or end times), accounting for
    the 24-hour cycle. It uses the amount of time spent (e.g., sleeping) as a weight for each sample.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing datetime columns representing time intervals (e.g., sleep start and end).

    key_from : str
        Column name of the start datetime (e.g., 'sleep_start').

    key_to : str
        Column name of the end datetime (e.g., 'sleep_end').

    first : bool, default=True
        If True, computes the circular mean of the `key_from` column (e.g., average sleep onset time).
        If False, computes the circular mean of the `key_to` column (e.g., average wake-up time).

    Returns:
    --------
    float
        The circular mean in hours (0-24), representing the average time of day.

    Notes:
    ------
    - The result is weighted by duration (`key_to` - `key_from`), assuming longer durations contribute more.
    - Handles wrap-around at midnight using modulo 24 and trigonometric averaging.
    """
    start=df[key_from]
    end=df[key_to]
    start=start.dt.hour + start.dt.minute/60
    end=end.dt.hour + end.dt.minute/60
    time_slept=(end-start)%24
    weights=time_slept/np.sum(time_slept)
    weights=weights.to_numpy()
    if first==True:
        hours = df[key_from].dt.hour + df[key_from].dt.minute / 60
    else:
        hours = df[key_to].dt.hour + df[key_to].dt.minute / 60
    radian_angles=(2*np.pi/24)*hours
    averaged_radian_angle=np.arctan2((np.sin(radian_angles)*weights).mean(),(np.cos(radian_angles)*weights).mean())
    return (averaged_radian_angle/(2*np.pi)*24)%24

def detecting_crossover(log_F,log_n):
    """
    Detects the optimal crossover point in a log-log plot by maximizing the combined R² of two linear fits.

    This function attempts to find the best point to split the data into two segments,
    each of which is fit with a linear regression. It identifies the point where the sum
    of the R² values (goodness of fit) of the two regressions is highest, while also penalizing
    unbalanced splits.

    Parameters
    ----------
    log_F : array-like
        Array of log-transformed dependent variable values (e.g., log power).
    log_n : array-like
        Array of log-transformed independent variable values (e.g., log frequency).

    Returns
    -------
    tuple
        A tuple containing:
        - best_split_index : int
            Index at which to split the data for the best combined fit.
        - slope1 : float
            Slope of the linear regression before the split.
        - slope2 : float
            Slope of the linear regression after the split.
    """
    best_split=None
    best_score=-np.inf
    results=[]
    min_points=3
    for i in range(min_points, len(log_n) - min_points):
        x1, y1 = log_n[:i], log_F[:i]
        x2, y2 = log_n[i:], log_F[i:]
        slope1, _, r1, p, _ = linregress(x1, y1)
        slope2, _, r2, p, _ = linregress(x2, y2)
        total_r_squared=r1**2+r2**2
        balance_penalty = min(len(x1), len(x2)) / len(log_n)
        total_r_squared *= balance_penalty
        if total_r_squared>best_score:
            best_score=total_r_squared
            best_split=(i,slope1,slope2)
    return best_split

def detecting_outliers(ECG_RR):
    """
    Identifies and removes outlier RR intervals based on a simple rolling average threshold.

    For each RR interval (starting from index 6), this function compares it to the mean
    of the previous 5 intervals. If it exceeds 120% of that average, it is marked as an outlier.

    Parameters:
        ECG_RR (array-like): A sequence of RR interval values (e.g., in milliseconds or seconds).

    Returns:
        mask (np.ndarray): A boolean mask array of the same length as ECG_RR.
                           Entries marked `False` indicate detected outliers;
                           `True` indicates normal data points.
    """
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
    """
    Performs Detrended Fluctuation Analysis (DFA) on a time series of RR intervals.

    DFA is a method used to detect long-range correlations in non-stationary time series.
    It integrates the RR series, divides it into windows of various sizes, detrends each
    window using linear regression, and calculates the root-mean-square fluctuation
    within each window.

    Parameters:
        RR (array-like): A sequence of RR intervals (typically in milliseconds or seconds).

    Returns:
        F_s (np.ndarray): The root-mean-square fluctuation for each window size.
        window_sizes (np.ndarray): The array of window sizes used in the DFA.
    """

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

def creating_or_updating_tables(con,table_name,columns,patient_num,value_matrix,column_matrix):
    """
    Creates or updates a SQLite table with patient data.

    This function performs the following:
    1. Creates a new table with the given name and a primary key column 'Number'.
    2. Adds additional columns to the table based on the provided list.
    3. Inserts or updates rows for each patient based on the column/value matrices.

    Parameters:
        con (sqlite3.Connection): An active SQLite database connection object.
        table_name (str): The name of the table to create or update.
        columns (list of str): A list of column names to add to the table.
        patient_num (list of str or int): Patient identifiers used as primary keys.
        value_matrix (list of list): Matrix containing values to insert or update in the table.
        column_matrix (list of list): Matrix containing corresponding column names for each value.

    """
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

def DFA_analysis(RR,patientNum,data_type,saving_path,plot=True,R_peaks=None):
    """
    Performs DFA analysis with crossover detection on RR intervals.

    Parameters:
        RR (np.ndarray): RR interval time series in seconds.
        patientNum (str or int): Identifier for patient/subject.
        data_type (str): Label used for file naming and plot titles.
        plot (bool): Whether to generate and save the plot.

    Returns:
        H_hat (tuple): (short-term exponent, long-term exponent, crossover point)
        m (np.ndarray): Continuous slope estimates (from alpha-beta filter)
        logn (np.ndarray): Corresponding log(n) values
    """
    if RR.max()>3:
        RR=RR/1000
    RR=(RR-np.mean(RR))/np.std(RR)
    F_s,window_sizes=DFA(RR) # performs DFA on the data
    log_n=np.log10(window_sizes)
    log_F=np.log10(F_s)
    params=detecting_crossover(log_F,log_n)
    cross_point=log_n[params[0]] # gets the crossover point from the params
    m1=params[1]
    m2=params[2]

    H_hat1 = np.polyfit(log_n[np.where(log_n<cross_point)],log_F[np.where(log_n<cross_point)],1,cov=False) # fits linear curve to the first section of the DFA plot
    
    H_hat2 = np.polyfit(log_n[np.where(log_n>=cross_point)],log_F[np.where(log_n>=cross_point)],1,cov=False) # fits linear curve to the second section of the DFA plot
    
    ax,fig=DFA_plot(params,log_n,log_F,H_hat1,H_hat2,patientNum,data_type,plot) # plots the DFA results
    m,logn=plotting_scaling_pattern(log_n,log_F,patientNum,fig,ax,data_type,saving_path)

    H_hat=(m1,m2 ,cross_point) # returns the scaling exponents and crossover point for PPG data
    return H_hat,m,logn

def setup_schema(cur):

    """
    sets up the schema for the database

    Parameters:
        cur: cursor for current database
    Returns: 
        None.
    """
    tables=['Months','Weeks','Activities','Patients','Months_HR','Weeks_HR','Dates','Daily_Vitals','Night_Vitals','ECG_Vitals']

    for table in tables:
        cur.execute(f"DROP TABLE IF EXISTS {table}")

    cur.execute("CREATE TABLE Patients(Patient_ID TEXT,overall_HR_avg FLOAT,scaling_exponent_noise FLOAT,scaling_exponent_linear FLOAT, ECG_scaling_exponent_noise FLOAT, ECG_scaling_exponent_linear FLOAT,crossover_PPG FLOAT, crossover_ECG FLOAT,sd1 FLOAT, sd2 FLOAT, sd_ratio FLOAT, ellipse_area FLOAT, PRIMARY KEY(Patient_ID))")
    cur.execute("CREATE TABLE Months(Month TEXT, PRIMARY KEY(Month))")
    cur.execute("CREATE TABLE Months_HR(Patient_ID TEXT, Month TEXT, avg_HR FLOAT, PRIMARY KEY(Patient_ID, Month), UNIQUE(Patient_ID, Month), FOREIGN KEY(Patient_ID) REFERENCES Patients(Patient_ID), FOREIGN KEY(Month) REFERENCES Months(Month))")
    cur.execute("CREATE TABLE Weeks(Week TEXT, PRIMARY KEY(Week))")
    cur.execute("CREATE TABLE Weeks_HR(Patient_ID TEXT, Week TEXT, avg_HR FLOAT, PRIMARY KEY(Patient_ID, Week),UNIQUE(Patient_ID, Week), FOREIGN KEY(Patient_ID) REFERENCES Patients(Patient_ID), FOREIGN KEY(Week) REFERENCES Weeks(Week))")    
    cur.execute("CREATE TABLE Dates(Date TEXT,Week TEXT, Month TEXT, PRIMARY KEY(Date), FOREIGN KEY(Week) REFERENCES Weeks(Week), FOREIGN KEY(Month) REFERENCES Months(Month))")
    cur.execute("CREATE TABLE Daily_Vitals(Date_Vitals Integer, Patient_ID TEXT, Date TEXT, Day_avg_HR FLOAT, Day_min_HR FLOAT, Day_max_HR FLOAT, Resting_HR FLOAT, Avg_PPG_HRV_Day FLOAT, Std_PPG_HRV_Day FLOAT,UNIQUE(Patient_ID, Date), PRIMARY KEY(Date_Vitals AUTOINCREMENT), FOREIGN KEY(Patient_ID) REFERENCES Patients(Patient_ID), FOREIGN KEY(Date) REFERENCES Dates(Date))")
    cur.execute("CREATE TABLE Activities(Date_Vitals Integer, Avg_Active_HR FLOAT, PRIMARY KEY(Date_Vitals), FOREIGN KEY(Date_Vitals) REFERENCES Daily_Vitals(Date_Vitals))")
    cur.execute("CREATE TABLE Night_Vitals(Date_Vitals Integer, Night_avg_HR FLOAT, Night_min_HR FLOAT, Night_max_HR FLOAT, Avg_PPG_HRV_Night FLOAT, Std_PPG_HRV_Night, PRIMARY KEY(Date_Vitals), FOREIGN KEY(Date_Vitals) REFERENCES Daily_Vitals(Date_Vitals))")
    cur.execute("CREATE TABLE ECG_Vitals(Date_Plus_Hour TEXT, Avg_ECG_HRV FLOAT, Std_ECG_HRV FLOAT,sd1 FLOAT, sd2 FLOAT, sd_Ratio FLOAT, ellipse_area FLOAT, Date_Vitals Integer, PRIMARY KEY(Date_Plus_Hour), FOREIGN KEY(Date_Vitals) REFERENCES Daily_Vitals(Date_Vitals))")




def databasing(metrics,Flag=True):
    """
    Creates and populates a SQLite normalised relational database with various patient or volunteer heart rate metrics.

    Parameters:
        metrics (dict): A dictionary containing all heart rate and patient-related metrics. Expected keys include:
            - 'Patient_num': list of patient IDs
            - 'months': list of lists of month labels (e.g., ['Jan 2025', ...])
            - 'avg_hr_per_month': list of lists of average HR values per month
            - 'weeks': list of lists of week labels (e.g., ['2025-W12', ...])
            - 'avg_hr_per_week': list of lists of average HR values per week
            - 'activities': list of lists of activity date strings
            - 'avg_hr_active': list of lists of average HRs during activity
            - 'avg_hr_night': list of average night heart rates
            - 'avg_hr_overall': list of overall average heart rates
            - 'scaling_exponent_noise', 'scaling_exponent_linear': PPG DFA exponents
            - 'ECG_scaling_exponent_noise', 'ECG_scaling_exponent_linear': ECG DFA exponents
            - 'crossover_PPG', 'crossover_ECG': crossover points for DFA plots
            - 'day_dates', 'night_dates', 'day_avg', 'night_avg', 'day_min', 'night_min', 'day_max', 'night_max', 'resting_hr', 'avg_PPG_HRV_day', 'std_PPG_HRV_day',
              'avg_PPG_HRV_night', 'std_PPG_HRV_night', 'avg_ECG_HRV', 'std_ECG_HRV': time-series metrics
        patient (bool): If True, saves data to `patient_metrics.db`; else saves to `volunteer_metrics.db`.

    Notes:
        - Existing tables will be dropped before being recreated.

    """
    db_name = 'patient_metrics.db' if Flag else 'volunteer_metrics.db'
    con = sqlite3.connect(db_name)
    cur = con.cursor()
    setup_schema(cur)
    patient_num=metrics['Patient_num'] # gets the patient number from the metrics dictionary

        
    for idx, patient_id in enumerate(metrics['Patient_num']):
        cur.execute("INSERT INTO Patients(Patient_ID,overall_HR_avg,scaling_exponent_noise,scaling_exponent_linear,ECG_scaling_exponent_noise,ECG_scaling_exponent_linear,crossover_PPG,crossover_ECG,sd1,sd2,sd_ratio,ellipse_area) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",(metrics['Patient_num'][idx],metrics['avg_hr_overall'][idx],metrics['scaling_exponent_noise'][idx],metrics['scaling_exponent_linear'][idx],metrics['ECG_scaling_exponent_noise'][idx],metrics['ECG_scaling_exponent_linear'][idx],metrics['crossover_PPG'][idx],metrics['crossover_ECG'][idx],float(metrics['PPG_sd1'][idx]),float(metrics['PPG_sd2'][idx]),float(metrics['PPG_sd_ratio'][idx]),float(metrics['PPG_ellipse_area'][idx])))

        try:

            months=metrics['months'][idx]
            avg_hrs=metrics['avg_hr_per_month'][idx]
            for month_id, avg_hr in zip(months,avg_hrs):
                cur.execute("""INSERT OR IGNORE INTO Months_HR(Patient_ID, Month, avg_HR) VALUES(?,?,?)""", (patient_id,month_id,avg_hr))
            weeks=metrics['weeks'][idx]
            avg_hrs=metrics['avg_hr_per_week'][idx]
            for week_id, avg_hr in zip(weeks,avg_hrs):
                cur.execute("""INSERT OR IGNORE INTO Weeks_HR(Patient_ID, Week, avg_HR) VALUES(?,?,?)""", (patient_id,week_id,avg_hr))
        except Exception as e:
            print(f'error occurred with :{e}')
    
        try:
            dates=metrics['day_dates'][idx]
            print('day dates',dates)
            day_avg_hrs=metrics['day_avg'][idx]
            day_min_hrs=metrics['day_min'][idx]
            day_max_hrs=metrics['day_max'][idx]
            resting_hrs=metrics['resting_hr'][idx]
            avg_PPG_HRV_Days=metrics['avg_PPG_HRV_day'][idx]
            std_PPG_HRV_Days=metrics['std_PPG_HRV_day'][idx]
            

            for date_id,day_avg_hr,day_min_hr,day_max_hr,resting_hr,avg_PPG_HRV_Day,std_PPG_HRV_Day in zip(dates,day_avg_hrs,day_min_hrs,day_max_hrs,resting_hrs,avg_PPG_HRV_Days,std_PPG_HRV_Days):
                cur.execute("""INSERT OR IGNORE INTO Daily_Vitals(Patient_ID, Date, Day_avg_HR, Day_min_HR, Day_max_HR, Resting_HR, Avg_PPG_HRV_Day, Std_PPG_HRV_Day) VALUES(?,?,?,?,?,?,?,?)""", (patient_id,date_id,day_avg_hr,day_min_hr,day_max_hr,resting_hr,avg_PPG_HRV_Day, std_PPG_HRV_Day))
        except Exception as e:
            print(f'error occurred with :{e}')

        try:
            avg_active_hrs=metrics['avg_hr_active'][idx]
            active_dates=metrics['activities'][idx]   
            
            for avg_active_hr,active_date in zip(avg_active_hrs,active_dates):
                date_vitals_id=cur.execute("""SELECT Date_Vitals FROM Daily_Vitals WHERE Patient_ID=? AND Date=?""",(patient_id,active_date)).fetchone()
                if date_vitals_id:
                    cur.execute("""INSERT OR IGNORE INTO Activities(Date_Vitals, Avg_Active_HR) VALUES(?,?)""", (date_vitals_id[0],avg_active_hr))
        except Exception as e:
            print(f'error occurred with :{e}')
    
        try:
            
            night_dates=metrics['night_dates'][idx]
            night_avg_hrs=metrics['night_avg'][idx]
            night_min_hrs=metrics['night_min'][idx]
            night_max_hrs=metrics['night_max'][idx]
            avg_PPG_HRV_Nights=metrics['avg_PPG_HRV_night'][idx]
            std_PPG_HRV_Nights=metrics['std_PPG_HRV_night'][idx]
            for night_date,night_avg_hr,night_min_hr,night_max_hr,avg_PPG_HRV_Night,std_PPG_HRV_Night in zip(night_dates,night_avg_hrs,night_min_hrs,night_max_hrs,avg_PPG_HRV_Nights,std_PPG_HRV_Nights):
                date_vitals_id=cur.execute("""SELECT Date_Vitals FROM Daily_Vitals WHERE Patient_ID=? AND Date=?""",(patient_id,night_date)).fetchone()
                if date_vitals_id:
                    cur.execute("""INSERT OR IGNORE INTO Night_vitals(Date_Vitals,Night_avg_HR, Night_min_HR, Night_max_HR, Avg_PPG_HRV_Night,Std_PPG_HRV_Night) VALUES(?,?,?,?,?,?)""",(date_vitals_id[0],night_avg_hr,night_min_hr,night_max_hr,avg_PPG_HRV_Night,std_PPG_HRV_Night))
            
        except Exception as e:
            print(f'error occurred with :{e}')

        try:
            dates_plus_hours=metrics['ECG_dates_and_hours'][idx]
            avg_ECG_HRVs=metrics['avg_ECG_HRV'][idx]
            std_ECG_HRVs=metrics['std_ECG_HRV'][idx]
            sd1s=metrics['ECG_sd1'][idx]
            sd2s=metrics['ECG_sd2'][idx]
            sd_ratios=metrics['ECG_sd_ratio'][idx]
            ellipse_areas=metrics['ECG_ellipse_area'][idx]
            for date_plus_hour,avg_ECG_HRV,std_ECG_HRV,sd1,sd2,sd_ratio,ellipse_area in zip(split_on_colon(dates_plus_hours),avg_ECG_HRVs,std_ECG_HRVs,sd1s,sd2s,sd_ratios,ellipse_areas):
                
                date_vitals_id=cur.execute("""SELECT Date_Vitals FROM Daily_Vitals WHERE Patient_ID=? AND Date=?""",(patient_id,date_plus_hour[0])).fetchone()
                if date_vitals_id:
                    date_with_hour=':'.join(date_plus_hour)
                    cur.execute("""INSERT OR IGNORE INTO ECG_Vitals(Date_Plus_Hour, Avg_ECG_HRV, Std_ECG_HRV,sd1,sd2,sd_ratio,ellipse_area, Date_Vitals) VALUES(?,?,?,?,?,?,?,?)""",(date_with_hour, avg_ECG_HRV, std_ECG_HRV,sd1,sd2,sd_ratio,ellipse_area, date_vitals_id[0]))
        
        except Exception as e:
            print(f'error occured with: {e}')

    try:
        #unique months in the metrics dictionary
        months=sorted(np.unique(np.concatenate(metrics['months'])),key=lambda x: datetime.strptime(x, '%b %Y'))

        for month in months:
            cur.execute("""INSERT OR IGNORE INTO Months(Month) VALUES(?)""", (month,))
    except Exception as e:
        print(f'No month data:{e}')

    try:
        weeks=sorted(np.unique(np.concatenate(metrics['weeks'])),key=lambda x: datetime.strptime(x, '%Y-W%W')) #unique weeks in the metrics dictionary
        for week in weeks:
            cur.execute("""INSERT OR IGNORE INTO Weeks(Week) VALUES(?)""", (week,))
    except Exception as e:
        print(f'No week data:{e}')

    try:
        dates=sorted(np.unique(np.concatenate(metrics['day_dates'])),key=lambda x: datetime.strptime(x, '%Y-%m-%d'))
        for date in dates:
            cur.execute("""INSERT OR IGNORE INTO Dates(Date,Week,Month) VALUES(?,?,?)""",(date,pd.to_datetime(date).strftime('%G-W%V'),pd.to_datetime(date).strftime('%b %Y')))
    except Exception as e:
        print(f'No day data:{e}')

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

def DFA_plot(params,log_n,log_F,H_hat1,H_hat2,patientNum,type,plot): 
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
    if not plot: 
        return None,None
     
    cross_indx,a1,a2=params
    cross_point=log_n[cross_indx]
    fig,ax=plt.subplots(2,1,figsize=(12, 8),layout='constrained')
    # plots log-log of DFA analysed data against n, cutting off the end where linearality is lost
    # plt.plot(np.log10(lag)[np.where(np.log10(lag)<3)], log_F[np.where(np.log10(lag)<3)], 'o',label='DFA data')
    ax[0].plot(log_n, log_F, 'o',label='DFA data',color='g')
    ax[0].set_xlabel('logged size of integral slices')
    ax[0].set_ylabel('log of fluctuations from fit for each slice size')
    
    ax[0].set_title('DFA - Measure of Randomness in HRV {} Patient {}'.format(type,patientNum))
    ax[0].axvline(log_n[cross_indx], color='r', linestyle='--', label="Crossover point")
    # closest=(np.abs(lag - len(RR)/10)).argmin()
    # A_L=10**(np.log10(dfa[closest])-2*np.log10(lag[closest]))
    # Linear_trend=(A_L)*lag**2
    # plt.plot(log_n,np.log10(Linear_trend))
    ax[0].grid(True)
    ax[0].legend()
    # fits a linear line to the log-log graph
    # H_hat = np.polyfit(log_n[np.where(log_n<3)],np.log10(dfa)[np.where(log_n<3)],1,cov=False)
    # plt.plot(log_n[np.where(log_n<3)],log_n[np.where(log_n<3)]*H_hat[0]+H_hat[1],color='red')
    
    regression_line_1=log_n[np.where(log_n<cross_point)]*H_hat1[0]+H_hat1[1]
    ax[0].plot(log_n[np.where(log_n<cross_point)],regression_line_1,color='red')
    regression_line_2=log_n[np.where(log_n>=cross_point)]*H_hat2[0]+H_hat2[1]
    ax[0].plot(log_n[np.where(log_n>=cross_point)],regression_line_2,color='blue')
    ax[0].axvline(log_n[cross_indx], color='r', linestyle='--')

    
    return ax[1],fig


def appending_metrics(metrics_dict,fields,patientNum,data):
    """
    Appends data for a set of fields into the corresponding lists in a cumulative metrics dictionary.

    This function safely handles both dictionaries and DataFrames to populate `metrics_dict`,
    inserting default empty lists when data is missing or an error occurs. It is commonly used to
    collect structured time-series or summary statistics for each patient.

    Parameters:
    -----------
    metrics_dict : dict
        The main dictionary where aggregated patient metrics are stored. Each key should already
        be initialized to a list.

    fields : list of str
        A list of keys to be extracted from the `data` object and appended into `metrics_dict`.

    patientNum : str or int
        Identifier for the patient, used for logging errors.

    data : dict or pandas.DataFrame
        A structured data source (e.g., output of heart rate feature analysis). If `dict`, values
        are accessed directly by key. If `DataFrame`, values are accessed and converted to numpy arrays.

    Returns:
    --------
    dict
        The updated `metrics_dict` with new entries for the specified fields, or empty entries
        if the data was unavailable or caused an exception.

    Notes:
    ------
    - If `data` is missing a specified field or contains errors, an empty list is appended in its place.
    - Useful when appending daily summaries, night stats, or other batch features from different stages.
    """
    if type(data)==dict:
        try:
            for key in fields:
                metrics_dict[key].append(data[key])
        except Exception as e:
            print(f"day_data error for patient {patientNum}: {e}")
            for key in fields:
                metrics_dict[key].append([])
    elif type(data)==pd.core.frame.DataFrame:
        try:
            for key in fields:
                metrics_dict[key].append(data[key].to_numpy())
        except Exception as e:
            print(f"day_data error for patient {patientNum}: {e}")
            for key in fields:
                metrics_dict[key].append([])
    return metrics_dict


def adding_to_dictionary(metrics,patientNum,RR,H_hat,H_hat_ECG,ECG_df):
    """
    Aggregates heart rate and DFA metrics for a given patient and appends them to the main metrics dictionary.

    This function consolidates per-patient heart rate variability metrics, daily/night statistics,
    and DFA results from both PPG and ECG modalities. The results are appended to a centralized
    dictionary that aggregates across all patients for later analysis or export.

    Parameters:
    -----------
    metrics : dict
        The central metrics dictionary that is being incrementally built for all patients.
        Keys must already be initialized with empty lists.

    patientNum : str or int
        The unique ID for the patient being analyzed.

    RR : dict
        Dictionary of heart rate variability and daily features, usually returned from `plotting()` or similar.
        Expected keys include:
            - 'avg_hr_per_week', 'avg_hr_per_month', 'avg_hr_overall', 'avg_hr_active',
            - 'months', 'weeks', 'activities',
            - 'resting_and_days' (DataFrame), 'nights' (DataFrame)

    H_hat : tuple of float
        DFA results from PPG analysis, formatted as:
            (scaling_exponent_noise, scaling_exponent_linear, crossover_point_PPG)

    H_hat_ECG : tuple of float
        DFA results from ECG analysis, formatted as:
            (scaling_exponent_noise, scaling_exponent_linear, crossover_point_ECG)

    Returns:
    --------
    dict
        The updated `metrics` dictionary with all relevant fields from this patient added.

    Notes:
    ------
    - If certain expected substructures are missing (like 'resting_and_days' or 'nights'), default
      empty values will be appended.
    - Internally uses `appending_metrics()` to handle structured fields safely.
    - This is intended to be called once per patient during a full cohort analysis loop.
    """

    metrics['Patient_num'].append(patientNum)
    metrics=appending_metrics(metrics,['avg_hr_per_week','avg_hr_per_month','avg_hr_overall','avg_hr_active','months','weeks','activities'],patientNum,RR)
    day_df=RR['resting_and_days']
    night_df=RR['nights']
    poincare_df=RR['poincare']
    metrics=appending_metrics(metrics,['day_dates','day_avg','day_min','day_max','resting_hr','avg_PPG_HRV_day','std_PPG_HRV_day'],patientNum,day_df)
    metrics=appending_metrics(metrics,['night_dates','night_avg','night_min','night_max','avg_PPG_HRV_night','std_PPG_HRV_night'],patientNum,night_df)
    metrics=appending_metrics(metrics,['PPG_sd1','PPG_sd2','PPG_sd_ratio','PPG_ellipse_area'],patientNum,poincare_df)
    metrics=appending_metrics(metrics,['ECG_dates_and_hours','avg_ECG_HRV','std_ECG_HRV','ECG_sd1','ECG_sd2','ECG_sd_ratio','ECG_ellipse_area'],patientNum,ECG_df)


    metrics['scaling_exponent_noise'].append(H_hat[0])
    metrics['scaling_exponent_linear'].append(H_hat[1])
    metrics['crossover_PPG'].append(H_hat[2])



    metrics['ECG_scaling_exponent_noise'].append(H_hat_ECG[0])
    metrics['ECG_scaling_exponent_linear'].append(H_hat_ECG[1])
    metrics['crossover_ECG'].append(H_hat_ECG[2])


    return metrics

def alpha_beta_filter(x,y,Q=500):
    """
    Applies an adaptive alpha-beta filter to estimate the smoothed signal and its first derivative (slope).

    This filter is designed to reduce noise in a signal while tracking its underlying trend (gradient).
    The alpha and beta coefficients are dynamically computed based on the current index and capped at `Q`
    to prevent instability over long sequences.

    Parameters:
        x (np.ndarray): 1D array of independent variable values (e.g., time).
        y (np.ndarray): 1D array of noisy measurements or signal values corresponding to `x`.
        Q (int): The maximum effective "memory" window size (default is 500). Beyond this, alpha/beta stop adapting.

    Returns:
        m_est (np.ndarray): Estimated first derivative (slope) at each point in `x`.
        G_est (np.ndarray): Smoothed signal estimate corresponding to each `x`.

    Notes:
        - This is a discrete version of the alpha-beta filter adapted for uniformly spaced data.
        - For large Q, the filter responds slower to changes but is more stable.
        - If Q is small, the filter is more responsive but potentially noisier.

    """
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

def interpolating_for_uniform(logn,log_f,min,max):
    """
    Interpolates and smooths log-log data onto a uniformly spaced grid using a spline.

    This function takes log-transformed x (`logn`) and y (`log_f`) data, and fits a smoothing spline.
    The resulting smoothed data is evaluated on a uniformly spaced array of `logn` values, which
    can be useful for further numerical analysis, plotting, or regression.

    Parameters:
        logn (np.ndarray): Log-transformed x-values (e.g., log(window sizes)).
        log_f (np.ndarray): Log-transformed y-values (e.g., log(fluctuation function)).

    Returns:
        uniform_log_n (np.ndarray): Uniformly spaced logn values (100 points between logn.min() and logn.max()).
        smoothed_log_f (np.ndarray): Smoothed and interpolated log_f values corresponding to `uniform_log_n`.

    Notes:
        - Uses a `UnivariateSpline` with `s=0` (interpolating spline, no smoothing).

    """
    uniform_log_n=np.linspace(min,max,100)
    spline=UnivariateSpline(logn,log_f,s=0)
    smoothed_log_f=spline(uniform_log_n)
    return uniform_log_n,smoothed_log_f

def plotting_scaling_pattern(log_n,log_f,patient_num,fig,ax,type,saving_path):
    """
    Plots the local scaling exponent (slope) across scales using alpha-beta filtering on DFA data.

    This function:
    - Interpolates the log-log DFA data for uniform spacing.
    - Applies an alpha-beta filter to estimate the local slope (scaling exponent).
    - Optionally plots the resulting local slope vs log(window size) on the provided axes.
    - Saves the plot to a file if plotting is enabled (i.e., ax is not None).

    Parameters:
        log_n (np.ndarray): Log-transformed window sizes from DFA.
        log_f (np.ndarray): Log-transformed fluctuation function values from DFA.
        patient_num (str): Identifier for the patient (used for saving the plot).
        ax (matplotlib.axes.Axes or None): Axes object to draw the plot on. If None, no plot is generated.
        type (str): A label specifying whether the data is from PPG, ECG, etc. Used in the filename.

    Returns:
        m (np.ndarray): Estimated local scaling exponent at each interpolated log_n.
        interpolated[0] (np.ndarray): Interpolated log_n values (uniformly spaced).

    Notes:
        - Dashed lines at 0.5, 1.0, and 1.5 are plotted as visual references.

    """
    interpolated=interpolating_for_uniform(log_n,log_f,log_n.min(),log_n.max())
    mask=np.where(interpolated[0]>0.55)
    m,log_f=alpha_beta_filter(*interpolated)
    if ax is None:
        return m,interpolated[0]
    if np.max(m[mask])>2:
        ax.set_ylim(0,np.max(m[mask])+0.5)
    else:
        ax.set_ylim(0,2)
    ax.plot(interpolated[0][mask],m[mask])
    ax.axhline(1,linestyle='dashed',color='k')
    ax.axhline(0.5,linestyle='dashed',color='k')
    ax.axhline(1.5,linestyle='dashed',color='k')
    ax.set_xlabel('logged size of integral slices')
    ax.set_ylabel('gradient at each value of n - $m_{e}(n)$')
    ax.set_title('continuous gradient over the DFA plot')
    plt.show()
    Path(f"{saving_path}/Graphs").mkdir(exist_ok=True)
    Path(f"{saving_path}/Graphs/scaling_patterns/").mkdir(exist_ok=True) # creating new directory
    fig.savefig(f'{saving_path}/Graphs/scaling_patterns/{patient_num}-{type}.png')
    plt.close()
    return m,interpolated[0]


def ECG_HRV_info(ECG_RR,ECG_R_times):
    """
    Determins the avg HRV and std for each ECG and returns it in a pandas dataframe

    Parameters:
        ECG_RR: np.ndarray
            contains RR intervals for all ECGs
        ECG_R_times: np.ndarray
            contains the dates of each ECG
    Returns:
        ECG_RR_df: pd.Dataframe
    
    """
    ECG_RR_data=[]    
    for i,RR in enumerate(ECG_RR.T):
        date=pd.to_datetime(ECG_R_times[i],format='ISO8601',utc=True).strftime('%Y-%m-%d:%H')
        RR = RR[~np.isnan(RR)] # removes nans
        ECG_RR_data.append({'avg_ECG_HRV':np.mean(RR),
                            'std_ECG_HRV':np.std(RR),
                            'ECG_dates_and_hours':date})
    ECG_RR_df=pd.DataFrame(ECG_RR_data)
    return ECG_RR_df

def poincare_plot_analysis(RR_intervals,input_type='',patient_number=None,poincare_flag=True):
    poincare_data=[]
    for i,peaks in enumerate(RR_intervals.T):
        peaks = peaks[~np.isnan(peaks)]
        try:
            dic=poincare_plot(peaks,input_type=input_type,patient_number=patient_number,plot_on=poincare_flag)
            poincare_data.append(dic)
        except:
           continue
    poincare_df=pd.DataFrame(poincare_data)
    return poincare_df

def ECG_HRV(ECG_RR,ECG_R_times,ECG_R_peaks,patientNum,saving_path,poincare_flag):
    """
    Processes and visualizes ECG-based RR interval data, before and after outlier removal.

    Parameters:
        ECG_RR (np.ndarray): 2D array of RR intervals from ECG data.
        patientNum (str or int): Identifier for the patient (used for file saving path).
        saving_path (str): directory path to save plots to.

    Returns:
        np.ndarray: Cleaned, flattened RR interval array.
    """
    ECG_df=ECG_HRV_info(ECG_RR,ECG_R_times)
    poincare_df=poincare_plot_analysis(ECG_RR,input_type='ECG',patient_number=patientNum,poincare_flag=poincare_flag)
    ECG_df=ECG_df.join(poincare_df)
    ECG_RR=(ECG_RR[:,:len(ECG_RR[0])-1].T).flatten()
    ECG_RR = ECG_RR[~np.isnan(ECG_RR)] # removes nans
    fig,ax=plt.subplots(1,2,figsize=(12,6),layout='constrained')
    ax[0].plot(np.arange(0,len(ECG_RR)),ECG_RR)
    ax[0].set_title('ECG HRV')
    ax[0].set_ylabel('RR interval (s)')
    ax[0].set_xlabel('Beats')
    mask=detecting_outliers(ECG_RR)
    ECG_RR=ECG_RR[mask] # removes outliers
    
    ax[1].plot(np.arange(0,len(ECG_RR)),ECG_RR)
    ax[1].set_title('ECG HRV Filtered')
    ax[1].set_ylabel('RR interval (s)')
    ax[1].set_xlabel('Beats')
    Path(f"{saving_path}/heartRateRecord{patientNum}").mkdir(exist_ok=True) # creating new directory
    fig.savefig(f"{saving_path}/heartRateRecord{patientNum}/ECG_HRV.png")
    plt.close()
    return ECG_RR,ECG_df # returns the RR intervals for the ECG data

def avg_scaling_pattern(scaling_patterns):
    """
    Computes the average and standard deviation of scaling gradients and log window sizes
    across multiple patients or time series.

    Parameters:
        scaling_patterns (DataFrame): A DataFrame with columns 'gradient' and 'log_n',
                                      where each row contains lists of values for a subject.

    Returns:
        avg_gradient (np.ndarray): Mean gradient values across subjects
        avg_log_n (np.ndarray): Mean log(window size) values across subjects
        std (np.ndarray): Standard deviation of gradient values
    """
    gradient= scaling_patterns['gradient']
    valid_gradient=gradient[gradient.apply(lambda x: len(x) > 0)]
    avg_gradient=np.mean(np.array(valid_gradient.tolist()),axis=0)  # Calculate the average gradient across all patients
    log_n=scaling_patterns['log_n']
    valid_log_n=log_n[log_n.apply(lambda x: len(x) > 0)]  # Filter out empty lists
    avg_log_n=np.mean(np.array(valid_log_n.tolist()),axis=0)

    std=np.std(np.array(valid_gradient.tolist()),axis=0)


    return avg_gradient, avg_log_n, std


def plotting_scaling_pattern_difference(scaling_patterns1,scaling_patterns2,type1,type2,saving_path,patient=True,DFA_on=True):
    """
    Measures the trend in differences between scaling patterns over patients and plots the differences for each patient, overlayed to see general pattern.
    Also calcualates the maximum sepreation in scaling pattern difference and produces a mean difference plot.

    Parameters:
        scaling_patterns (DataFrame): DataFrame containing 'gradient' and 'log_n' columns.
        patientNum (str): Patient number for labeling the plot.
        type1 (str), type2 (str): Type of data (e.g., 'PPG', 'ECG') for labeling the plot.
        saving_path (str): directory path to save plots to.
        patient (bool): determins if current plot is for a patient or a volunteer for accurate file naming on saving.
        DFA_on (bool): determins if a plot should be created.
    
    
    """
    if not DFA_on:
        print('DFA_on flag must be activated to plot the scaling pattern differences')
        return
    if scaling_patterns1.empty:
        print('missing ECG data')
        return
    if scaling_patterns2.empty:
        print('missing PPG data')
        return
    print(np.shape(len(scaling_patterns1['gradient'])))
    merged=pd.merge(scaling_patterns1,scaling_patterns2,on='patient_num',suffixes=('_1','_2'))
    merged['gradient_diff']=merged['gradient_1']-merged['gradient_2']
    merged['log_n_diff']=merged.apply(lambda row: (row['log_n_1']+row['log_n_2'])/2,axis=1)
    print(merged['patient_num'].to_list())
    fig,ax=plt.subplots(3,1,figsize=(18,14), gridspec_kw={'height_ratios': [3, 3, 3]})
    viridis=plt.colormaps['viridis']
    new_colors=viridis(np.linspace(0,1,len(merged['patient_num'])))
    diff_matrix=np.vstack(merged['gradient_diff'].to_numpy())
    mean_diff=np.mean(diff_matrix,axis=0)
    sem_diff=np.std(diff_matrix,axis=0)/np.sqrt(diff_matrix.shape[0])
    log_n=np.mean(np.vstack(merged['log_n_diff'].to_numpy()),axis=0)
    mask=log_n>0.55
    ax[2].plot(log_n[mask],mean_diff[mask],label='Mean Difference', color='black')
    ax[2].fill_between(log_n[mask],mean_diff[mask]-sem_diff[mask],mean_diff[mask]+sem_diff[mask],color='grey',alpha=0.4)
    ax[2].axhline(0,color='k',linestyle='--')
    ax[2].legend()
    ax[2].set_xlabel('logged size of integral slices')
    ax[2].set_ylabel('difference in gradient at each value of n -$\\Delta m_e(n)$')
    ax[2].set_title('Mean difference Plot')
    
    for i,row in merged.iterrows():
        x=row['log_n_diff']
        y=row['gradient_diff'][x>0.55]
        mask=(x>2)
        percentage_through=i/len(merged['patient_num'])
        ax[0].plot(x[x>0.55],y,label=f"Patient Number {row['patient_num']}",color=new_colors[i])
    ax[0].set_title('Difference in scaling patterns between ECG and PPG - $m_{ECG}-m_{PPG}$')
    ax[0].set_xlabel('logged size of integral slices')
    ax[0].set_ylabel('difference in gradient at each value of n -$\\Delta m_e(n)$')
    ax[0].grid()
    ax[0].axhline(0,linestyle='dashed',color='k')
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    log_n_in_range,gradient_range=scaling_pattern_difference_analysis(merged,ax)
    ax[1].plot(log_n_in_range,gradient_range,color='red', label='gradient range', linewidth=3, zorder=5, alpha=0.5)
    ax[1].legend()
    ax[1].set_xlabel('logged size of itegral slices')
    ax[1].set_ylabel('range in scaling pattern difference')
    ax[1].set_title('Range of scaling pattern')
    ax[1].grid()

    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(hspace=0.4)  # Optional fine-tuning
    plt.show()
    fig.savefig(f"{saving_path}/Graphs/scaling_patterns/{'patients' if patient==True else 'volunteers'}-{type1}-{type2}-difference.png")
    plt.close()


    


def scaling_pattern_difference_analysis(df,ax):
    """
    function to measure the range of gradient differences over patients to visualise how range changes over log(n)
    interpolates the 2 scaling patterns to make sure that they are the same length so they can be subtracted.

    Parameters:
        df (pd.Dataframe): DataFrame containing the subtracted scaling patterns
    Returns:
        log_n_avg(np.ndarray): Array of mean log(n) values to be used as the x axis
        gradient_range(np.ndarrray): Array of ranges of gradient_differences for each value of log(n)
    """
    print(df)
    df[['interpolated_log_n','interpolated_gradient']]=df.apply(lambda row: interpolating_for_uniform(np.array(row['log_n_diff']), np.array(row['gradient_diff']),2,3), axis=1,result_type='expand')
    # analysing_data=np.array(list(zip(df['interpolated_log_n'],df['interpolated_gradient'].to_numpy())))
    # print(analysing_data)
    # print(analysing_data[:,0])
    log_n_array = np.vstack(df['interpolated_log_n'].to_numpy())
    gradient_array = np.vstack(df['interpolated_gradient'].to_numpy())
    log_n_avg = np.mean(log_n_array, axis=0)
    gradient_range = np.max(gradient_array, axis=0) - np.min(gradient_array, axis=0)

    return log_n_avg,gradient_range



def plotting_average_scaling_pattern(scaling_patterns1,scaling_patterns2,type1,type2,patient=True,DFA_on=True):
    """
    Plots the average scaling pattern from a DataFrame of scaling patterns.

    Parameters:
        scaling_patterns (DataFrame): DataFrame containing 'gradient' and 'log_n' columns.
        patientNum (str): Patient number for labeling the plot.
        type (str): Type of data (e.g., 'PPG', 'ECG') for labeling the plot.

    Returns: 
        None
    """
    if not DFA_on:
        print('DFA_on flag must be activated to plot the average scaling patterns')
        return np.nan,np.nan,np.nan,np.nan
    if scaling_patterns1.empty:
        print('missing PPG data')
        return np.nan,np.nan,np.nan,np.nan
    if scaling_patterns2.empty:
        print('missing ECG data')
        return np.nan,np.nan,np.nan,np.nan
    
    avg_gradient1, avg_log_n1, std1 = avg_scaling_pattern(scaling_patterns1)
    avg_gradient2, avg_log_n2, std2 = avg_scaling_pattern(scaling_patterns2)
    

    try:
        mask=np.where(avg_log_n1>0.55)
    except:
        print('total_on must be activated for PPG HRV data to be extracted')
        return np.nan,np.nan,np.nan,np.nan
    fig,ax=plt.subplots(2,1,figsize=(12, 8),layout='constrained')
    if np.max(avg_gradient1[mask])>2:
        ax[0].set_ylim(0,np.max(avg_gradient1[mask])+0.5)
    elif np.max(avg_gradient2[mask])>2:
        ax[0].set_ylim(0,np.max(avg_gradient2[mask])+0.5)
    else:
        ax[0].set_ylim(0,2)

    print('avg grad 1',avg_gradient1)
    print('avg grad 2',avg_gradient2)
    print('avg log 1',avg_log_n1)
    print('avg log 2',avg_log_n2)
    ax[0].errorbar(avg_log_n1[mask], avg_gradient1[mask], yerr=std1[mask], fmt='-', label=f'Average Scaling Pattern - {type1}', color='blue', capsize=5,zorder=1)
    ax[0].errorbar(avg_log_n2[mask], avg_gradient2[mask], yerr=std2[mask], fmt='-', label=f'Average Scaling Pattern - {type2}', color='orange', capsize=5,zorder=1)

    r,p=spearmanr(avg_gradient1[mask],avg_gradient2[mask])
    print(f'spearman correlation coefficient={r}')
    ax[0].axhline(1,linestyle='dashed',color='k')
    ax[0].axhline(0.5,linestyle='dashed',color='k')
    ax[0].axhline(1.5,linestyle='dashed',color='k')
    ax[0].set_xlabel('logged size of integral slices')
    ax[0].set_ylabel(f'Average gradient at each value of n - $\\overline{{m}}_e(n)$')
    ax[0].set_title(f'Average Scaling Pattern for {type1} and {type2}')
    ax[0].legend()
    ax[0].grid(zorder=0)
    ax[1].scatter(avg_gradient1[mask],avg_gradient2[mask])
    ax[1].set_xlabel('PPG scaling pattern')
    ax[1].set_ylabel('ECG scaling pattern')
    ax[1].set_title('Correlation plot for scaling patterns')
    plt.show()
    fig.savefig(f"/data/t/smartWatch/patients/completeData/DamianInternshipFiles/Graphs/scaling_patterns/{'patients' if patient==True else 'volunteers'}-{type1}-{type2}-average.png")
    plt.close()
    return avg_gradient1,avg_gradient2,avg_log_n1,avg_log_n2

# %%
def main():
    from scipy.fft import fft, ifft, fftshift, ifftshift
    from scipy.interpolate import interp1d
    flags=namedtuple('Flags',['months','weeks','activities','total','day_night','plot_DFA','poincare_plot_on','patient_analysis'])
    Flags=flags(False,False,False,False,False,True,True,True)
    if Flags.plot_DFA:
        iterable=[Flags.months,Flags.weeks,Flags.activities,True,Flags.day_night,Flags.plot_DFA,Flags.poincare_plot_on,Flags.patient_analysis]
        Flags=flags._make(iterable)

    patient_data_path="/data/t/smartWatch/patients/completeData/patData"
    volunteer_data_path="/data/t/smartWatch/patients/volunteerData"
    saving_path="/data/t/smartWatch/patients/completeData/DamianInternshipFiles"



    if Flags.patient_analysis==True:
        data_path=patient_data_path
    else:
        data_path=volunteer_data_path
    
    
    # dictionary storing all patient data calcualted in the code to be outputted to db
    metrics={'Patient_num':[],
                'avg_hr_per_month':[],
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
                'day_dates':[],
                'night_dates':[],
                'day_avg':[],
                'night_avg':[],
                'day_min':[],
                'night_min':[],
                'day_max':[],
                'night_max':[],
                'resting_hr':[],
                'avg_PPG_HRV_day':[],
                'std_PPG_HRV_day':[],
                'avg_PPG_HRV_night':[],
                'std_PPG_HRV_night':[],
                'ECG_dates_and_hours':[],
                'avg_ECG_HRV':[],
                'std_ECG_HRV':[],
                'PPG_sd1':[],
                'PPG_sd2':[],
                'PPG_sd_ratio':[],
                'PPG_ellipse_area':[],
                'ECG_sd1':[],
                'ECG_sd2':[],
                'ECG_sd_ratio':[],
                'ECG_ellipse_area':[]}
    surrogate_dictionary={'Patient_num':[],
                          'Surrogate_data_linear':[],
                          'Surrogate_data_noise':[]}
    counter=0
    scaling_pattern_ECG_rows=[]
    scaling_pattern_PPG_rows=[]
    volunteer_nums=['data_001_1636025623','data_CHE_1753362494','data_AMC_1633769065','data_AMC_1636023599','data_LEE_1636026567','data_DAM_1753261083','data_JAS_1753260728','data_CHA_1753276549','data_DAM_1752828759']
    for i in range(2,10):
        print(i)
        if Flags.patient_analysis:
            if i==42 or i==24:
                continue
            if i<10:
                patientNum='0{}'.format(str(i))
            else:
                patientNum='{}'.format(str(i))
        elif i-2<len(volunteer_nums):
            patientNum=volunteer_nums[i-2]
        else:
            break
        print(patientNum)
        try:
            ECG_RR,ECG_R_times,ECG_R_peaks=patient_output(patientNum,patient=Flags.patient_analysis)
            ECG_RR,ECG_df=ECG_HRV(ECG_RR,ECG_R_times,ECG_R_peaks,patientNum,saving_path,Flags.poincare_plot_on)
            if ECG_RR is None or len(ECG_RR)<1000 and Flags.patient_analysis:
                print('not enough ECG data to perform DFA analysis')
                # scaling_patterns_ECG.loc[i]=[[],[]]
                H_hat_ECG=(np.nan,np.nan,np.nan)
            else:
                H_hat_ECG,m,log_n=DFA_analysis(ECG_RR,patientNum,'ECG',saving_path,plot=Flags.plot_DFA,R_peaks=ECG_R_peaks)
                scaling_pattern_ECG_rows.append({'patient_num':patientNum,
                                                 'gradient':m,
                                                 'log_n':log_n
                })
        except Exception as e:
            print(f"ECG error for patient {patientNum}: {e}")
            print(ECG_RR)
            traceback.print_exc()
            H_hat_ECG=(np.nan,np.nan,np.nan)
            ECG_df=pd.DataFrame()
            pass
        try:
            heartRateData_sorted=sortingHeartRate(patientNum,data_path,patient=Flags.patient_analysis)
            RR=plotting(heartRateData_sorted,patientNum,data_path,saving_path,Flags)
            if RR['HRV'].size<1000:
                print('not enough PPG data to perform DFA analysis')
                # scaling_patterns_PPG.loc[i]=[[],[]]
                H_hat=(np.nan,np.nan,np.nan)
            else:
                H_hat,m,log_n=DFA_analysis(RR['HRV'],patientNum,'PPG',saving_path,plot=Flags.plot_DFA)
                scaling_pattern_PPG_rows.append({'patient_num':patientNum,
                                                    'gradient':m,
                                                    'log_n':log_n
                    })
        except Exception as e:
            print(f"PPG error for patient {patientNum}: {e}")
            traceback.print_exc()
            continue
        
        #surrogate_data=surrogate(RR[0])



        
        
        metrics=adding_to_dictionary(metrics,patientNum,RR,H_hat,H_hat_ECG,ECG_df)
    scaling_patterns_ECG=pd.DataFrame(scaling_pattern_ECG_rows)
    scaling_patterns_PPG=pd.DataFrame(scaling_pattern_PPG_rows)
    plotting_scaling_pattern_difference(scaling_patterns_ECG,scaling_patterns_PPG,'ECG','PPG',saving_path,Flags.patient_analysis,Flags.plot_DFA)
    plotting_average_scaling_pattern(scaling_patterns_PPG,scaling_patterns_ECG,'PPG','ECG',Flags.patient_analysis,Flags.plot_DFA)
    #print(surrogate_dictionary)
    #surrogate_databasing(surrogate_dictionary,'IAAFT')
    databasing(metrics,Flags.patient_analysis)
    

if __name__=="__main__":
    main()
    

"""
inputs are /data/t/smartWatch
outputs write to same place
"""
# %%
