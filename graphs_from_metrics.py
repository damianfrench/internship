#%%

import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from scipy.stats import shapiro
from scipy.stats import iqr
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd
from patsy import dmatrix
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def shapiro_testing(data):
    stat, p = shapiro(data,nan_policy='omit') # runs shapiro test
    print(f"Shapiro-Wilk test p-value = {p}")
    print(f"test statistic={stat}")
    if p>0.05:
        print('deviation from normality is not statistically significant')
    else:
        print('deviation from normality is statistically significant')
    return p

def boxplot(data,ax,Name):
    plt.subplot(1,2,2)
    jitter = np.random.normal(loc=0, scale=0.05, size=len(data))  # jitter along x-axis
    plt.scatter(np.ones_like(data) + jitter, data, alpha=0.7, color='black',zorder=3, s=40,label=f'{Name} patients') # plots the patient datapoints
    plt.boxplot(data,
                  vert=True,         # Horizontal boxplot
                  patch_artist=True,  # Fill the box with color
                  boxprops=dict(facecolor='skyblue', color='black'),
                  medianprops=dict(color='red', linewidth=2),
                  whiskerprops=dict(color='black'),
                  capprops=dict(color='black'),
                  flierprops=dict(marker='o', color='black', markersize=5)) # creates boxplots of patient and volunteer data
    plt.legend()
    plt.ylabel('scaling exponent (\u03b1)')
    return ax

def histogram(data,Name):
    p=shapiro_testing(data)
    mean=np.mean(data)
    std=np.std(data)
    IQR=iqr(data)
    median=np.median(data)
    ax,fig=plt.subplots(1,2,figsize=(12,6))
   
    plt.subplot(1,2,1)
    print('mean={:.3f}'.format(mean))
    plt.text(1,3.35, 'mean={:.3f}'.format(mean), fontsize = 10)
    plt.text(1,3.2,'standard dev.={:.3f}'.format(std),fontsize=10)
    plt.text(1,3.05,'IQR={:.3f}'.format(IQR),fontsize=10)
    plt.text(1,2.9,'median={:.3f}'.format(median),fontsize=10)
    
    plt.hist(data, bins='auto', color='skyblue', edgecolor='black', alpha=0.7, density=True,label='distribution of {} (shapiro test p={:.2f})'.format(Name,p)) # generates a histagram of the scaling coefficients
    # fits a normal distribution to the histagram data using KDE to estimate the density function
    xs = np.linspace(min(data), max(data), 200)
    kde = gaussian_kde(data)
    plt.plot(xs, kde(xs), color='black', lw=2,label='gaussian fit') # fits a curve to the histogram
    plt.axvline(np.mean(data),color='red')
    plt.legend()
    plt.xlabel('scaling exponent (\u03b1)')
    plt.ylabel('frequency density')
    ax=boxplot(data,ax,Name)
    
    plt.show()
    plt.savefig('/data/t/smartWatch/patients/completeData/DamianInternshipFiles/Graphs/Hist{}.png'.format(Name))
    plt.close()
    plt.plot(np.linspace(1,len(data),len(data)),data,'x')
    plt.axhline(np.average(data),color='k')
    plt.savefig('/data/t/smartWatch/patients/completeData/DamianInternshipFiles/Graphs/Plot{}.png'.format(Name))
    plt.close()

def weeks_hr(cur,con):
    Patient_data=[]
    for p in np.array(cur.execute("SELECT Number FROM Weeks").fetchall()).flatten(): # gets week hr data from database as 2D array where each row is a patient
        Patient_data.append(np.array(cur.execute("SELECT * FROM Weeks WHERE Number='{}'".format(p)).fetchall()))
    Groups=np.vstack(np.array(Patient_data))[:,0]
    Patient_data=np.delete(np.vstack(np.array(Patient_data)),0,axis=1) # removes the first column which contains indexes
    df=pd.DataFrame(Patient_data)
    df['patient_Id']=Groups
    plt.subplots(1,2,figsize=(12,6))
    weekly_table = df.melt(id_vars='patient_Id', var_name='week', value_name='Heart_rate')
    weekly_table["week"] = pd.to_numeric(weekly_table["week"], errors="coerce")
    weekly_table["Heart_rate"] = pd.to_numeric(weekly_table["Heart_rate"], errors="coerce")
    weekly_table = weekly_table.dropna(subset=['Heart_rate'])
    # patient_counts=weekly_table['patient_Id'].value_counts()
    # patients_drop=patient_counts[patient_counts<3].index
    weekly_table=weekly_table[weekly_table['patient_Id']!='31']
    weekly_table['aligned_week']=weekly_table.groupby('patient_Id')['week'].transform(lambda x: x-x.min()) # creates new series/column in table that numbers each week for each patient starting at 0
    baseline_hr = weekly_table.loc[weekly_table.groupby('patient_Id')['week'].idxmin(), ['patient_Id', 'Heart_rate']]
    baseline_hr=baseline_hr.rename(columns={'Heart_rate': 'Baseline_hr'})
    weekly_table=weekly_table.merge(baseline_hr, on='patient_Id')
    weekly_table['normalised']=weekly_table['Heart_rate']-weekly_table['Baseline_hr'] # creates a normalised series/column where normalised is distance from baseline HR.

    p=shapiro_testing((weekly_table['Heart_rate']))
    model = smf.mixedlm("normalised ~ week", data=weekly_table, groups=weekly_table['patient_Id'], re_formula="~week")
    result = model.fit()
    weekly_table['fitted']=result.predict(weekly_table)
    result=model.fit()
    weekly_table['hr_zNorm']=weekly_table.groupby('patient_Id')['Heart_rate'].transform(lambda x:(x-x.mean())/x.std())
    weekly_table['fitted_zNorm']=weekly_table.groupby('patient_Id')['fitted'].transform(lambda x:(x-x.mean())/x.std()) # also adds a z-score normalisation column for Hr and fitted Hr.
    plt.subplot(1,2,1)
    plt.hist((weekly_table['hr_zNorm']), bins='auto', color='skyblue', edgecolor='black', alpha=0.7, density=True,label='distribution of scaling exponents (shapiro test p={:.4f})'.format(p))
    plt.legend()
    
    Patients=weekly_table['patient_Id'].unique()
    # print(result.summary())
    print(weekly_table)
    
    plt.subplot(1,2,2)
    
    shapiro_testing(weekly_table['hr_zNorm'])
    for patient_Id, group in weekly_table.groupby('patient_Id'):
        plt.plot(group['aligned_week'],group['normalised'],alpha=0.3,label=f'Patient {patient_Id}')
    # for P in Patients:
    #     Patient_data=weekly_table[weekly_table['patient_Id']==P]
    #     plt.scatter(Patient_data['week'],Patient_data['normalised'],label=f'Patient {P}', alpha=0.6)
    #     sorted_patient_data=Patient_data.sort_values('week')
    #     plt.plot(sorted_patient_data['week'],sorted_patient_data['fitted'],linewidth=2)

    plt.savefig('/data/t/smartWatch/patients/completeData/DamianInternshipFiles/Graphs/weekly_hr.png')
    plt.close()

def months_hr(cur,con):
    Patient_data=[]

    for p in np.array(cur.execute("SELECT Number FROM Months").fetchall()).flatten():
        Patient_data.append(np.array(cur.execute("SELECT * FROM Months WHERE Number='{}'".format(p)).fetchall()))
    Groups=np.vstack(np.array(Patient_data))[:,0]
    Patient_data=np.delete(np.vstack(np.array(Patient_data)),0,axis=1)
    #print(Patient_data)
    df=pd.DataFrame(Patient_data)
    df['patient_Id']=Groups
    monthly_table = df.melt(id_vars='patient_Id', var_name='Month', value_name='Heart_rate')
    
    monthly_table["Month"] = pd.to_numeric(monthly_table["Month"], errors="coerce")
    monthly_table["Heart_rate"] = pd.to_numeric(monthly_table["Heart_rate"], errors="coerce")
    monthly_table = monthly_table.dropna(subset=['Heart_rate'])
    model = smf.mixedlm("Heart_rate ~ Month", data=monthly_table, groups=monthly_table['patient_Id'], re_formula="~Month")
    result = model.fit()
    monthly_table['fitted']=result.predict(monthly_table)
    result=model.fit()
    Patients=monthly_table['patient_Id'].unique()
    print(result.summary())

def comparing_scaling_exponents(cur,type):
    se_noise=scaling_exponent_noise(cur,type)
    se_linear=scaling_exponent_linear(cur,type)
    diff=se_linear-se_noise
    x_axis=Patients(cur)
    #x_axis=np.linspace(1,len(se_linear),len(se_linear))
    fig,ax=plt.subplots(1,2,figsize=(14,8))
    #fig.suptitle('{} Surrogate test'.format(type))
    fig.suptitle('Trend–Noise DFA Exponent Differences in Smartwatch Heart Rate Data')
    plt.subplot(1,2,1)
    plt.scatter(x_axis,se_noise,marker='d',color='orange',label='noise dominant')
    plt.scatter(x_axis,se_linear,marker='D',color='blue',label='trend dominant')
    plt.tick_params(axis='x',labelrotation=-90,length=0.1)
    plt.scatter(x_axis,diff,marker='p',color='red',label='trend-noise')
    plt.vlines(x_axis,diff,0,color='red')
    IQR=iqr(se_linear-se_noise,nan_policy='omit')
    plt.axhline(np.nanmean(se_linear-se_noise),color='red')
    #plt.axhline(np.mean(scaling_exponent_linear(cur,'')-scaling_exponent_noise(cur,'')),color='green')
    mean=np.nanmean(se_linear-se_noise)
    # plt.axhline(mean+IQR/2)
    # plt.axhline(mean-IQR/2)
    # plt.axhline(-0.1)
    # plt.axhline(0.1)
    #plt.scatter(x_axis,scaling_exponent_linear(cur,'')-scaling_exponent_noise(cur,''),marker='p',color='green',label='trend-noise HR')
    # plt.vlines(x_axis,scaling_exponent_linear(cur,'')-scaling_exponent_noise(cur,''),np.nanmean(scaling_exponent_linear(cur,'')-scaling_exponent_noise(cur,'')),color='green')
    plt.axhline(0,color='k')
    plt.xlabel('patient number')
    plt.ylabel('scaling exponents')
    plt.title('Scaling Exponents from noise and trend dominated regions')
    plt.legend()
    plt.subplot(1,2,2)
    plt.grid()
    se_linear= se_linear[~np.isnan(se_linear)]
    se_noise= se_noise[~np.isnan(se_noise)]
    v1=violin_plot([se_linear-se_noise],[1],'#D43F3A',alpha=1)
    v2=violin_plot([scaling_exponent_linear(cur,'')-scaling_exponent_noise(cur,'')],[2],color="#683AD4",alpha=1)
    plt.legend([v1['bodies'][0],v2['bodies'][0]], [ 'ECG data','HR data'], loc=3)

    plt.xticks([])
    plt.xlabel('different methods')
    plt.ylabel('calculated values')
    plt.title('Violin plots')
    plt.show()
    plt.savefig('/data/t/smartWatch/patients/completeData/DamianInternshipFiles/Graphs/scalingExponents_comparison{}Filtered.png'.format(type))
    plt.close()

def scaling_exponent_linear(cur,type):
    se=cur.execute("SELECT {}scaling_exponent_linear FROM Patients".format(type)).fetchall() # fetches patient scaling exponents from database
    se=np.array(se).flatten() # database is read above and then the data is converted into a useable array
    return np.array(se,dtype=np.float64)

def scaling_exponent_noise(cur,type):
    se=cur.execute("SELECT {}scaling_exponent_noise FROM Patients".format(type)).fetchall() # fetches patient scaling exponents from database
    se=np.array(se).flatten() # database is read above and then the data is converted into a useable array
    return np.array(se,dtype=np.float64)

def Patients(cur):
    n=cur.execute("SELECT Number from Patients").fetchall()
    n=np.array(n).flatten()
    return n

def surrogate_linear_scaling_exponent(cur,type):
    se=cur.execute("SELECT {}_linear FROM Surrogates".format(type)).fetchall()
    se=np.array(se).flatten()
    return se

def surrogate_noise_scaling_exponent(cur,type):
    se=cur.execute("SELECT {}_noise FROM Surrogates".format(type)).fetchall()
    se=np.array(se).flatten()
    return se

def t_test(data,null):
    print(data,null)
    if type(null)==int:
        t_stat, p_value = ttest_1samp(data,null,nan_policy='omit')
    else:
        t_stat, p_value = ttest_rel(data,null,nan_policy='omit')
    if p_value<0.05:
        print('t-test result is there is evidence to reject the null hypothesis with a p value of {}'.format(p_value))
    else:
        print('t-test result is that we cannot reject the null hypothesis with p value of {}'.format(p_value))
    return t_stat,p_value

def cohens_d(cur,multi,type):
    if multi==False:

        data=scaling_exponent_linear(cur,type)-scaling_exponent_noise(cur,type)
        d=np.nanmean(data) / np.nanstd(data,ddof=1)
    else:
        data1=scaling_exponent_linear(cur,type)-scaling_exponent_noise(cur,type)
        data2=scaling_exponent_linear(cur,'')-scaling_exponent_noise(cur,'')
        diff=data1-data2
        d=np.nanmean(diff) / np.nanstd(diff,ddof=1)
    print("cohen's d={}".format(d))
    if abs(d)>=0.8:
        print('large effect so mean difference is large compared to the variability in the data')
    elif abs(d)>0.5:
        print('medium sized effect so difference is somewhat greater than variability in the data')
    else:
        print('small effect')
    return d

def violin_plot(data,loc,color,alpha):
    parts=plt.violinplot(data,positions=loc,showmedians=True,showmeans=False,showextrema=True)
    # for pc in parts['bodies']:
    #     pc.set_facecolor(color)
    #     pc.set_edgecolor('black')
    #     pc.set_alpha(alpha)
    return parts

def splitting_into_groups(cur,type):
    se_noise=scaling_exponent_noise(cur,type)
    se_linear=scaling_exponent_linear(cur,type)
    x_axis=Patients(cur)
    avg_hr_all=np.array(cur.execute("SELECT overall_hr_avg FROM Patients").fetchall()).flatten()
    mean=np.nanmean(avg_hr_all)
    IQR=iqr(avg_hr_all,nan_policy='omit')
    
    fig,ax=plt.subplots(1,2,figsize=(14,8))
    plt.subplot(1,2,1)
    #fig.suptitle('{} Surrogate test'.format(type))
    Group1_filter=np.where(avg_hr_all<mean-IQR/2)
    Group1_average=np.nanmean(avg_hr_all[Group1_filter])
    Group1_members=x_axis[Group1_filter]
    Group1_placeholders = ",".join("?" for _ in Group1_members)
    query = f"SELECT ECG_scaling_exponent_linear,ECG_scaling_exponent_noise FROM Patients WHERE Number IN ({Group1_placeholders})" # creates a mask of patients with scaling exponents in bottom quartile and then fetches their data
    results=cur.execute(query, Group1_members).fetchall()
    Group1_ECG_scaling_exponents=np.array(results,dtype=np.float64)

    Group2_filter=np.where((avg_hr_all<mean+IQR/2) & (avg_hr_all>mean-IQR/2))
    Group2_average=np.nanmean(avg_hr_all[Group2_filter])
    Group2_members=x_axis[Group2_filter]
    Group2_placeholders = ",".join("?" for _ in Group2_members)
    query = f"SELECT ECG_scaling_exponent_linear,ECG_scaling_exponent_noise FROM Patients WHERE Number IN ({Group2_placeholders})"# creates a mask of patients with scaling exponents in middle 2 quartiles and then fetches their data
    results=cur.execute(query, Group2_members).fetchall()
    Group2_ECG_scaling_exponents=np.array(results,dtype=np.float64)

    Group3_filter=np.where(avg_hr_all>mean+IQR/2)
    Group3_average=np.nanmean(avg_hr_all[Group3_filter])
    Group3_members=x_axis[Group3_filter]
    Group3_placeholders = ",".join("?" for _ in Group3_members)
    query = f"SELECT ECG_scaling_exponent_linear,ECG_scaling_exponent_noise FROM Patients WHERE Number IN ({Group3_placeholders})"# creates a mask of patients with scaling exponents in uppper quartile and then fetches their data
    results=cur.execute(query, Group3_members).fetchall()
    Group3_ECG_scaling_exponents=np.array(results,dtype=np.float64)
    print(Group3_ECG_scaling_exponents,len(Group3_members))
    plt.scatter(Group1_members,Group1_ECG_scaling_exponents[:,0]-Group1_ECG_scaling_exponents[:,1])
    plt.scatter(Group2_members,Group2_ECG_scaling_exponents[:,0]-Group2_ECG_scaling_exponents[:,1])
    plt.scatter(Group3_members,Group3_ECG_scaling_exponents[:,0]-Group3_ECG_scaling_exponents[:,1])
    plt.tick_params(axis='x',labelrotation=-90,length=0.1)

    plt.subplot(1,2,2)
    
    plt.boxplot(avg_hr_all)
    """better to create groups on average heart rate and then look at scaling exponents in those groups"""
    # fig.suptitle('Trend–Noise DFA Exponent Differences in Smartwatch ECG and Heart Rate Data')
    # plt.subplot(1,2,1)
    # plt.scatter(x_axis[filter],se_noise[filter],marker='d',color='orange',label='noise dominant ECG')
    # plt.scatter(x_axis[filter],se_linear[filter],marker='D',color='blue',label='trend dominant ECG')
    # plt.tick_params(axis='x',labelrotation=-90,length=0.1)
    # plt.plot(x_axis[filter],diff[filter],'p',color='red',label='trend-noise ECG',linestyle='-')
    # plt.axhline(np.nanmean(diff[filter]))
    # plt.xlabel('patient number')
    # plt.ylabel('scaling exponents')
    # plt.title('Scaling Exponents from noise and trend dominated regions')
    # plt.legend()
    # plt.subplot(1,2,2)
    # plt.grid()
    # se_linear= se_linear[~np.isnan(se_linear)]
    # se_noise= se_noise[~np.isnan(se_noise)]
    # violin_diff=se_linear-se_noise
    # violin_filter=np.where((violin_diff<mean+IQR/2) & (violin_diff>mean-IQR/2))
    # v1=violin_plot((se_linear-se_noise)[violin_filter],[1],'#D43F3A',alpha=1)
    plt.savefig('/data/t/smartWatch/patients/completeData/DamianInternshipFiles/Graphs/splittingScalingExponents')

def scaling_exponent_differences(cur,type):

    # finding difference in SE for HR and ECG data
    HR_se_linear=scaling_exponent_linear(cur,'')
    HR_se_noise=scaling_exponent_noise(cur,'')
    HR_se_diff=HR_se_linear-HR_se_noise
    
    ECG_se_linear=scaling_exponent_linear(cur,type)
    ECG_se_noise=scaling_exponent_noise(cur,type)
    ECG_se_diff=ECG_se_linear-ECG_se_noise

    variation=HR_se_diff-ECG_se_diff
    nan_mask=~np.isnan(variation) # mask of patients with nan values showing no ECGs taken
    BA_plot(HR_se_diff[nan_mask],ECG_se_diff[nan_mask])
    variation= variation[nan_mask]
    labels,masks=Gaussian_mixing(variation)
    plotting_se_differences(cur,nan_mask,variation,masks,labels)
    return masks,variation,nan_mask

def plotting_se_differences(cur,nan_mask,variation,masks,labels):
    x_axis=Patients(cur)[nan_mask]
    fig,ax=plt.subplots(1,2,figsize=(14,8))
    fig.suptitle('Variation in Trend–Noise DFA Exponent Differences in Smartwatch ECG and Heart Rate Data')
    plt.subplot(1,2,1)
    plt.title('variation in Trend-noise')
    
    plt.xlabel('Date')
    plt.ylabel('variation in $\Delta$ scaling exponent')
    plt.scatter(x_axis, variation, c=labels, cmap='viridis')
    plt.scatter(x_axis[masks[0]],variation[masks[0]],marker='d',color='orange',label='variation if SE difference')
    plt.axhline(0,color='k')
    plt.axhline(np.nanmean(variation[masks[0]]),color='orange')
    plt.tick_params(axis='x',labelrotation=-90,length=0.1)
    plt.vlines(x_axis[masks[0]],variation[masks[0]],np.nanmean(variation[masks[0]]),color='orange')
    IQR=iqr(variation,nan_policy='omit')
    # plt.axhline(np.nanmedian(variation),color='red')
    # plt.axhline(np.nanmedian(variation)+IQR,color='red')
    # plt.axhline(np.nanmedian(variation)-IQR,color='red')
    plt.legend()
    plt.subplot(1,2,2)
    plt.title('violin plot(s)')
    plt.xticks([])
    plt.xlabel('violin plot(s)')
    plt.ylabel('calcualed values')
    v1=violin_plot(variation[masks[0]],[1],'','')
    plt.legend([v1['bodies'][0]], ['variation'], loc=3)
    plt.savefig('/data/t/smartWatch/patients/completeData/DamianInternshipFiles/Graphs/scalingExponents_variation.png')
    shapiro_testing(variation)

def fetching_week_data(cur,mask,nan_mask):
    sql="SELECT Weeks.* FROM Weeks INNER JOIN Patients ON Weeks.Number = Patients.Number"
    weekly_hr=np.array(cur.execute(sql).fetchall())[nan_mask][mask]
    df_weeks=pd.DataFrame(weekly_hr)
    df_weeks.rename(columns={df_weeks.columns[0]:'Patient_number'},inplace=True) # creates a dataframe containing the week data for the specific group wanted
    return df_weeks

def fetching_patient_data(cur,nan_mask,mask,columns,table):
    if not columns:
        colums=['*']
    columns_str=', '.join(columns)
    sql=f"SELECT {columns_str} FROM {table}"
    if len(mask)!=0:
        Group_data=np.array(cur.execute(sql).fetchall())[nan_mask][mask]
    else:
        Group_data=np.array(cur.execute(sql).fetchall())
    df_patients=pd.DataFrame(Group_data,columns=columns)
    return df_patients


def fetching_DayAndNight_data(cur,patient_num):
    if patient_num!='all':
        DayAndNight_Data=cur.execute("SELECT * FROM DayAndNight WHERE Number='{}' ORDER BY date".format(patient_num)).fetchall()
        
        
    else:
        DayAndNight_Data=cur.execute("SELECT * FROM DayAndNight ORDER BY date".format(patient_num)).fetchall()

    df=pd.DataFrame(DayAndNight_Data)
    df=df.drop(columns=0)
    columns=df.columns
    df.rename(columns={columns[0]:'patient_number',columns[1]:'date',columns[2]:'day_avg_hr',columns[3]:'night_avg_hr',columns[4]:'day_min_hr',columns[5]:'night_min_hr',columns[6]:'day_max_hr',columns[7]:'night_max_hr'},inplace=True)
    start_date=pd.to_datetime(df['date']).min()
    df['days_since_start']=(pd.to_datetime(df['date'])-start_date).dt.total_seconds()/3600/24
    return df


def comparing_scaling_in_patients_and_volunteers():
    patient_con=sqlite3.connect('patient_metrics.db')
    patient_cur=patient_con.cursor()
    patient_data=fetching_patient_data(patient_cur,[],[],['Number','scaling_exponent_noise','scaling_exponent_linear','ECG_scaling_exponent_noise','ECG_scaling_exponent_linear'],'Patients')
    patient_data=patient_data.apply(pd.to_numeric,errors='coerce')
    volunteer_con=sqlite3.connect('volunteer_metrics.db')
    volunteer_cur=volunteer_con.cursor()
    volunteer_data=fetching_patient_data(volunteer_cur,[],[],['Number','scaling_exponent_noise','scaling_exponent_linear','ECG_scaling_exponent_noise','ECG_scaling_exponent_linear'],'Patients')
    volunteer_data=volunteer_data.apply(pd.to_numeric,errors='coerce')
    print(patient_data)
    print(volunteer_data)

def week_analysis(cur,masks,nan_mask,Group_num):
    print(fetching_patient_data(cur,nan_mask,masks[Group_num]))
    df=fetching_week_data(cur,masks[Group_num],nan_mask)
    df=df.melt(id_vars='Patient_number', var_name='week', value_name='Heart_rate')
    df['Heart_rate']=pd.to_numeric(df['Heart_rate'], errors="coerce")
    df.dropna(subset=['week', 'Heart_rate'], inplace=True)
    return group_analysis(df,x_var='week',y_var='Heart_rate',Group_num=Group_num,normalised=False)

def DayVsNight_analysis(cur):
    Patients=fetching_patient_data(cur,[],[])['Patient_number']
    for p in Patients:
        df=fetching_DayAndNight_data(cur,p)
        graphing_day_and_night(cur,df)

def adding_to_table(cur,table_name,values,identifiers,identifier_value,multiple_identifiers=False,multiple_values=False,column_name=None,row_name=None,column=True,row=False):

    location = "column" if column else "row"
    case = (location, multiple_values, multiple_identifiers)
    if case==("column",False,False):
        cur.execute(f"ALTER TABLE '{table_name}' ADD COLUMN'{column_name}' TEXT ")
        cur.execute(f"UPDATE TABLE '{table_name}' SET '{column_name}'=value WHERE '{identifiers}'='{identifier_value}'")
    elif case==("column",False,True):
        pass
    elif case==("column",True,True):
        pass
    elif case==("column",True,False):
        pass
    elif case==("row",False,False):
        pass
    elif case==("row",False,True):
        pass
    elif case==("row",True,True):
        pass
    elif case==("row",True,False):
        pass
    cur.commit()

def graphing_day_and_night(cur,df):
    x_axis=pd.to_datetime(df['date'])
    plt.scatter(x_axis,df['day_avg_hr'])
    coeffs=np.polyfit(df['days_since_start'],df['day_avg_hr'],deg=1,cov=False)
    polyf=np.poly1d(coeffs)
    patient_nums=np.unique(df['patient_number'])
    adding_to_table(cur,'Patients',coeffs[0],'Number',patient_nums[0],column_name='day_avg_hr_gradient')
    df['fitted_day_avg']=polyf(df['days_since_start'])
    plt.plot(x_axis,df['fitted_day_avg'])
    plt.tick_params(axis='x',labelrotation=-90,length=0.1)
    plt.tight_layout()
    
    if len(patient_nums)==1:
        plt.savefig(f'/data/t/smartWatch/patients/completeData/DamianInternshipFiles/Graphs/testing_metrics/{patient_nums[0]}_DayVsNight.png')
        plt.close()

def crossover(cur):
    df_volunteers=fetching_patient_data(cur,[],[],['Number','crossover_PPG','crossover_ECG'],'Patients')
    df_patients=fetching_patient_data(sqlite3.connect('patient_metrics.db').cursor(),[],[],['Number','crossover_PPG','crossover_ECG'],'Patients')
    df_patients=df_patients.apply(pd.to_numeric,errors='coerce')
    df_volunteers['crossover_ECG']=pd.to_numeric(df_volunteers['crossover_ECG'],errors='coerce')
    df_volunteers['crossover_PPG']=pd.to_numeric(df_volunteers['crossover_PPG'],errors='coerce')


    # df_volunteers=np.array(df_volunteers.drop(index=2))

    # t_test(df_patients[:,7],df_volunteers[:,7].astype(np.float64))
    x_axis1=df_patients['Number']
    x_axis2=np.arange(0,len(df_volunteers['Number']),1)
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    ax[0][0].set_xlabel('Patient number')
    ax[0][0].tick_params(axis='x',labelrotation=-90,length=0.1)
    ax[0][0].axhline(np.mean(df_patients['crossover_PPG'].astype(np.float64)),color='blue',label='average PPG patient crossover')
    print(f'Patient PPG p value:{shapiro_testing(df_patients['crossover_PPG'])}')
    print(f'Patient ECG p value:{shapiro_testing(df_patients['crossover_ECG'])}')
    print(f'volunteer PPG p value:{shapiro_testing(df_volunteers['crossover_PPG'])}')
    print(f'volunteer ECG p value:{shapiro_testing(df_volunteers['crossover_ECG'])}')

    ax[0][0].set_ylabel('crossover point')
    ax[0][0].set_title('crossover points on DFA graph for patients')
    ax[0][0].scatter(x_axis1,df_patients['crossover_PPG'].astype(np.float64),color='blue',label='PPG patient crossovers')
    ax[0][0].scatter(x_axis1,df_patients['crossover_ECG'].astype(np.float64),color='orange',label='ECG patient crossovers')
    ax[0][0].axhline(np.mean(df_patients['crossover_ECG'].astype(np.float64)),color='orange',label='average ECG patient crossover')
    

    ax[0][1].scatter(x_axis2,df_volunteers['crossover_PPG'].astype(np.float64),color='blue',label='PPG patient crossovers')
    ax[0][1].scatter(x_axis2,df_volunteers['crossover_ECG'].astype(np.float64),color='orange',label='ECG patient crossovers')
    ax[0][1].set_xticks(x_axis2,df_volunteers['Number'])
    ax[0][1].tick_params(axis='x',labelrotation=-90,length=0.08)
    ax[0][1].set_xlabel('Volunteer Label')
    ax[0][1].set_ylabel('crossover point')
    ax[0][1].set_title('crossover points on DFA graph for Volunteers')




    df_patients.plot(kind='box',x='Number',y=['crossover_PPG','crossover_ECG'],ax=ax[1][0],title='Patient Box Plots')
    df_volunteers.plot(kind='box',x='Number',y=['crossover_PPG','crossover_ECG'],ax=ax[1][1],title='Volunteer Box Plots')


    ax[0][0].legend()
    ax[0][1].legend()
    fig.savefig('/data/t/smartWatch/patients/completeData/DamianInternshipFiles/Graphs/testing_metrics/crossovers.png')


def group_analysis(df,x_var,y_var,Group_num,normalised=True,index='Patient_number'):
    if not normalised:
        df[f'aligned_{x_var}']=df.groupby(index)[x_var].transform(lambda x: x-x.min())
        baseline_y_var = df.loc[df.groupby(index)[x_var].idxmin(), [index, y_var]]
        baseline_y_var=baseline_y_var.rename(columns={y_var: f'Baseline_{y_var}'})
        df=df.merge(baseline_y_var, on=index)
        df[f'normalised_{y_var}']=df[y_var]-df[f'Baseline_{y_var}']
    counts=df[f'aligned_{x_var}'].value_counts()
    sparse=counts[counts<3].index
    df=df[~df[f'aligned_{x_var}'].isin(sparse)]
    model = smf.mixedlm(f"normalised_{y_var} ~ aligned_{x_var}", data=df, groups=df[index])
    result = model.fit()
    df[f'fitted_{y_var}']=result.predict(df)

    return plotting_groups(df,'week','Heart_rate',Group_num)
    

def plotting_groups(dataFrame,x_var,y_var,Group_num,index='Patient_number'):
    
    print(dataFrame)
    plt.figure(figsize=(12, 6))
    plt.title(f'Normalised Plot of {x_var} for group {Group_num}')
    plt.xlabel(f'aligned_{x_var}')
    plt.ylabel(f'normalised_{y_var}')
    for i, group in dataFrame.groupby(index): 
        plt.scatter(group[f'aligned_{x_var}'], group[f'normalised_{y_var}'], alpha=1)
        sorted_df=dataFrame.sort_values(f'aligned_{x_var}')
        plt.plot(group[f'aligned_{x_var}'],group[f'fitted_{y_var}'])
    plt.savefig('/data/t/smartWatch/patients/completeData/DamianInternshipFiles/Graphs/testing_metrics/{}_group{}.png'.format(x_var,Group_num))
    plt.close()


def BA_plot(data1,data2):
    fig,ax=plt.subplots(1)
    plt.title('Bland-Altman Plot of ECG and HR determined $\Delta$ scaling exponent')
    sm.graphics.mean_diff_plot(data1,data2,ax=ax) # creates a mean difference plot
    plt.tight_layout()
    plt.show()
    plt.savefig('/data/t/smartWatch/patients/completeData/DamianInternshipFiles/Graphs/Bland-Altman.png')

def resting_hr(cur):
    data=cur.execute("SELECT night_hr_avg FROM Patients").fetchall()
    print('resting heart rate (at night)=',np.mean(data))

def Gaussian_mixing(data):
    features=pd.DataFrame({
        'difference': data,
    })
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42) # uses gaussian mixing (assuming each group comes from its own distribution) to split the data into subgroups
    gmm.fit(X_scaled)
    features['gmm_cluster'] = gmm.predict(X_scaled)
    Group1_mask=features['gmm_cluster']==0
    Group2_mask=features['gmm_cluster']==1
    Group3_mask=features['gmm_cluster']==2
    Group4_mask=features['gmm_cluster']==3
    Group5_mask=features['gmm_cluster']==4

    return features['gmm_cluster'],[Group1_mask,Group2_mask,Group3_mask,Group4_mask,Group5_mask]
def main():
    type='ECG_'
    con=sqlite3.connect('volunteer_metrics.db') # opens database
    cur=con.cursor()
    #months_hr(cur,con)
    #weeks_hr(con,cur)

    # se=scaling_exponent(cur)
    # histogram(se,'scalingLinear')

    #comparing_scaling_exponents(cur,type)
    # print('T-test with mean of 0 as null')
    # t_test(scaling_exponent_linear(cur,'ECG_')-scaling_exponent_noise(cur,'ECG_'),0)
    # shapiro_testing(scaling_exponent_linear(cur,'ECG_')-scaling_exponent_noise(cur,'ECG_'))
    # print('T-test with HR mean as null')
    # t_test(scaling_exponent_linear(cur,'ECG_')-scaling_exponent_noise(cur,'ECG_'),scaling_exponent_linear(cur,'')-scaling_exponent_noise(cur,''))
    # cohens_d(cur,True,type)
    #Group_masks,variation,nan_mask=scaling_exponent_differences(cur,'ECG_')
    #week_analysis(cur,Group_masks,nan_mask,4)
    #night_analysis(cur)
    #DayVsNight_analysis(cur)
    crossover(cur)
    #resting_hr(cur)
    #comparing_scaling_in_patients_and_volunteers()




if __name__=="__main__":
    main()  

#%%


