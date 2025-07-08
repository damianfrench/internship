import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib as Path

def sortingActivityData(number):
    activities=(pd.read_csv('/data/t/smartWatch/patients/completeData/patData/record{}/activities.csv'.format(number),header=0))
    activitiesData = activities[['from','to','from (manual)','to (manual)','Timezone','Activity type','Data','GPS','Modified']].to_numpy(dtype='str')
    starts=activitiesData[:,0]
    ends=activitiesData[:,1]
    print(starts,ends)



if __name__=="__main__":
    sortingActivityData('04')