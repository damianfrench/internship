# SmartWatch Metrics Pipeline

This pipeline takes in paths to folders containing the data from **Withings Scanwatch** watches, with no preprocessing of the data required. The PPG Heart Rate data is then processed, and the ECG signals are processed using a seperate file. All metrics are then added to a SQL Database to be easily accessed for analysis. Graphs for different metrics can be toggled on or off.

## Setting Up

Download and unzip the user data from the smartwatches, either via the app or the website. 
Update ***patient_data_path*** with the path to the folder and ***volunteer_data_path*** if relevent.
Update ***sys.path*** statement at beginning of code to the location of the ***patient_analysis3*** python script for ECG analysis.
Update ***saving_path*** with the location you wish to save the plots generated to.

### Required Packages

numpy - https://pypi.org/project/numpy/
matplotlib.pyplot - https://pypi.org/project/matplotlib/
pandas - https://pypi.org/project/pandas/
pathlib - https://pypi.org/project/pathlib/
scipy - https://pypi.org/project/scipy/
datetime - https://docs.python.org/3/library/datetime.html
sqlite3 - https://docs.python.org/3/library/sqlite3.html
sys - https://docs.python.org/3/library/sys.html
traceback - https://docs.python.org/3/library/traceback.html
collections - https://docs.python.org/3/library/collections.html
statsmodels.api - https://pypi.org/project/statsmodels/
pyhrv.nonlinear - https://pypi.org/project/pyhrv/

