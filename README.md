# SmartWatch Metrics Pipeline

This pipeline takes in paths to folders containing the data from **Withings Scanwatch** watches, with no preprocessing of the data required. The PPG Heart Rate data is then processed, and the ECG signals are processed using a seperate file. All metrics are then added to a SQL Database to be easily accessed for analysis. Graphs for different metrics can be toggled on or off.
