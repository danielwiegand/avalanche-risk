"""General functions for modeling the avalanche warning level
"""

import re
import pandas as pd
import os

def import_weather_file(folder, file):
    """Import a CSV file with weather data from lawinenwarndienst-bayern.de/

    Args:
        folder (str): Folder where the data is
        file (str): File to import

    Returns:
        DataFrame: DataFrame with imported weather data
    """
    metric = re.findall(r"-(\D.*)-", file)[0]
    station = re.findall(r"\d{8}", file)[0]    
    df = pd.read_csv(folder + file, index_col = 0, skiprows = 15, names = [metric], delimiter = ";", header = None, na_values = ["---"], decimal = ",", parse_dates = False)
    df.index = pd.to_datetime(df.index, format="%d.%m.%Y %H:%M:%S")
    # df["station"] = station
    return df

def import_files(filepath):
    """Function to import several weather files from lawinenwarndienst-bayern.de at once

    Args:
        filepath (str): Path to folder containing files to be imported

    Returns:
        DataFrame: DataFrame with imported data
    """
    files = os.listdir(filepath)
    dfs = [import_weather_file(filepath, file) for file in files]
    data = dfs[0].join(dfs[1:])
    data = data.reindex(sorted(data.columns), axis=1)
    return data