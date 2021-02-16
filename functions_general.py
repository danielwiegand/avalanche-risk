import re
import pandas as pd
import os

def import_weather_file(folder, file):
    metric = re.findall(r"-(\D.*)-", file)[0]
    station = re.findall(r"\d{8}", file)[0]    
    df = pd.read_csv(folder + file, index_col = 0, skiprows = 15, names = [metric], delimiter = ";", header = None, na_values = ["---"], decimal = ",", parse_dates = False)
    df.index = pd.to_datetime(df.index, format="%d.%m.%Y %H:%M:%S")
    # df["station"] = station
    return df

def import_files(filepath):
    files = os.listdir(filepath)
    dfs = [import_weather_file(filepath, file) for file in files]
    data = dfs[0].join(dfs[1:])
    data = data.reindex(sorted(data.columns), axis=1)
    return data