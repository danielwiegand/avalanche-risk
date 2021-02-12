import re
import pandas as pd

def import_weather_file(folder, file):
    metric = re.findall(r"(?<=-).+(?=-)", file)[0]
    station = re.findall(r"\d{8}", file)[0]    
    df = pd.read_csv(folder + file, index_col = 0, skiprows = 15, names = [f"{metric}_{station}"], delimiter = ";", header = None, na_values = ["---"], decimal = ",", parse_dates = False)
    df.index = pd.to_datetime(df.index, format="%d.%m.%Y %H:%M:%S")
    # df["station"] = station
    return df