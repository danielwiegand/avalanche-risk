import sys

sys.path.append("/home/daniel/Schreibtisch/Projekte/avalanche-risk")

import os
import pickle
import re

import numpy as np
import pandas as pd
from functions_general import import_files, import_weather_file


WEATHER_FILEPATH = "../data/lawinenwarndienst/weather_data/"

METRICS_PER_STATION = pickle.load(open("../data/lawinenwarndienst/metrics_per_station.p", "rb"))
METRICS_PER_REGION = pickle.load(open("../data/lawinenwarndienst/metrics_per_region.p", "rb"))
FILEPATHS = pickle.load(open("../data/lawinenwarndienst/filepaths_stations.p", "rb"))
FILEPATHS_PER_REGION = pickle.load(open("../data/lawinenwarndienst/filepaths_regions.p", "rb"))


# Get Intersecting variables of all regions
sets = [set(a) for a in METRICS_PER_REGION.values()]
intersecting_metrics = list(sets[0].intersection(*sets))


# Import all (non-overlapping) metrics per region and pickle them
# In case of overlapping metrics, only one instance of this metric is taken
for region in FILEPATHS_PER_REGION.keys():
    region_df = pd.DataFrame(index = pd.date_range(start = "09-01-2012", end = "07-01-2018", freq = "10min")) # format: month-day-year!
    region_df["region"] = region
    for station in FILEPATHS_PER_REGION[region]:
        print(f"Importing {station}...")
        data = import_files(station)
        columns_to_append = [column for column in data.columns if column not in region_df.columns]
        region_df = region_df.join(data[columns_to_append])
    pickle.dump(region_df, open(f"../data/lawinenwarndienst/weather_data/pickles/{region}.p", "wb"))
    
