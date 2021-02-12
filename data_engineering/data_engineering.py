import sys
sys.path.append("/home/daniel/Schreibtisch/Projekte/avalanche-risk")

import os
import pickle
import re

import numpy as np
import pandas as pd
import seaborn as sns
from functions_general import import_weather_file
from matplotlib import pyplot as plt
from eda.functions_eda import plot_missing_values
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from eda.functions_eda import plot_correlations

WEATHER_FILEPATH = "../data/lawinenwarndienst/weather_data/1-allgaeu/hochgrat-hoermoos/"
FILES = os.listdir(WEATHER_FILEPATH)
METRICS = [re.findall(r"(?<=-).+(?=-)", file)[0] for file in FILES]


# * DATA IMPORT ###########################################

dfs = [import_weather_file(WEATHER_FILEPATH, file) for file in FILES]
data = dfs[0].join(dfs[1:])
data = data.reindex(sorted(data.columns), axis=1)
del dfs

warning_levels = pickle.load(open("../data/lawinenwarndienst/warning_levels_preprocessed.p", "rb"))
warning_levels = warning_levels[(warning_levels.low_high == 1) & (warning_levels.Zone == "Allgäuer Alpen")][["Warnstufe"]].astype(int)


# * TRAIN TEST SPLIT #######################################

data = data.drop(["Nsgew_86991032"], axis = 1)

test = data[["HS_86991032", "LD_05100419", "LF_05100419", "LT_86991032", "N_86991032", "T0_86991032", "TS.060_86991032", "WG_05100419", "WR_05100419"]]

# Threshold: August 2017
train = data[data.index < "2017-08-01"]
test = data[data.index >= "2017-08-01"]


# * IMPUTE MISSING VALUES ##################################

def preprocess(df, agg_func, columns_to_keep):
    # Aggregate
    df = getattr(train.groupby(train.index.date), agg_func)()
    # Keep only relevant columns
    df = df[columns_to_keep]
    # Impute
    knn = KNNImputer(n_neighbors = 10)
    # Scale
    scale = MinMaxScaler()
    pipeline = make_pipeline(knn, scale)
    out = pd.DataFrame(pipeline.fit_transform(df), columns = df.columns, index = df.index)
    return out

def get_shifted_features(df, max_shift, include_zero = False):
    out = pd.DataFrame(index = df.index)
    start = [0 if include_zero else 1][0]
    for shift in range(start, max_shift+1):
        data = preprocess(df).shift(shift)
        data.columns = df.columns + f"-{shift}"
        out = out.join(data)
    return out

# * remove correlated features
plot_correlations(data, labels = data.columns)
columns_to_keep = ["HS_86991032", "LD_05100419", "LF_05100419", "LT_86991032", "N_86991032", "T0_86991032", "TS.060_86991032", "WG_05100419", "WR_05100419"]

train_prep = preprocess(train, agg_func = "mean", columns_to_keep = columns_to_keep)

train_shifted = get_shifted_features(train_prep, max_shift = 1, include_zero = False)

