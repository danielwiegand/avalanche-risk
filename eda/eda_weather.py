"""Exploratory data analysis on the weather data
"""

import sys
sys.path.append("/home/daniel/Schreibtisch/Projekte/avalanche-risk")

import os
import pickle
import re

import numpy as np
import pandas as pd
import seaborn as sns
from functions_general import import_files
from matplotlib import pyplot as plt

from functions_eda import (plot_correlations, plot_missing_values,
                           show_data_overview, compare_agg_funcs, correlate_aggregate_per_day, compare_shift_correlations, compare_agg_func_correlations)

from model.functions_model import preprocess_X_values

WEATHER_FILEPATH = "../data/lawinenwarndienst/weather_data/1-allgaeu/fellhorn/1/"
FILES = os.listdir(WEATHER_FILEPATH)
METRICS = [re.findall(r"-(\D.*)-", file) for file in FILES]


# * DATA IMPORT ###########################################

# data = import_files(WEATHER_FILEPATH) # Can only import data from before or after 2012 separately, and only from one station. Alternative: Use the pickled data
data = pickle.load(open("../data/lawinenwarndienst/weather_data/pickles/allgaeu.p", "rb"))

warning_levels = pickle.load(open("../data/lawinenwarndienst/warning_levels_preprocessed.p", "rb"))
warning_levels = warning_levels[(warning_levels.low_high == 1) & (warning_levels.Zone == "allgaeu")][["Warnstufe"]].astype(int)


# * MISSING VALUES #########################################

fig = plot_missing_values(data, aspect = 0.00003)
fig.savefig("output/missing_values.png", dpi = 600, facecolor = 'white', transparent = False)


# * TRAIN TEST SPLIT #######################################

# Threshold: August 2017
train = data[data.index < "2017-08-01"]
test = data[data.index >= "2017-08-01"]


# * UNIVARIATE EXPLORATION ##################################

# Graphic for presentation
train_subset = train.loc["20140701":"20150630",["WG", "LD", "LT", "HS"]].reset_index().melt(id_vars = "index")
plot = sns.relplot(data = train_subset, y = "value", x = "index", col = "variable", col_wrap = 2, facet_kws = dict(sharey = False, sharex = False), kind = "line")
plot.fig.subplots_adjust(top=0.9)
plot.fig.autofmt_xdate()
plot.fig.suptitle('Selected variables at yearly resolution', fontsize=22)
plot.savefig('output/overview_year.png', transparent = True)

train_subset = train.loc["20140101":"20140201",["WG", "LD", "LT", "HS"]].reset_index().melt(id_vars = "index")
train_subset["index"] = pd.to_datetime(train_subset["index"])
plot = sns.relplot(data = train_subset, y = "value", x = "index", col = "variable", col_wrap = 2, facet_kws = dict(sharey = False, sharex = False), kind = "line")
plot.fig.subplots_adjust(top=0.9)
plot.fig.autofmt_xdate()
plot.fig.suptitle('Selected variables at monthly resolution', fontsize=22)
plot.savefig('output/overview_month.png', transparent = True)

# GS - Global radiation
show_data_overview(train[["GS"]])

# T0 - Surface temperature
show_data_overview(train[["T0"]])

# WG - wind speed
show_data_overview(train[["WG_05100419"]])
show_data_overview(train[["WG.Boe_05100419"]])

# N - precipitation
show_data_overview(train[["N"]])
show_data_overview(train[["N.MengeParsivel"]])

# LT - air temperature
show_data_overview(train[["Lt"]])

# WR - wind direction
show_data_overview(train[["WR.Boe"]])
show_data_overview(train[["WR"]])


# TS - snow temperature
show_data_overview(train[["TS.000"]])
show_data_overview(train[["TS.020"]])
show_data_overview(train[["TS.040"]])
show_data_overview(train[["TS.060"]])

# LD - air pressure
show_data_overview(train[["LD"]])

# Nsgew - PRECIPITATION WEIGHT
show_data_overview(train[["Nsgew"]])

# HS - SNOW HEIGHT
show_data_overview(train[["HS"]])

# NW - PRECIPITATION BOOLEAN 
show_data_overview(train[["NW"]])

# LT - AIR TEMPERATURE 
show_data_overview(train[["LT"]])

# WS - optical range in meters
show_data_overview(train[["WS-kl"]])
show_data_overview(train[["SW"]])


# * MULTIVARIATE EXPLORATION #####################

train_agg = train.groupby(train.index.date).mean().shift(1)
train_full_agg = train_agg.join(warning_levels).dropna(subset = warning_levels.columns)

# PAIRPLOT ALL VARIABLES
fig = sns.pairplot(train_full_agg)
fig.savefig("output/pairplot_all_vars_shift1.png", dpi = 300, facecolor = 'white', transparent = False)

# MISSING VALUES ALL VARIABLES
fig = plot_missing_values(train_full_agg, labels = train_full_agg.columns, aspect = 0.02)
fig.savefig("output/missing_values_warnstufe.png", dpi = 600, facecolor = 'white', transparent = False, bbox_inches = "tight")

# CORRELATIONS OF INDEPENDENT VARIABLES
out = plot_correlations(train, labels = sorted(METRICS))
out.savefig("output/correlations.png", dpi = 600, facecolor = 'white', transparent = False)

# CORRELATION WITH TARGET VARIABLE
# Compare different shift values
for shift in range(4):
    plt.figure()
    corr_shift = compare_agg_func_correlations(train, warning_levels, shift = shift)
    sns.set(font_scale = 0.4)
    fig = sns.heatmap(corr_shift, cmap = "coolwarm_r", annot = True)
    fig.figure.savefig(f"output/corr_shift{shift}.png", dpi = 600, facecolor = 'white', transparent = False)

# Compare different shifts and aggregation functions
for agg_func in ["mean", "max", "min", "sum"]:
    plt.figure()
    shift_corr = compare_shift_correlations(train, warning_levels, max_shift = 14, agg_func = agg_func)
    sns.set(font_scale = 0.4)
    fig = sns.heatmap(shift_corr, cmap = "coolwarm_r", annot = True)
    fig.figure.savefig(f"output/corr_shift_{agg_func}.png", dpi = 600, transparent = True)

# Compare different aggregation functions

def compare_agg_funcs(df, y):
    """Function to compare correlations with target variable after applying different aggregation functions

    Args:
        df (DataFrame): DataFrame with predictor variables
        y (pd.Series): pd.Series with target variables

    Returns:
        DataFrame: DataFrame with correlations between predictors and target variable
    """
    agg_funcs = ["mean", "max", "min", "sum", "median"]
    compare_agg_func = pd.DataFrame(columns = agg_funcs, index = df.columns)
    for column in df:
        for agg in agg_funcs:
            prep = preprocess_X_values(df[[column]], agg_func = agg)
            prep = prep.join(y).dropna(subset = y.columns)
            compare_agg_func.loc[column, agg] = prep.corr()[y.columns[0]][0]
    sns.heatmap(np.abs(compare_agg_func.fillna(0)), annot = True, cmap = "Greys")
    return compare_agg_func

compare_agg_funcs(train, warning_levels)

# RELATION TO TARGET VARIABLE
# Mean - shift 1
train_agg = train.groupby(train.index.date).mean().shift(1)
train_full_agg = train_agg.join(warning_levels)
fig = sns.pairplot(train_full_agg, y_vars = ['Warnstufe'], x_vars = train_full_agg.columns[::-1])
fig.savefig("output/pairplot_shift1.png", dpi = 600)