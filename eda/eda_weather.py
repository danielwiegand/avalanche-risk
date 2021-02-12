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

from functions_eda import (plot_correlations, plot_missing_values,
                           show_data_overview, correlate_aggregate_per_day, compare_shift_correlations)

WEATHER_FILEPATH = "../data/lawinenwarndienst/weather_data/1-allgaeu/hochgrat-hoermoos/"
FILES = os.listdir(WEATHER_FILEPATH)
METRICS = [re.findall(r"(?<=-).+(?=-)", file)[0] for file in FILES]


# DATA IMPORT ############
dfs = [import_weather_file(WEATHER_FILEPATH, file) for file in FILES]
data = dfs[0].join(dfs[1:])
data = data.reindex(sorted(data.columns), axis=1)
del dfs

warning_levels = pickle.load(open("../data/lawinenwarndienst/warning_levels_preprocessed.p", "rb"))
warning_levels = warning_levels[(warning_levels.low_high == 1) & (warning_levels.Zone == "Allgäuer Alpen")][["Warnstufe"]].astype(int)


# MISSING VALUES #############
fig = plot_missing_values(data, labels = sorted(METRICS), aspect = 0.00003)
fig.savefig("output/missing_values.png", dpi = 600)


# TRAIN TEST SPLIT ##############
# Threshold: August 2017
train = data[data.index < "2017-08-01"]
test = data[data.index >= "2017-08-01"]


# UNIVARIATE EXPLORATION ###########

# T0 - Surface temperature
show_data_overview(train[["T0_86991032"]])
#?  Auffällig:
#   Monatsverlauf: Größere Schwankungen / Abstürze

# WG - wind speed
show_data_overview(train[["WG_05100419"]])
#?  Auffällig:
#   Teils viele NAs (bis zu 16%)
#   Viel Rauschen in den Daten (natürlich)

show_data_overview(train[["WG.Boe_05100419"]])
#?  Auffällig:
# Bis zu 17% NAs

# N - precipitation
show_data_overview(train[["N_86991032"]])
#?  Auffällig:
#   N ist wohl neuer Niederschlag pro Zeitfenster

# LT - air temperature
show_data_overview(train[["LT_05100419"]])
show_data_overview(train[["LT_86991032"]])

# WR - wind direction
show_data_overview(train[["WR.Boe_05100419"]])
show_data_overview(train[["WR_05100419"]])
#?  Auffällig:
#   Teils viele NAs
#   Gemessen in Grad von 0 bis 359
#   Unwahrscheinlich, dass ich diese Variable brauche
#   Interessant das Histogramm

# TS - snow temperature
show_data_overview(train[["TS.000_86991032"]])
show_data_overview(train[["TS.020_86991032"]])
show_data_overview(train[["TS.040_86991032"]])
show_data_overview(train[["TS.060_86991032"]])

snow_temp = train.loc[:"2013-09-01",train.columns.str.startswith("TS")]
snow_temp["time"] = snow_temp.index.date
snow_temp = snow_temp.melt(id_vars = "time")
sns.lineplot(x = snow_temp["time"], y = snow_temp["value"], hue = snow_temp["variable"])

#?  Auffällig:
#   Absolute Werte für Schneetemperatur können nicht stimmen - bis 40 Grad? MW von 5 Grad? Wahrscheinlich: Wenn kein Schnee, dann einfach Umgebungstemperatur!
#   Die höher angebrachten sind auch im Winter manchmal oberhalb des Schnees. Insgesamt aber nur wenige Tage Unterschied

# LD - air pressure
show_data_overview(train[["LD_05100419"]])

# Nsgew - PRECIPITATION WEIGHT
show_data_overview(train[["Nsgew_86991032"]])
#?  Auffällig:
#   Nur sehr wenige Werte (der maximal möglichen) vorhanden (nur von 2016 bis 2018)  
# Daten für 2017 sehen seltsam aus

# LF - RELATIVE HUMIDITY 
show_data_overview(train[["LF_86991032"]]) # wind station
#?  Auffällig:
#   Relative Feuchtigkeit in Prozent
#   Viel Rauschen in den Daten
#   Verteilung sehr linksschief

show_data_overview(train[["LF_05100419"]]) # snow station
#?  Auffällig:
#   In manchen Jahren bis zu 14% NAs
#   Insgesamt feuchter als in der Windstation
#   Verteilung sehr linksschief; die meisten Werte bei 100%

# HS - SNOW HEIGHT
show_data_overview(train[["HS_86991032"]])
#?  Auffällig:
#   MW = 30, Median = 0

# NW - PRECIPITATION BOOLEAN 
show_data_overview(train[["NW_05100419"]])
#?  Auffällig:
#   Boolean variable
#   In 6% Niederschlag

# LT - AIR TEMPERATURE 
show_data_overview(train[["LT_05100419"]]) # snow station
show_data_overview(train[["LT_86991032"]]) # snow station
#?  Auffällig:
#   Viel Rauschen


# MULTIVARIATE EXPLORATION

# CORRELATIONS
out = plot_correlations(train, labels = sorted(METRICS))
out.savefig("output/correlations.png", dpi = 600)
#? Auffällig:
#   Schneehöhe ist negativ korreliert mit Luftdruck, Lufttemperatur und Schneetemperatur
#   Luftdruck ist positiv korreliert mit Temperatur, negativ mit Windgeschwindigkeit und Luftfeuchtigkeit
#   Luftfeuchtigkeit ist negativ korreliert mit Lufttemperatur

# CORRELATION WITH TARGET VARIABLE

# Variation 1: On 10 minutes levels
train_full = train.join(warning_levels)
# Forward fill "warnstufe" per day
train_full["Warnstufe"] = train_full[["Warnstufe"]].groupby(train_full.index.date).fillna(method = "ffill")
# Plot missing values
labels_plot = sorted(METRICS)
labels_plot.append("Warnstufe")
fig = plot_missing_values(train_full, labels = labels_plot, aspect = 0.00003)
fig.savefig("output/missing_values_warnstufe.png", dpi = 600)

# Variation 2: On daily levels
# Comparison: Mean, max and min
correlation_mean = correlate_aggregate_per_day(train, warning_levels, "mean", shift = 0)
correlation_max = correlate_aggregate_per_day(train, warning_levels, "max", shift = 0)
correlation_min = correlate_aggregate_per_day(train, warning_levels, "min", shift = 0)
correlation_sum = correlate_aggregate_per_day(train, warning_levels, "sum", shift = 0)

fig = sns.heatmap(pd.concat([correlation_mean, correlation_max, correlation_min, correlation_sum], axis = 1), cmap = "coolwarm_r", annot = True)
fig.figure.savefig("output/corr_shift0.png", dpi = 600)

#? Erkenntnisse:
#   Alle Temperaturen: max /mean ist am stärksten korreliert, min am wenigsten
#   Sum verhält sich sehr ähnlich wie mean


# # LAG 1
# correlation_mean = correlate_aggregate_per_day(train, warning_levels, "mean", shift = 1)
# correlation_max = correlate_aggregate_per_day(train, warning_levels, "max", shift = 1)
# correlation_min = correlate_aggregate_per_day(train, warning_levels, "min", shift = 1)
# correlation_sum = correlate_aggregate_per_day(train, warning_levels, "sum", shift = 1)

# fig = sns.heatmap(pd.concat([correlation_mean, correlation_max, correlation_min, correlation_sum], axis = 1), cmap = "coolwarm_r", annot = True)
# fig.figure.savefig("output/corr_shift1.png", dpi = 600)


# Compare lag correlations: mean
shift_corr_mean = compare_shift_correlations(train, warning_levels, max_shift = 14, agg_func = "mean")
sns.set(font_scale=0.4)
fig = sns.heatmap(shift_corr_mean, cmap = "coolwarm_r", annot = True)
fig.figure.savefig("output/corr_shift_mean.png", dpi = 600)

# Compare lag correlations: max
shift_corr_mean = compare_shift_correlations(train, warning_levels, max_shift = 14, agg_func = "max")
sns.set(font_scale=0.4)
fig = sns.heatmap(shift_corr_mean, cmap = "coolwarm_r", annot = True)
fig.figure.savefig("output/corr_shift_max.png", dpi = 600)

# Compare lag correlations: min
shift_corr_mean = compare_shift_correlations(train, warning_levels, max_shift = 14, agg_func = "min")
sns.set(font_scale=0.4)
fig = sns.heatmap(shift_corr_mean, cmap = "coolwarm_r", annot = True)
fig.figure.savefig("output/corr_shift_min.png", dpi = 600)

# Compare lag correlations: sum
shift_corr_mean = compare_shift_correlations(train, warning_levels, max_shift = 14, agg_func = "sum")
sns.set(font_scale=0.4)
fig = sns.heatmap(shift_corr_mean, cmap = "coolwarm_r", annot = True)
fig.figure.savefig("output/corr_shift_sum.png", dpi = 600)