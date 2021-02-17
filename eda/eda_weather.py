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
                           show_data_overview, correlate_aggregate_per_day, compare_shift_correlations, compare_agg_func_correlations)

from model.functions_model import preprocess_X_values

# WEATHER_FILEPATH = "../data/lawinenwarndienst/weather_data/1-allgaeu/hochgrat-hoermoos/"
WEATHER_FILEPATH = "../data/lawinenwarndienst/weather_data/1-allgaeu/fellhorn/1/"
FILES = os.listdir(WEATHER_FILEPATH)
METRICS = [re.findall(r"-(\D.*)-", file) for file in FILES]


# * DATA IMPORT ###########################################

# data = import_files(WEATHER_FILEPATH) # Can only import data from before or after 2012 separately, and only from one station. Better use the pickles files.
data = pickle.load(open("../data/lawinenwarndienst/weather_data_before_2012/pickles/werdenfels.p", "rb"))

warning_levels = pickle.load(open("../data/lawinenwarndienst/warning_levels_preprocessed.p", "rb"))
warning_levels = warning_levels[(warning_levels.low_high == 1) & (warning_levels.Zone == "Allgäuer Alpen")][["Warnstufe"]].astype(int)


# * MISSING VALUES #########################################

fig = plot_missing_values(data, aspect = 0.00003)
fig.savefig("output/missing_values.png", dpi = 600, facecolor = 'white', transparent = False)


# * TRAIN TEST SPLIT #######################################

# Threshold: August 2017
train = data[data.index < "2017-08-01"]
test = data[data.index >= "2017-08-01"]


# * UNIVARIATE EXPLORATION ##################################

# GS - Global radiation
show_data_overview(train[["GS"]])

# T0 - Surface temperature
show_data_overview(train[["T0"]])
#? Monatsverlauf: Größere Schwankungen / Abstürze

# WG - wind speed
show_data_overview(train[["WG_05100419"]])
#? Teils viele NAs (bis zu 16%)
#? Viel Rauschen in den Daten (natürlich)

show_data_overview(train[["WG.Boe_05100419"]])
#? Bis zu 17% NAs

# N - precipitation
show_data_overview(train[["N"]])
show_data_overview(train[["N.MengeParsivel"]])
#? N ist wohl neuer Niederschlag pro Zeitfenster

# LT - air temperature
show_data_overview(train[["LT_05100419"]])
show_data_overview(train[["LT_86991032"]])

# WR - wind direction
show_data_overview(train[["WR.Boe_05100419"]])
show_data_overview(train[["WR_05100419"]])
#? Teils viele NAs
#? Gemessen in Grad von 0 bis 359
#? Unwahrscheinlich, dass ich diese Variable brauche
#? Interessant das Histogramm

# TS - snow temperature
show_data_overview(train[["TS.000_86991032"]])
show_data_overview(train[["TS.020_86991032"]])
show_data_overview(train[["TS.040_86991032"]])
show_data_overview(train[["TS.060_86991032"]])

snow_temp = train.loc[:"2013-09-01",train.columns.str.startswith("TS")]
snow_temp["time"] = snow_temp.index.date
snow_temp = snow_temp.melt(id_vars = "time")
sns.lineplot(x = snow_temp["time"], y = snow_temp["value"], hue = snow_temp["variable"])
#? Absolute Werte für Schneetemperatur können nicht stimmen - bis 40 Grad? MW von 5 Grad? Wahrscheinlich: Wenn kein Schnee, dann einfach Umgebungstemperatur!
#? Die höher angebrachten sind auch im Winter manchmal oberhalb des Schnees. Insgesamt aber nur wenige Tage Unterschied

# LD - air pressure
show_data_overview(train[["LD_05100419"]])

# Nsgew - PRECIPITATION WEIGHT
show_data_overview(train[["Nsgew_86991032"]])
#? Nur sehr wenige Werte (der maximal möglichen) vorhanden (nur von 2016 bis 2018)  
#? Daten für 2017 sehen seltsam aus

# LF - RELATIVE HUMIDITY 
show_data_overview(train[["LF_86991032"]]) # wind station
#? Relative Feuchtigkeit in Prozent
#? Viel Rauschen in den Daten
#? Verteilung sehr linksschief

show_data_overview(train[["LF_05100419"]]) # snow station
#? In manchen Jahren bis zu 14% NAs
#? Insgesamt feuchter als in der Windstation
#? Verteilung sehr linksschief; die meisten Werte bei 100%

# HS - SNOW HEIGHT
show_data_overview(train[["HS"]])
#? MW = 30, Median = 0

# NW - PRECIPITATION BOOLEAN 
show_data_overview(train[["NW_05100419"]])
#? Boolean variable
#? In 6% Niederschlag

# LT - AIR TEMPERATURE 
show_data_overview(train[["LT"]])
#? Viel Rauschen

# WS - SNOW PILLOW WASSERSÄULE ???
# The snow pillow measures the water equivalent of the snow pack based on hydrostatic pressure created by overlying snow.
show_data_overview(train[["WS-kl"]])
show_data_overview(train[["SW"]])

# Status - ???
show_data_overview(train[["Status"]])



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
#? Schneehöhe ist negativ korreliert mit Luftdruck, Lufttemperatur und Schneetemperatur
#? Luftdruck ist positiv korreliert mit Temperatur, negativ mit Windgeschwindigkeit und Luftfeuchtigkeit
#? Luftfeuchtigkeit ist negativ korreliert mit Lufttemperatur

# CORRELATION WITH TARGET VARIABLE
# Compare different shift values
for shift in range(4):
    plt.figure()
    corr_shift = compare_agg_func_correlations(train, warning_levels, shift = shift)
    sns.set(font_scale = 0.4)
    fig = sns.heatmap(corr_shift, cmap = "coolwarm_r", annot = True)
    fig.figure.savefig(f"output/corr_shift{shift}.png", dpi = 600, facecolor = 'white', transparent = False)
#? Alle Temperaturen: max /mean ist am stärksten korreliert, min am wenigsten
#? Sum verhält sich sehr ähnlich wie mean

# Compare different aggregation functions
for agg_func in ["mean", "max", "min", "sum"]:
    plt.figure()
    shift_corr = compare_shift_correlations(train, warning_levels, max_shift = 14, agg_func = agg_func)
    sns.set(font_scale = 0.4)
    fig = sns.heatmap(shift_corr, cmap = "coolwarm_r", annot = True)
    fig.figure.savefig(f"output/corr_shift_{agg_func}.png", dpi = 600, facecolor = 'white', transparent = False)
#? Shift von 1 bis 2 hat stärkere Korrelationen als die Daten am selben Tag (logisch)

def compare_agg_funcs(df, y):
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
#? mean erzielt insgesamt die besten Ergebnisse und ist für alle Variablen halbwegs ok

# RELATION TO TARGET VARIABLE
# Mean - shift 1
train_agg = train.groupby(train.index.date).mean().shift(1)
train_full_agg = train_agg.join(warning_levels)
fig = sns.pairplot(train_full_agg, y_vars = ['Warnstufe'], x_vars = train_full_agg.columns[::-1])
fig.savefig("output/pairplot_shift1.png", dpi = 600)
#? Keine ganz klaren Trends erkennbar
