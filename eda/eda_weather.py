import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import re


WEATHER_FILEPATH = "../data/lawinenwarndienst/weather_data/1-allgaeu/hochgrat-hoermoos/"

def import_file(folder, file):
    metric = re.findall(r"(?<=-).+(?=-)", file)[0]
    station = re.findall(r"\d{8}", file)[0]    
    df = pd.read_csv(folder + file, index_col = 0, skiprows = 15, names = [f"{metric}_{station}"], delimiter = ";", header = None, na_values = ["---"], decimal = ",", parse_dates = False)
    df.index = pd.to_datetime(df.index, format="%d.%m.%Y %H:%M:%S")
    # df["station"] = station
    return df

def show_data_overview(df):
    print(f"Shape: {df.shape}\n")
    print(f"Info: {df.info()}\n")
    print(f"Describe: {df.describe()}\n")
    print(f"Head:\n{df.head()}\n")
    print(f"NA values (is NA?):\n\n{df.iloc[:,0].isna().value_counts(normalize = True)*100}\n")
    print("NA values per year (is NA?):\n")
    print(df.iloc[:,0].isna().groupby(df.index.year).value_counts(normalize = True)*100)
    df.iloc[:,0].cumsum().plot(title = "cumulative sum") 
    df.plot(title = "alle jahre")   
    plt.figure()
    subset = df.loc["2014-01-01":"2015-01-01"]
    subset.plot(title = "jahr")
    plt.figure()
    subset.iloc[:,0].rolling(window = 10, min_periods = 1).mean().plot(title = "jahr - rolling 10")
    plt.figure()
    subset.iloc[:,0].rolling(window = 100, min_periods = 1).mean().plot(title = "jahr - rolling 100")
    plt.figure()
    subset.iloc[:,0].rolling(window = 1000, min_periods = 1).mean().plot(title = "jahr - rolling 1000")
    subset = df.loc["2014-01-01":"2014-02-27"]
    subset.plot(title = "2 monate")
    subset = df.loc["2014-01-01":"2014-01-31"]
    subset.plot(title = "monat")
    plt.figure()
    subset.iloc[:,0].rolling(window = 100, min_periods = 1).mean().plot(title = "monat - rolling 100")
    subset = df.loc["2014-01-01":"2014-01-02"]
    subset.plot(title = "2 tage")
    plt.figure()
    subset.iloc[:,0].rolling(window = 50, min_periods = 1).mean().plot(title = "2 tage - rolling 50")
    print(f"\nValue counts:\n{df.iloc[:,0].value_counts(normalize = True, dropna = False)}\n")
    df.plot(kind = "hist", bins = 20)
    
FILES = os.listdir(WEATHER_FILEPATH)
METRICS = [re.findall(r"(?<=-).+(?=-)", file)[0] for file in FILES]


#######################################################
# * T0 - SURFACE TEMPERATURE ##########################
#######################################################

t0 = import_file(WEATHER_FILEPATH, FILES[0])

show_data_overview(t0)

#?  Auffällig:
#   Monatsverlauf: Größere Schwankungen / Abstürze


#######################################################
# * WG - WIND SPEED          ##########################
#######################################################

wg = import_file(WEATHER_FILEPATH, FILES[1])

show_data_overview(wg)

#?  Auffällig:
#   Teils viele NAs (bis zu 16%)
#   Viel Rauschen in den Daten (natürlich)



#######################################################
# * N - PRECIPITATION        ##########################
#######################################################

n = import_file(WEATHER_FILEPATH, FILES[2])

show_data_overview(n)

n.loc["2014-01-05":"2014-01-05"].plot()

#?  Auffällig:
#   Was ist N genau? Neuer Niederschlag pro 10 Minuten?



#######################################################
# * LT - AIR TEMPERATURE     ##########################
#######################################################

lt = import_file(WEATHER_FILEPATH, FILES[3])

show_data_overview(lt)



#######################################################
# * WR - WIND DIRECTION      ##########################
#######################################################

wr = import_file(WEATHER_FILEPATH, FILES[4])

show_data_overview(wr)

#?  Auffällig:
#   Teils viele NAs
#   Gemessen in Grad von 0 bis 359
#   Unwahrscheinlich, dass ich diese Variable brauche
#   Interessant das Histogramm, wo man die bevorzugten Windrichtungen sieht


#######################################################
# * TS - SNOW TEMPERATURE 02 ##########################
#######################################################

ts02 = import_file(WEATHER_FILEPATH, FILES[5])

show_data_overview(ts02)

#?  Auffällig:
#   Absolute Werte für Schneetemperatur können nicht stimmen - bis 40 Grad? MW von 5 Grad? Wahrscheinlich: Wenn kein Schnee, dann einfach Umgebungstemperatur!


#######################################################
# * TS - SNOW TEMPERATURE 00 ##########################
#######################################################

ts00 = import_file(WEATHER_FILEPATH, FILES[6])

show_data_overview(ts00)

#?  Auffällig:
#   Absolute Werte für Schneetemperatur können nicht stimmen - bis 40 Grad? MW von 5 Grad? Wahrscheinlich: Wenn kein Schnee, dann einfach Umgebungstemperatur!



#######################################################
# * TS - SNOW TEMPERATURE 06 ##########################
#######################################################

ts06 = import_file(WEATHER_FILEPATH, FILES[7])

show_data_overview(ts06)

#?  Auffällig:
#   Absolute Werte für Schneetemperatur können nicht stimmen - bis 40 Grad? MW von 5 Grad? Wahrscheinlich: Wenn kein Schnee, dann einfach Umgebungstemperatur!
#   Das scheint so hoch angebracht zu sein, dass es auch im Winter manchmal oberhalb des Schnees ist!


#######################################################
# * LD - AIR PRESSURE        ##########################
#######################################################

ld = import_file(WEATHER_FILEPATH, FILES[8])

show_data_overview(ld)



#######################################################
# * Nsgew - PRECIPITATION WEIGHT ######################
#######################################################

nsgew = import_file(WEATHER_FILEPATH, FILES[9])

show_data_overview(nsgew)

nsgew.loc["2017-01-01":"2017-12-31"].plot()
nsgew.loc["2017-01-05":"2017-01-05"].plot()

#?  Auffällig:
#   Nur sehr wenige Werte (der maximal möglichen) vorhanden (nur von 2016 bis 2018)  
# Daten für 2017 sehen seltsam aus


#######################################################
# * WGboe - Speed gusts          ######################
#######################################################

wgboe = import_file(WEATHER_FILEPATH, FILES[10])

show_data_overview(wgboe)

#?  Auffällig:
# Bis zu 17% NAs


#######################################################
# * TS - SNOW TEMPERATURE 04 ##########################
#######################################################

ts04 = import_file(WEATHER_FILEPATH, FILES[11])

show_data_overview(ts04)

#?  Auffällig:
#   Absolute Werte für Schneetemperatur können nicht stimmen - bis 40 Grad? MW von 5 Grad? Wahrscheinlich: Wenn kein Schnee, dann einfach Umgebungstemperatur!



#######################################################
# * LF - RELATIVE HUMIDITY ############################
#######################################################

lf_w = import_file(WEATHER_FILEPATH, FILES[12])

show_data_overview(lf_w)

#?  Auffällig:
#   Relative Feuchtigkeit in Prozent
#   Viel Rauschen in den Daten
#   Verteilung sehr linksschief


#######################################################
# * LF - RELATIVE HUMIDITY - SNOW STATION #############
#######################################################

lf_s = import_file(WEATHER_FILEPATH, FILES[13])

show_data_overview(lf_s)

#?  Auffällig:
#   In manchen Jahren bis zu 14% NAs
#   Insgesamt feuchter als in der Windstation
#   Verteilung sehr linksschief; die meisten Werte bei 100%


#######################################################
# * HS - SNOW HEIGHT ##################################
#######################################################

hs = import_file(WEATHER_FILEPATH, FILES[14])

show_data_overview(hs)

#?  Auffällig:
#   MW = 30, Median = 0


#######################################################
# * NW - PRECIPITATION BOOLEAN ########################
#######################################################

nw = import_file(WEATHER_FILEPATH, FILES[15])

show_data_overview(nw)

#?  Auffällig:
#   Boolean variable
#   In 6% Niederschlag



#######################################################
# * WRboe - WIND DIRECTION GUSTS ######################
#######################################################

wrboe = import_file(WEATHER_FILEPATH, FILES[16])

show_data_overview(wrboe)



#######################################################
# * LT - AIR TEMPERATURE ##############################
#######################################################

lt = import_file(WEATHER_FILEPATH, FILES[17])

show_data_overview(lt)

#?  Auffällig:
#   Viel Rauschen


#*## MULTIVARIATE ANALYSIS ###

dfs = [import_file(WEATHER_FILEPATH, file) for file in FILES]
data = dfs[0].join(dfs[1:])
data = data.reindex(sorted(data.columns), axis=1)
del dfs

data.shape

# MISSING VALUES

fig, ax = plt.subplots()
im = ax.imshow(data.isna().astype(int).to_numpy(), aspect = 0.00003, cmap = "magma", interpolation = "nearest")
ax.set_xticks(np.arange(data.shape[1]))
ax.set_xticklabels(sorted(METRICS), rotation = 90)
fig.savefig("missing_values.png", dpi = 600)

# CORRELATIONS
sns.set(font_scale=0.4)
sns.heatmap(data.corr(), cmap = "coolwarm", annot = True, xticklabels = sorted(METRICS), yticklabels = sorted(METRICS))
plt.savefig("correlations.png", dpi = 600)

# crosstab, pairplot


# Beziehung zur Target Variable