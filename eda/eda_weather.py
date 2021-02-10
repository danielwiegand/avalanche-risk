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
    df = pd.read_csv(folder + file, index_col = False, skiprows = 15, names = ["date", metric], delimiter = ";", header = None, na_values = ["---"], decimal = ",")
    df["date"] = df["date"].replace(r";.+", "", regex = True)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace = True)
    df["station"] = station
    return df

def show_data_overview(df):
    print(f"Shape: {df.shape}\n")
    print(f"Info: {df.info()}\n")
    print(f"Describe: {df.describe()}\n")
    print(f"Head:\n{df.head()}\n")
    print(f"NA values (is NA?):\n\n{df.iloc[:,0].isna().value_counts(normalize = True)*100}\n")
    print("NA values per year (is NA?):\n")
    print(df.iloc[:,0].isna().groupby(df.index.year).value_counts(normalize = True)*100)
    subset = df.loc["2014-01-01":"2015-01-01"]
    subset.iloc[:,0].plot(title = "jahr")
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


#! Neue Auswertung machen: Wieviel der maximal möglichen Werte vorhanden?

#######################################################
# * T0 - SURFACE TEMPERATURE ##########################
#######################################################

t0 = import_file(WEATHER_FILEPATH, FILES[0])

show_data_overview(t0)

#?  Auffällig:
#   Zyklische große Schwankungen im Jahresverlauf (monatlich)


#######################################################
# * WG - WIND SPEED          ##########################
#######################################################

wg = import_file(WEATHER_FILEPATH, FILES[1])

show_data_overview(wg)

#?  Auffällig:
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

#?  Auffällig:
#   Zyklische große Schwankungen im Jahresverlauf (monatlich)


#######################################################
# * WR - WIND DIRECTION      ##########################
#######################################################

wr = import_file(WEATHER_FILEPATH, FILES[4])

show_data_overview(wr)

#?  Auffällig:
#   Gemessen in Grad von 0 bis 359
#   Unwahrscheinlich, dass ich diese Variable brauche
#   Interessant das Histogramm, wo man die bevorzugten Windrichtungen sieht


#######################################################
# * TS - SNOW TEMPERATURE 02 ##########################
#######################################################

ts02 = import_file(WEATHER_FILEPATH, FILES[5])

show_data_overview(ts02)

#?  Auffällig:
#   Auch hier wieder die monatlichen starken Schwankungen - bestes Beispiel!!
#   Absolute Werte für Schneetemperatur können nicht stimmen - bis 40 Grad? MW von 5 Grad?


#######################################################
# * TS - SNOW TEMPERATURE 00 ##########################
#######################################################

ts00 = import_file(WEATHER_FILEPATH, FILES[6])

show_data_overview(ts00)

#?  Auffällig:
#   Auch hier wieder die monatlichen starken Schwankungen
#   Absolute Werte für Schneetemperatur können nicht stimmen - bis 40 Grad? MW von 5 Grad?



#######################################################
# * TS - SNOW TEMPERATURE 06 ##########################
#######################################################

ts06 = import_file(WEATHER_FILEPATH, FILES[7])

show_data_overview(ts06)

#?  Auffällig:
#   Auch hier wieder die monatlichen starken Schwankungen
#   Absolute Werte für Schneetemperatur können nicht stimmen - bis 40 Grad? MW von 5 Grad?


#######################################################
# * LD - AIR PRESSURE        ##########################
#######################################################

ld = import_file(WEATHER_FILEPATH, FILES[8])

show_data_overview(ld)

#?  Auffällig:
#   Monatliche Schwankungen



#######################################################
# * Nsgew - PRECIPITATION WEIGHT ######################
#######################################################

nsgew = import_file(WEATHER_FILEPATH, FILES[9])

show_data_overview(nsgew)

nsgew.loc["2017-01-01":"2017-12-31"].plot()
nsgew.loc["2017-01-05":"2017-01-05"].plot()

#?  Auffällig:
#   Nur sehr wenige Werte (der maximal möglichen) vorhanden (nur von 2016 bis 2018)  
# Monatliche Schwankungen


#######################################################
# * WGboe - Speed gusts          ######################
#######################################################

wgboe = import_file(WEATHER_FILEPATH, FILES[10])

show_data_overview(wgboe)


#?  Auffällig:
# Monatliche Schwankungen


FILES