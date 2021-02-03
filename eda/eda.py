import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pickle.load(open("../data/lawinenwarndienst/warning_levels.p", "rb"))

df.index = pd.to_datetime(df.index, dayfirst = True)

df_zones = df["2018-04-09":"2008-02-03"]

df_zones_dangers = df_zones.replace(r".{3}$", "", regex = True)




def separate_zones(df):
    output = pd.DataFrame()
    for i in df:
        zone = df[i].str.split("", expand = True).iloc[:,1:3]
        zone = zone.melt(ignore_index = False, var_name = "low_high", value_name = "danger_level")
        zone["zone"] = i
        # zone["zone"] = i
        output = output.append(zone)
    output["year"] = output.index.year
    return output

asd = separate_zones(df_zones_dangers)

sns.catplot(data = asd, x = "danger_level", kind = "count", hue = "year", col = "zone", col_wrap = 2, palette = "flare")



df_zones_dangers_1 = df_zones_dangers[1].str.split("", expand = True).iloc[:,1:3]

df_zones_dangers_1 = df_zones_dangers_1.apply(lambda x: x.astype(int))

asd = df_zones_dangers_1.melt(value_vars = [1, 2], ignore_index = False)
asd["year"] = asd.index.year

import seaborn as sns
sns.catplot(data = asd, x = "value", kind = "count", col = "year", col_wrap = 4)

sns.set_palette("mako")
sns.catplot(data = asd, x = "value", kind = "count", hue = "year", palette = "flare")
