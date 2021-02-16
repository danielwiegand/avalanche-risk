import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

df = pickle.load(open("../data/lawinenwarndienst/warning_levels.p", "rb"))

df.index = pd.to_datetime(df.index, dayfirst = True)

# Only retain timesteps with division in six zones
df_zones = df["2018-04-09":"2008-02-03"]

# Only retain the two first digits (warning levels)
df_zones_dangers = df_zones.replace(r".{3}$", "", regex = True)

def separate_zones(df):
    output = pd.DataFrame()
    for i in df:
        zone = df[i].str.split("", expand = True).iloc[:,1:3]
        zone = zone.melt(ignore_index = False, var_name = "low_high", value_name = "danger_level")
        zone["zone"] = i
        # zone["zone"] = i
        output = output.append(zone)
    output["Jahr"] = output.index.year
    return output

df_long = separate_zones(df_zones_dangers)
df_long = df_long[~(df_long["danger_level"] == "0")]
df_long["zone"] = df_long["zone"].replace([1, 2, 3, 4, 5, 6], ["allgaeu", "ammergau", "werdenfels", "voralpen", "chiemgau", "berchtesgaden"])
df_long = df_long.rename(columns = {"zone": "Zone", "danger_level": "Warnstufe"})

df_long["Saison"] = 1
for i in range(2008, 2018, 1):
    df_long.Saison[df_long.index > f"{i}-08-01"] += 1
    
pickle.dump(df_long, open("../data/lawinenwarndienst/warning_levels_preprocessed.p", "wb"))

# * Basic descriptive statistics
df_zones_dangers
df_zones_dangers.shape

df_long
df_long.shape

# * How often are low and high equal?
lower_values = df_zones_dangers.replace(r"^.{1}", "", regex = True).apply(lambda x: x.astype(int))
upper_values = df_zones_dangers.replace(r".{1}$", "", regex = True).apply(lambda x: x.astype(int))

((upper_values - lower_values) == 0).sum() / df_zones_dangers.shape[0] * 100 # equal in between 43% to 60%

# * How are the differences distributed?
differences_danger = (upper_values - lower_values).apply(pd.Series.value_counts, normalize = True).mul(100).round(2).fillna(0)
sns.heatmap(differences_danger, cmap = "binary", annot = True)
# In most cases, both levels are equal or the "upper" value is 1 larger. Seldom, "upper" is lower or 2 steps larger than "lower".

# * "Smoothed" course of levels over the year
rolling_mean = pd.DataFrame(df_long.groupby(["Saison", "low_high"])["Warnstufe"].rolling("10d").mean()).reset_index()
sns.relplot(data = rolling_mean, x = "level_2", y = "Warnstufe", hue = "low_high", kind = "line", row = "Saison", aspect = 2, facet_kws = dict(sharex = False, sharey = True), palette = "Set2")

rolling_mean = pd.DataFrame(df_long.groupby(["Saison", "Zone", "low_high"])["Warnstufe"].rolling("10d").mean()).reset_index()
sns.relplot(data = rolling_mean, x = "level_3", y = "Warnstufe", hue = "low_high", kind = "line", row = "Saison", col = "Zone", aspect = 2, facet_kws = dict(sharex = False, sharey = True), palette = "Set2")

# Barplot danger_level counts depending on zone and year
g = sns.catplot(data = df_long, x = "Warnstufe", kind = "count", hue = "Jahr", col = "Zone", col_wrap = 3, palette = "flare")
g.set_axis_labels("Warnstufe", "Anzahl")
g.savefig("barplot_zone_jahr.png", dpi = 300, pad_inches = 0.8)

# Barplot danger_level counts depending on year and zone
g = sns.catplot(data = df_long, x = "Jahr", kind = "count", hue = "Warnstufe", col = "Zone", col_wrap = 3, palette = "flare")
g.set_axis_labels("Jahr", "Anzahl")
g.savefig("barplot_jahr_zone.png", dpi = 300, pad_inches = 0.8)

# Heatmap danger_level vs zone (counts) over all years
z = df_long.groupby(["Zone"])["Warnstufe"].value_counts().unstack()
sns.heatmap(data = z, cmap = "flare", annot = True, fmt = "d")

# Heatmap danger_level vs. zone depending on year
def facet(data, color):
    data = data.groupby(["Zone"])["Warnstufe"].value_counts().unstack().fillna(0)
    sns.heatmap(data, cbar = False, square = False, annot = True, cmap = "flare", fmt = "0g")
    
g = sns.FacetGrid(df_long, col = "Jahr", col_wrap = 4, size = 3)
g.map_dataframe(facet)
g.set_axis_labels("Warnstufe", "")
g.savefig("heatmap.png", dpi = 300, pad_inches = 0.8)

# Heatmap danger_level vs. zone depending on year
def facet(data, color):
    data = data.groupby(["Zone"])["Warnstufe"].value_counts(normalize = True).unstack().fillna(0)
    sns.heatmap(data, cbar = False, square = False, annot = True, cmap = "flare", fmt = ".0%")
    
g = sns.FacetGrid(df_long, col = "Jahr", col_wrap = 4, size = 3)
g.map_dataframe(facet)
g.set_axis_labels("Warnstufe", "")
g.savefig("heatmap_prozent.png", dpi = 300, pad_inches = 0.8)