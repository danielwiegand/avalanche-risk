import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

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
    
def plot_missing_values(df, **kwargs): 
    kwargs.setdefault("aspect", 0.004)
    plt.tight_layout()
    fig, ax = plt.subplots()
    im = ax.imshow(df.isna().astype(int).to_numpy(), cmap = "magma", interpolation = "nearest", **kwargs)
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation = 90)
    return fig

def plot_correlations(df, labels):
    sns.set(font_scale=0.4)
    out = sns.heatmap(df.corr(), cmap = "coolwarm", annot = True, xticklabels = labels, yticklabels = labels)
    return out.figure

def correlate_aggregate_per_day(df, warning_levels, agg_func, shift):
    if agg_func == "mean":
        df_agg = df.groupby(df.index.date).mean().shift(shift)
    elif agg_func == "max":
        df_agg = df.groupby(df.index.date).max().shift(shift)
    elif agg_func == "min":
        df_agg = df.groupby(df.index.date).min().shift(shift)
    elif agg_func == "sum":
        df_agg = df.groupby(df.index.date).sum().shift(shift)
    df_full_agg = df_agg.join(warning_levels)
    df_full_agg = df_full_agg.dropna(subset = warning_levels.columns)
    cor = df_full_agg.corr().iloc[-1,:].rename(f"{agg_func}_{shift}")
    return cor

def compare_shift_correlations(df, warning_levels, max_shift, agg_func):
    correlations = pd.DataFrame(index = df.columns.tolist())
    for shift in range(max_shift+1):
        correlation = correlate_aggregate_per_day(df, warning_levels, agg_func, shift = shift)
        correlations = correlations.join(correlation)
    return correlations

def compare_agg_func_correlations(df, warning_levels, shift):
    correlations = pd.DataFrame(index = df.columns.tolist())
    for agg_func in ["mean", "max", "min", "sum"]:
        correlation = correlate_aggregate_per_day(df, warning_levels, agg_func, shift)
        correlations = correlations.join(correlation)
    return correlations