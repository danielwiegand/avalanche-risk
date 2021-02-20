"""Functions for the exploratory data analysis
"""

import sys

sys.path.append("/home/daniel/Schreibtisch/Projekte/avalanche-risk")

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

def separate_zones(df):
    """Function to separate upper and lower warning zone per region of the avalanche warning data

    Args:
        df (DataFrame): DataFrame with the avalanche warning data

    Returns:
        DataFrame: DataFrame with separated zones per region
    """
    output = pd.DataFrame()
    for i in df:
        zone = df[i].str.split("", expand = True).iloc[:,1:3]
        zone = zone.melt(ignore_index = False, var_name = "low_high", value_name = "danger_level")
        zone["zone"] = i
        # zone["zone"] = i
        output = output.append(zone)
    output["Jahr"] = output.index.year
    return output

def show_data_overview(df):
    """Print some general information about a single variable

    Args:
        df (DataFrame): DataFrame containing a single column with a numerical variable
    """
    print(f"Shape: {df.shape}\n")
    print(f"Info: {df.info()}\n")
    print(f"Describe: {df.describe()}\n")
    print(f"Head:\n{df.head()}\n")
    print(f"NA values (is NA?):\n\n{df.iloc[:,0].isna().value_counts(normalize = True)*100}\n")
    print("NA values per year (is NA?):\n")
    print(df.iloc[:,0].isna().groupby(df.index.year).value_counts(normalize = True)*100)
    df.iloc[:,0].cumsum().plot(title = "cumulative sum") 
    df.plot(title = "all years")   
    plt.figure()
    subset = df.loc["2014-01-01":"2015-01-01"]
    subset.plot(title = "year")
    plt.figure()
    subset.iloc[:,0].rolling(window = 10, min_periods = 1).mean().plot(title = "year - rolling 10")
    plt.figure()
    subset.iloc[:,0].rolling(window = 100, min_periods = 1).mean().plot(title = "year - rolling 100")
    plt.figure()
    subset.iloc[:,0].rolling(window = 1000, min_periods = 1).mean().plot(title = "year - rolling 1000")
    subset = df.loc["2014-01-01":"2014-02-27"]
    subset.plot(title = "2 months")
    subset = df.loc["2014-01-01":"2014-01-31"]
    subset.plot(title = "1 month")
    plt.figure()
    subset.iloc[:,0].rolling(window = 100, min_periods = 1).mean().plot(title = "month - rolling 100")
    subset = df.loc["2014-01-01":"2014-01-02"]
    subset.plot(title = "2 days")
    plt.figure()
    subset.iloc[:,0].rolling(window = 50, min_periods = 1).mean().plot(title = "2 days - rolling 50")
    print(f"\nValue counts:\n{df.iloc[:,0].value_counts(normalize = True, dropna = False)}\n")
    df.plot(kind = "hist", bins = 20)
    
def plot_missing_values(df, **kwargs): 
    """Plot a heatmap of the missing values in a DataFrame

    Args:
        df (DataFrame): DataFrame to be examined

    Returns:
        seaborn plot: Heatmap generated
    """
    kwargs.setdefault("aspect", 0.004)
    plt.tight_layout()
    fig, ax = plt.subplots()
    im = ax.imshow(df.isna().astype(int).to_numpy(), cmap = "magma", interpolation = "nearest", **kwargs)
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation = 90)
    return fig

def plot_correlations(df):
    """Plots a correlation heatmap of variables in a DataFrame

    Args:
        df (DataFrame): DataFrame with numerical variables

    Returns:
        seaborn plot: Heatmap generated
    """
    labels = df.columns
    sns.set(font_scale=0.4)
    out = sns.heatmap(df.corr(), cmap = "coolwarm", annot = True, xticklabels = labels, yticklabels = labels)
    return out.figure

def correlate_aggregate_per_day(df, warning_levels, agg_func, shift):
    """Aggregate a time series DataFrame to "days" by means of an aggregation function. Shift this data backwards a specified number of days and show correlations with target variable for that time shift.

    Args:
        df (DataFrame): DataFrame containing numerical variables
        warning_levels (DataFrame): DataFrame containing the target variable and having a timeSeries index
        agg_func (str): Function to aggregate time series data to "days"
        shift (int): Number of days to display shifted correlations

    Returns:
        DataFrame: Matrix with the correlations.
    """
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
    cor = df_full_agg.corr().iloc[-1,:].rename(f"day{shift}")
    return cor

def compare_shift_correlations(df, warning_levels, max_shift, agg_func):
    """Compare correlations between target variable and different time shifts of predictors

    Args:
        df (DataFrame): DataFrame containing numerical predictors and a timeSeries index
        warning_levels (pd.Series): The target variable to correlate with
        max_shift (int): Maximum number of days for which a time shift correlation should be displayed
        agg_func (str): Function by which the data should be aggregated to daily level

    Returns:
        DataFrame: DataFrame with correlations to target variable for different time shifted days
    """
    correlations = pd.DataFrame(index = df.columns.tolist())
    for shift in range(max_shift+1):
        correlation = correlate_aggregate_per_day(df, warning_levels, agg_func, shift = shift)
        correlations = correlations.join(correlation)
    return correlations

def compare_agg_func_correlations(df, warning_levels, shift):
    """Compare correlations between target variable and different aggregation functions. Compares "mean", "max", "min" and "sum".

    Args:
        df (DataFrame): DataFrame containing numerical predictors and a timeSeries index
        warning_levels (pd.Series): The target variable to correlate with
        shift (int): Number of days the predictors should be shifted backwards before calculating the correlations with the target variable

    Returns:
        DataFrame: DataFrame with correlations per predictor and aggregation function
    """
    correlations = pd.DataFrame(index = df.columns.tolist())
    for agg_func in ["mean", "max", "min", "sum"]:
        correlation = correlate_aggregate_per_day(df, warning_levels, agg_func, shift)
        correlations = correlations.join(correlation)
    return correlations