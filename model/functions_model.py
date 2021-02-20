"""Functions for modeling the avalanche risk levels
"""

import sys

sys.path.append("/home/daniel/Schreibtisch/Projekte/avalanche-risk")

import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import re
from eda.functions_eda import plot_correlations, plot_missing_values
from imblearn.over_sampling import SMOTE, SMOTENC
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import KNNImputer
from sklearn.linear_model import (ElasticNet, Lasso, LinearRegression,
                                  LogisticRegression, Ridge)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_squared_error, r2_score)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (MinMaxScaler, PowerTransformer,
                                   StandardScaler)
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

WEATHER_FILEPATH = "../data/lawinenwarndienst/weather_data/"

def preprocess_X_values(df, agg_func):
    """Preprocess features for modeling

    Args:
        df (DataFrame): DataFrame with timeSeries index and numerical features
        agg_func (str): Function to aggregate time series data on a daily level

    Returns:
        DataFrame: Preprocessed features
    """
    # Aggregate with agg_func per day
    df = getattr(df.groupby(df.index.date), agg_func)()
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
    # Impute
    knn = KNNImputer(n_neighbors = 10)
    # Scale
    scale = StandardScaler()
    pipeline = make_pipeline(knn, scale)
    out = pd.DataFrame(pipeline.fit_transform(df), columns = df.columns, index = df.index)
    return out

def get_shifted_features(df, min_shift, max_shift):
    """Get a time series DataFrame which is shifted between min_shift and max_shift days backwards

    Args:
        df (DataFrame): DataFrame with timeSeries index and numerical features
        min_shift (int): Minimum number of days to shift
        max_shift (int): Maximum number of days to shift

    Returns:
        DataFrame: DataFrame with shifted features
    """
    out = pd.DataFrame(index = df.index)
    for shift in range(min_shift, max_shift+1):
        data = df.shift(shift)
        data.columns = df.columns + f"-{shift}"
        out = out.join(data).dropna()
    return out

def preprocess(X_train, y_train, agg_func, min_shift, max_shift, include_y, smote, **kwargs):
    """Wrapper function to do all feature preprocessing (aggregate, impute, scale, shift, drop NAs, smote)

    Args:
        X_train (DataFrame): DataFrame with timeSeries index and numerical features
        y_train (pd.Series): pd.Series with timeSeries index containing the target variable
        agg_func (str): Function to aggregate time series data on a daily level
        min_shift (int): Minimum number of days to shift
        max_shift (int): Maximum number of days to shift
        include_y (bool): Should the time shifted dependent variable be added as a predictor to the model?
        smote (bool): Should SMOTE be performed on the model data? This can be useful in case of an imbalanced dataset.

    Returns:
        X: DataFrame with preprocessed features
        y: pd.Series with target variable in the same date range
    """
    # Impute, Scale
    X_train_prep = preprocess_X_values(X_train, agg_func = agg_func)
    if include_y:
        X_train_prep = X_train_prep.join(y_train)
    # Shift
    X_train_shifted = get_shifted_features(X_train_prep, min_shift = min_shift, max_shift = max_shift) # this function also removes NAs

    # Merge to align date index
    train = X_train_shifted.join(y_train).dropna(subset = y_train.columns) 
    X = train.iloc[:,:-1]
    y = train.iloc[:,-1]
    
    # SMOTE (important: AFTER getting the lags)
    if smote:
        kwargs.setdefault("k_neighbors", 5)
        assert all(y.value_counts() > kwargs["k_neighbors"]), "SMOTE will fail, because some levels of y have less occurences than 'k_neighbors', which is set to 5 as a standard. Specify a lower 'k_neighbors' via **kwargs or set smote to False (and execute it with a larger dataset)."
        sm = SMOTE(random_state = 10, **kwargs)
        X, y = sm.fit_resample(X, y)
    return X, y

def upsample(X, y, **kwargs):
    """Function to upsample a dataset by means of SMOTENC. Useful for imbalanced data.

    Args:
        X (DataFrame): DataFrame containing the predicting variables
        y (pd.Series): pd.Series containing the target variable

    Returns:
        X: DataFrame containing the upsampled predicting variables
        y: pd.Series containing the upsampled target variable
    """
    kwargs.setdefault("k_neighbors", 5)
    assert all(y.value_counts() > kwargs["k_neighbors"]), "SMOTE will fail, because some levels of y have less occurences than 'k_neighbors', which is set to 5 as a standard. Specify a lower 'k_neighbors' via **kwargs or set smote to False (and execute it with a larger dataset)."
    sm = SMOTENC(random_state = 10, **kwargs)
    X, y = sm.fit_resample(X, y)
    return X, y

def import_preprocess_region(region, metrics, agg_func, min_shift, max_shift, include_y, smote):
    """Wrapper function to import and preprocess data from a single avalanche warning region

    Args:
        region (str): The region to import and preprocess. One of ["allgaeu", "ammergau", "werdenfels", "voralpen", "chiemgau", "berchtesgaden"]
        metrics (list): List of strings for the predictor variables which should be retained
        agg_func (str): Function to aggregate time series data on a daily level
        min_shift (int): Minimum number of days to shift
        max_shift (int): Maximum number of days to shift
        include_y (bool): Should the time shifted dependent variable be added as a predictor to the model?
        smote (bool): Should SMOTE be performed on the model data? This can be useful in case of an imbalanced dataset.

    Returns:
        X_train: DataFrame containing preprocessed predictor variables for training set
        y_train: pd.Series containing the target variable for training
        X_test: DataFrame containing preprocessed predictor variables for test set
        y_test: pd.Series containing the target variable for testing
    """
    
    data = pickle.load(open(WEATHER_FILEPATH + f"pickles/{region}.p", "rb"))
    
    # Filter the relevant metrics
    data = data[metrics]
    
    plot = plot_missing_values(data, aspect = 0.00004)
    
    warning_levels = pickle.load(open("../data/lawinenwarndienst/warning_levels_preprocessed.p", "rb"))
    warning_levels = warning_levels[(warning_levels.low_high == 1) & (warning_levels.Zone == region)][["Warnstufe"]].astype(int)
    
    # Train-test split based on time
    time_threshold = "2017-08-01"

    X_train_raw = data.loc[data.index < time_threshold]
    X_test_raw = data.loc[data.index >= time_threshold]

    y_train_raw = warning_levels.loc[warning_levels.index < time_threshold]
    y_train_raw = y_train_raw[~y_train_raw.index.duplicated()] # drop duplicates
    
    y_test_raw = warning_levels.loc[warning_levels.index >= time_threshold]
    y_test_raw = y_test_raw[~y_test_raw.index.duplicated()] # drop duplicates
    
    # Test for NA-only columns in X_test
    na_list = X_test_raw.isna().sum() == len(X_test_raw)
    assert sum(na_list) == 0, f"X_test contains a feature with only NAs: {na_list.index[list(na_list).index(True)]}"
    
    # Preprocessing
    X_train, y_train = preprocess(X_train_raw, y_train_raw, agg_func = agg_func, min_shift = min_shift, max_shift = max_shift, include_y = include_y, smote = smote)
    
    X_test, y_test = preprocess(X_test_raw, y_test_raw, agg_func = agg_func, min_shift = min_shift, max_shift = max_shift, include_y = include_y, smote = False) # no smote for test data
    
    return X_train, y_train, X_test, y_test

def import_preprocess_multiple_regions(regions, metrics, agg_func, min_shift, max_shift, include_y, smote):
    """[summary]

    Args:
        regions (list): List of strings for the regions to be imported and preprocessed. One or several of ["allgaeu", "ammergau", "werdenfels", "voralpen", "chiemgau", "berchtesgaden"]
        metrics (list): List of strings for the predictor variables which should be retained
        agg_func (str): Function to aggregate time series data on a daily level
        min_shift (int): Minimum number of days to shift
        max_shift (int): Maximum number of days to shift
        include_y (bool): Should the time shifted dependent variable be added as a predictor to the model?
        smote (bool): Should SMOTE be performed on the model data? This can be useful in case of an imbalanced dataset.

    Returns:
        X_train: DataFrame containing preprocessed predictor variables for training set
        y_train: pd.Series containing the target variable for training
        X_test: DataFrame containing preprocessed predictor variables for test set
        y_test: pd.Series containing the target variable for testing
    """

    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = pd.Series()
    y_test = pd.Series()

    for region in regions:
        # Import and preprocess region
        X_train_reg, y_train_reg, X_test_reg, y_test_reg = import_preprocess_region(region, metrics = metrics, agg_func = agg_func, min_shift = min_shift, max_shift = max_shift, include_y = include_y, smote = False)
        # Add "region" variable to model fixed effects
        X_train_reg["region"] = region
        X_test_reg["region"] = region
        X_train = X_train.append(X_train_reg)
        X_test = X_test.append(X_test_reg)
        y_train = y_train.append(y_train_reg)
        y_test = y_test.append(y_test_reg)
    
    # Reset index (otherwise, there will be problems during merging due to duplicates in the index)
    X_train.reset_index(drop = True, inplace = True)
    X_test.reset_index(drop = True, inplace = True)    
    y_train.reset_index(drop = True, inplace = True)  
    y_test.reset_index(drop = True, inplace = True)
    
    X_train["region"] = X_train.region.astype("category")
    X_test["region"] = X_test.region.astype("category")
    
    print(f"y_train value counts before SMOTE:\n{y_train.value_counts()}")

    if smote:
        X_train, y_train = upsample(X_train, y_train, categorical_features = [X_train.columns.get_loc("region")])
        
    # One-hot encoding of the "region" variable
    if len(regions) > 1:
        dummies = pd.get_dummies(X_train["region"], drop_first = True)
        X_train.drop("region", axis = 1, inplace = True)
        X_train = X_train.join(dummies)
        
        dummies = pd.get_dummies(X_test["region"], drop_first = True)
        X_test.drop("region", axis = 1, inplace = True)
        X_test = X_test.join(dummies)
        
    else:
        X_train.drop("region", axis = 1, inplace = True)
        X_test.drop("region", axis = 1, inplace = True)
    
    return X_train, y_train, X_test, y_test


def fit_model(model, X_train, y_train, **kwargs):
    """Fit a model on X and y data

    Args:
        model (str): Model to use for fitting. One of ["LogisticRegression", "DecisionTree", "RandomForest", "LinearSVC", "NaiveBayes", "SVC"]. Specify **kwargs to provide further options for the model.
        X_train (DataFrame): DataFrame containing the predictors
        y_train (pd.Series): pd.Series containing the target variable

    Returns:
        m: The fit model
        y_pred (pd.Series): The class predictions
        residuals (pd.Series): The residuals per sample
    """
    
    if model == "LogisticRegression":
        kwargs.setdefault("l1_ratio", 1)
        m = LogisticRegression(penalty = "elasticnet", class_weight = None, solver = "saga", random_state = 10, verbose = 10, **kwargs)
    elif model == "DecisionTree":
        m = DecisionTreeClassifier(max_depth = 3, random_state = 10, class_weight = None, **kwargs)
    elif model == "RandomForest":
        kwargs.setdefault("max_depth", 4)
        kwargs.setdefault("min_samples_split", 2)
        kwargs.setdefault("min_samples_leaf", 1)
        m = RandomForestClassifier(
            n_jobs = -1,
            verbose = 1,
            random_state = 10,
            **kwargs)
    elif model == "LinearSVC":
        m = LinearSVC(penalty = "l2", class_weight = None, random_state = 10, **kwargs)
    elif model == "NaiveBayes":
        kwargs.setdefault("var_smoothing", 1e-9)
        m = GaussianNB(**kwargs)
    elif model == "SVC":
        m = SVC(random_state = 10, **kwargs)
        
    m.fit(X_train, y_train)
    # y_pred_reg = m.predict(X_train) # for regression
    y_pred_clas = m.predict(X_train).round()
    residuals = y_pred_reg - y_train
    return m, y_pred_clas, residuals

def evaluate_regression(m, y_true, y_pred, X_train, **kwargs):
    """Display evaluation metrics for a regression model. Specify **kwargs for further options to the cross-validation score calculation.

    Args:
        m (sklearn model): Fit model
        y_true (pd.Series): The true values
        y_pred (pd.Series): The predicted values
        X_train (pd.DataFrame): The predictors used
    """
    print(f"MSE: {round(mean_squared_error(y_pred, y_true), 3)}\n")
    print(f"R2: {round(r2_score(y_true, y_pred), 3)}\n")
    print(f"CrossVal: {np.round(cross_val_score(estimator = m, X = X_train, y = y_true, cv = 5, **kwargs), 3)}\n")
    print(f"CrossVal mean: {round(np.mean(cross_val_score(estimator = m, X = X_train, y = y_true, cv = 5, **kwargs)), 3)}")
    
def evaluate_classification(m, y_true, y_pred, X_train, **kwargs):
    """Display evaluation metrics for a classification model. Specify **kwargs for further options to the cross-validation score calculation.

    Args:
        m (sklearn model): Fit model
        y_true (pd.Series): The true values
        y_pred (pd.Series): The predicted values
        X_train (pd.DataFrame): The predictors used
    """
    print(classification_report(y_true = y_true, y_pred = y_pred))
    ax = sns.heatmap(confusion_matrix(y_true = y_true, y_pred = y_pred), cmap = "Greys", annot = True, fmt = "d", xticklabels = "1234", yticklabels = "1234")
    ax.set(xlabel = "ypred", ylabel = "ytrue")
    cv = np.round(cross_val_score(estimator = m, X = X_train, y = y_true, cv = 5, verbose = 0, **kwargs), 3)
    print(f"CrossVal: {cv}\n")
    print(f"CrossVal mean: {np.mean(cv)}")

def conduct_rfe(X, y, top_n):
    """Conduct a recursive feature elimination by means of RandomForest and LogisticRegression. Gives back a ranked list of the top_n predictors.

    Args:
        X (DataFrame): Predictors to use
        y (pd.Series): Target variable
        top_n (int): Number of most relevant predictors to return

    Returns:
        pd.Series: top_n predictors for this model
    """
    rf = RandomForestClassifier(max_depth = 4, n_estimators = 100, random_state = 10)
    logreg = LogisticRegression(penalty = "elasticnet", class_weight = None, solver = "saga", random_state = 10, verbose = 10, l1_ratio = 1)
    
    used_metrics = X.columns
    result = pd.DataFrame()
    result["metric"] = used_metrics
    
    for model in [rf, logreg]:
        model_name = re.findall(r"[a-zA-Z]+", str(model))[0]
        rfe = RFE(estimator = model, n_features_to_select = 10, verbose = 1)
        rfe = rfe.fit(X, y)
        result[f"ranking_{model_name}"] = rfe.ranking_
    
    # Average results of the models
    result["average"] = result.iloc[:,1:3].mean(axis = 1)
    result = result.sort_values(by = "average")
    
    plt.figure(figsize = (18,6))
    sns.lineplot(data = result.melt(id_vars = "metric"), x = "metric", y = "value", hue = "variable")
    plt.xticks(rotation = 90)
    plt.ylabel("Rank")
    plt.ylabel("Metric")
    plt.tight_layout()
    plt.savefig("../eda/output/rfe.png")
    
    output = result.iloc[:top_n,0].tolist()
    
    return output
