import sys

sys.path.append("/home/daniel/Schreibtisch/Projekte/avalanche-risk")

import os
import pickle
import re

import numpy as np
import pandas as pd
import seaborn as sns
from eda.functions_eda import plot_correlations, plot_missing_values
from functions_general import import_weather_file
from matplotlib import pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_squared_error, r2_score)
from sklearn.model_selection import cross_val_score
from sklearn.tree import plot_tree

from functions_model import (fit_model, get_shifted_features, preprocess, preprocess_X_values, evaluate_regression, evaluate_classification)


WEATHER_FILEPATH = "../data/lawinenwarndienst/weather_data/1-allgaeu/hochgrat-hoermoos/"
FILES = os.listdir(WEATHER_FILEPATH)
METRICS = [re.findall(r"(?<=-).+(?=-)", file)[0] for file in FILES]


# * DATA IMPORT ###########################################

dfs = [import_weather_file(WEATHER_FILEPATH, file) for file in FILES]
data = dfs[0].join(dfs[1:])
data = data.reindex(sorted(data.columns), axis=1)
del dfs

warning_levels = pickle.load(open("../data/lawinenwarndienst/warning_levels_preprocessed.p", "rb"))
warning_levels = warning_levels[(warning_levels.low_high == 1) & (warning_levels.Zone == "Allgäuer Alpen")][["Warnstufe"]].astype(int)


# * TRAIN TEST SPLIT #######################################

# plot_correlations(data, labels = data.columns) # see correlated features

col_to_retain = ["HS_86991032", "LD_05100419", "LF_05100419", "LT_86991032", "N_86991032", "T0_86991032", "TS.060_86991032", "WG_05100419", "WR_05100419"]

time_threshold = "2017-08-01"

# Threshold: August 2017
X_train_raw = data.loc[data.index < time_threshold, col_to_retain]
X_test_raw = data.loc[data.index >= time_threshold, col_to_retain]

y_train_raw = warning_levels.loc[warning_levels.index < time_threshold]
y_test_raw = warning_levels.loc[warning_levels.index >= time_threshold]


# * DATA PREPROCESSING #####################################

X_train, y_train = preprocess(X_train_raw, y_train_raw, agg_func = "mean", min_shift = 1, max_shift = 1, include_y = True, smote = True)

# * MODELING ################################################


#! logreg parameter c. Like in support vector machines, smaller values specify stronger regularization.

m, y_pred_reg, y_pred, residuals = fit_model("RandomForest", X_train, y_train, max_depth = 4)

# Für boosting:
# m.fit(X_train, y_train)
# y_pred = m.predict(X_train)


# * EVALUATION ##############################################

# Coefficients
# pd.DataFrame(m.coef_, index = X_train.columns).plot(kind = "bar")
# pd.DataFrame(m.coef_, columns = X_train.columns).plot(kind = "bar")

# evaluate_regression(m, y_train, y_pred_reg, X_train, y_train, scoring = "r2")

# plot_tree(m, feature_names = X_train.columns) # für decisiontree
# m.feature_importances_ # für catboost

evaluate_classification(m, y_train, y_pred, X_train, y_train, scoring = "accuracy")

# Precision: Percentage of positive classified observations that are positive
# Recall: Percentage of positive observations correctly classified as positive






# * CHECK ASSUMPTIONS OF LINEAR REGRESSION ###################


# Autocorrelation of residuals

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(residuals)
plot_pacf(residuals)


# Zero conditional mean: plot residuals against features
test = X_train.join(residuals).rename(columns = {"Warnstufe": "residuals"})

test.corr()["residuals"].drop("residuals").plot(kind = "bar")


# Residuals are normally distributed

residuals.hist()

import matplotlib.pyplot as plt
from scipy.stats import probplot

probplot(residuals, plot=plt)


# Homoscedasticity

test = X_train.join(residuals).rename(columns = {"Warnstufe": "residuals"}).melt(id_vars = "residuals")

sns.relplot(data = test, x = "value", y = "residuals", col = "variable", col_wrap = 2)


# Multicollinearity

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

test = add_constant(X_train)
pd.Series([variance_inflation_factor(test.values, i)
           for i in range(test.shape[1])],
           index = test.columns)
