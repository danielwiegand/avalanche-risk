import sys

sys.path.append("/home/daniel/Schreibtisch/Projekte/avalanche-risk")

import os
import pickle
import re

import numpy as np
import pandas as pd
import seaborn as sns
from eda.functions_eda import plot_correlations, plot_missing_values
from functions_general import import_files, import_weather_file
from matplotlib import pyplot as plt

from functions_model import (fit_model, preprocess, evaluate_regression, evaluate_classification, preprocess_X_values, import_preprocess_region, import_preprocess_multiple_regions)


# * DATA IMPORT ###########################################

# METRICS_PER_REGION = pickle.load(open("../data/lawinenwarndienst/metrics_per_region.p", "rb"))
# REGIONS = ['allgaeu', 'ammergau', 'werdenfels', 'voralpen', 'chiemgau','berchtesgaden']
# INTERSECTING_METRICS = pickle.load(open("../data/lawinenwarndienst/intersecting_metrics.p", "rb")) # load intersecting metrics

# #? Import a single region

# metrics = ["N", "HS", "N.MengeParsivel", "GS", "TS.060", "WS-kl", "T0", "LF", "LT", "WR", "WG", "LD"]

# # ["HS_86991032", "LD_05100419", "LF_05100419", "LT_86991032", "N_86991032", "T0_86991032", "TS.060_86991032", "WG_05100419", "WR_05100419"]

X_train, y_train, X_test, y_test = import_preprocess_region("allgaeu", metrics = metrics, agg_func = "mean", min_shift = 1, max_shift = 2, include_y = True, smote = True)


#? Import several regions

# [metric for metric in METRICS_PER_REGION["allgaeu"] if metric in METRICS_PER_REGION["ammergau"]]

metrics = ["N", "TS.040", "HS", "GS", "T0", "LF", "LT", "WR", "WG"]
regions = ["allgaeu", "ammergau", "werdenfels"]

X_train, y_train, X_test, y_test = import_preprocess_multiple_regions(regions, metrics, agg_func = "mean", min_shift = 1, max_shift = 2, include_y = True, smote = True)

y_train.value_counts()


# * MODELING ################################################

#! logreg parameter c. Like in support vector machines, smaller values specify stronger regularization.

m, y_pred_reg, y_pred, residuals = fit_model("RandomForest", X_train, y_train, max_depth = 4, class_weight = "balanced", n_estimators = 100)

# Für boosting:
# m.fit(X_train, y_train)
# y_pred = m.predict(X_train)


# * EVALUATION ##############################################

# Coefficients
# pd.DataFrame(m.coef_, index = X_train.columns).plot(kind = "bar")
# pd.DataFrame(m.coef_, columns = X_train.columns).plot(kind = "bar")

# evaluate_regression(m, y_train, y_pred_reg, X_train, y_train, scoring = "r2")

# plot_tree(m, feature_names = X_train.columns) # für decisiontree
pd.DataFrame(m.feature_importances_, index = X_train.columns).plot(kind = "bar") # für RF / catboost

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
