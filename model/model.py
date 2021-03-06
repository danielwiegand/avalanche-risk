"""Import and preprocess predictors and target variables, run a model, evaluate it and predict on a test set."""

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

from functions_model import (fit_model, evaluate_classification, import_preprocess_region, import_preprocess_multiple_regions, conduct_rfe)

# * DATA IMPORT ###########################################

METRICS_PER_REGION = pickle.load(open("../data/lawinenwarndienst/metrics_per_region.p", "rb"))
# REGIONS = ['allgaeu', 'ammergau', 'werdenfels', 'voralpen', 'chiemgau','berchtesgaden']
INTERSECTING_METRICS = pickle.load(open("../data/lawinenwarndienst/intersecting_metrics.p", "rb")) # load intersecting metrics

#? Import a single region

# all_metrics_allgaeu = [metric for metric in list(set(METRICS_PER_REGION["allgaeu"])) if metric not in ["LS.berechnet", "LS.Rohdaten", "Nsgew", "LS.unten.Rohdaten", "LS.unten.berechnet"]]

metrics_rfe = ["N.MengeParsivel", "N", "SW", "GS", "LT", "HS", "TS.100", "WG.Boe", "TS.080", "LF"]

X_train, y_train, X_test, y_test = import_preprocess_region("allgaeu", metrics = metrics_rfe, agg_func = "mean", min_shift = 1, max_shift = 2, include_y = True, smote = True)

#? Import several regions

# set([metric for metric in METRICS_PER_REGION["allgaeu"] if metric in METRICS_PER_REGION["ammergau"]])
# metrics = ["N", "GS", "LT", "HS", "TS.100", "WG.Boe", "TS.080", "LF", "WG", "TS.000", "WR"]
# regions = ["allgaeu", "ammergau"]

# X_train, y_train, X_test, y_test = import_preprocess_multiple_regions(regions, metrics, agg_func = "mean", min_shift = 1, max_shift = 3, include_y = True, smote = True)


# * RFE #####################################################
 
# rfe = conduct_rfe(X_train, y_train, top_n = 20)


# * MODELING ################################################

m, y_pred, residuals = fit_model("RandomForest", X_train, y_train, max_depth = 4) 


# * EVALUATION ##############################################

# Coefficients
pd.DataFrame(m.coef_, columns = X_train.columns).plot(kind = "bar")

pd.DataFrame(m.feature_importances_, index = X_train.columns).plot(kind = "bar") # für RF / catboost

evaluate_classification(m, y_train, y_pred, X_train, scoring = "accuracy")


# * PREDICTION ##############################################

y_pred_test = m.predict(X_test)

evaluate_classification(m, y_test, y_pred_test, X_test, scoring = "accuracy")



# ? ANNEX ##################################################


# * VOTING CLASSIFIER

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

m1 = LogisticRegression(penalty = "elasticnet", class_weight = None, solver = "saga", random_state = 10, verbose = 10, l1_ratio = 1)

m2 = RandomForestClassifier(n_estimators = 100, random_state = 10, n_jobs = -1, max_depth = 4)

m3 = SVC(C = 0.5, class_weight = None, random_state = 10)

m = VotingClassifier(estimators = [('lr', m1), 
                                   ('rf', m2), 
                                   ('svm', m3)], 
                     voting = 'hard')

m.fit(X_train, y_train)

y_pred = m.predict(X_train)
m.score(X_train, y_train)

evaluate_classification(m, y_train, y_pred, X_train, scoring = "accuracy")


# * DEEP LEARNING #########################################

y_train_label_encoded = y_train - 1

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_squared_error, r2_score)

input_shape = (X_train.shape[1],)

m = Sequential([
    Dense(units = 5, activation = 'elu', input_shape = input_shape),
    Dropout(rate = 0.2),
    BatchNormalization(),
    Dense(units = 10, activation = 'elu'),
    Dense(units = 4, activation = 'softmax')
])

m.summary()

earlystopping = EarlyStopping(monitor = 'val_loss', patience = 20)

opt = Adam(learning_rate = 0.005)
m.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = m.fit(X_train, y_train_label_encoded, batch_size = 50, epochs = 500, validation_split = 0.2, callbacks = [earlystopping])

plt.plot(history.history['loss'], label='training_loss')
plt.plot(history.history['val_loss'], label='validation_loss')
plt.legend()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

y_pred = pd.DataFrame(m.predict_classes(X_train)) + 1

print(classification_report(y_train, y_pred))
ax = sns.heatmap(confusion_matrix(y_pred, y_train), cmap = "Greys", annot = True, fmt = "d", xticklabels = "1234", yticklabels = "1234")
ax.set(xlabel = "ypred", ylabel = "ytrue")



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
