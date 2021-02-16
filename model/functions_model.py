from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import (ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from catboost import CatBoostClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_squared_error, r2_score)
from sklearn.model_selection import cross_val_score
import seaborn as sns
import numpy as np
import pickle

WEATHER_FILEPATH = "../data/lawinenwarndienst/weather_data/"

def preprocess_X_values(df, agg_func):
    # Aggregate
    df = getattr(df.groupby(df.index.date), agg_func)()
    # Impute
    knn = KNNImputer(n_neighbors = 10)
    # Scale
    scale = StandardScaler()
    pipeline = make_pipeline(knn, scale)
    out = pd.DataFrame(pipeline.fit_transform(df), columns = df.columns, index = df.index)
    return out

def get_shifted_features(df, min_shift, max_shift):
    out = pd.DataFrame(index = df.index)
    for shift in range(min_shift, max_shift+1):
        data = df.shift(shift)
        data.columns = df.columns + f"-{shift}"
        out = out.join(data).dropna()
    return out

def preprocess(X_train, y_train, agg_func, min_shift, max_shift, include_y, smote, **kwargs):
    X_train_prep = preprocess_X_values(X_train, agg_func = agg_func)
    if include_y:
        X_train_prep = X_train_prep.join(y_train)
    X_train_shifted = get_shifted_features(X_train_prep, min_shift = min_shift, max_shift = max_shift)
    # Merge to align date index
    train = X_train_shifted.join(y_train).dropna(subset = y_train.columns) 
    X = train.iloc[:,:-1]
    y = train.iloc[:,-1]
    # # SMOTE (important: AFTER getting the lags)
    if smote:
        kwargs.setdefault("k_neighbors", 5)
        assert all(y.value_counts() > kwargs["k_neighbors"]), "SMOTE will fail, because some levels of y have less occurences than 'k_neighbors', which is set to 5 as a standard. Specify a lower 'k_neighbors' via **kwargs or set smote to False (and execute it with a larger dataset)."
        sm = SMOTE(random_state = 10, **kwargs)
        X, y = sm.fit_resample(X, y)
    return X, y


def upsample(X, y, **kwargs):
    kwargs.setdefault("k_neighbors", 5)
    assert all(y.value_counts() > kwargs["k_neighbors"]), "SMOTE will fail, because some levels of y have less occurences than 'k_neighbors', which is set to 5 as a standard. Specify a lower 'k_neighbors' via **kwargs or set smote to False (and execute it with a larger dataset)."
    sm = SMOTENC(random_state = 10, **kwargs)
    X, y = sm.fit_resample(X, y)
    return X, y
    

def import_preprocess_region(region, features, agg_func, min_shift, max_shift, include_y, smote):
    
    data = pickle.load(open(WEATHER_FILEPATH + f"pickles/{region}.p", "rb"))
    
    data = data[features]
    
    warning_levels = pickle.load(open("../data/lawinenwarndienst/warning_levels_preprocessed.p", "rb"))
    warning_levels = warning_levels[(warning_levels.low_high == 1) & (warning_levels.Zone == region)][["Warnstufe"]].astype(int)
    
    time_threshold = "2017-08-01"

    X_train_raw = data.loc[data.index < time_threshold]
    X_test_raw = data.loc[data.index >= time_threshold]

    y_train_raw = warning_levels.loc[warning_levels.index < time_threshold]
    y_test_raw = warning_levels.loc[warning_levels.index >= time_threshold]
    
    # Test for NA-only columns in X_test
    na_list = X_test_raw.isna().sum() == len(X_test_raw)
    assert sum(na_list) == 0, f"X_test contains a feature with only NAs: {na_list.index[list(na_list).index(True)]}"
    
    X_train, y_train = preprocess(X_train_raw, y_train_raw, agg_func = agg_func, min_shift = min_shift, max_shift = max_shift, include_y = include_y, smote = smote)
    
    X_test, y_test = preprocess(X_test_raw, y_test_raw, agg_func = agg_func, min_shift = min_shift, max_shift = max_shift, include_y = include_y, smote = False) # no smote for test data
    
    return X_train, y_train, X_test, y_test

def import_preprocess_all_regions(features, agg_func, min_shift, max_shift, include_y, smote):
    
    regions = ['allgaeu', 'ammergau', 'werdenfels', 'voralpen', 'chiemgau','berchtesgaden']

    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = pd.Series()
    y_test = pd.Series()

    for region in regions:
        X_train_reg, y_train_reg, X_test_reg, y_test_reg = import_preprocess_region(region, features = features, agg_func = agg_func, min_shift = min_shift, max_shift = max_shift, include_y = include_y, smote = False)
        X_train_reg["region"] = region
        X_test_reg["region"] = region
        X_train = X_train.append(X_train_reg)
        X_test = X_train.append(X_test_reg)
        y_train = y_train.append(y_train_reg)
        y_test = y_test.append(y_test_reg)
        
    X_train["region"] = X_train.region.astype("category")

    if smote:
        X_train, y_train = upsample(X_train, y_train, categorical_features = [X_train.columns.get_loc("region")])
    
    return X_train, y_train, X_test, y_test

def fit_model(model, X_train, y_train, **kwargs):
    
    if model == "LogisticRegression":
        m = LogisticRegression(penalty = "elasticnet", class_weight = None, solver = "saga", random_state = 10, verbose = 10, l1_ratio = 1, **kwargs)
    elif model == "DecisionTree":
        m = DecisionTreeClassifier(max_depth = 3, random_state = 10, class_weight = None, **kwargs)
    elif model == "RandomForest":
        kwargs.setdefault("max_depth", 4)
        kwargs.setdefault("min_samples_split", 2)
        kwargs.setdefault("min_samples_leaf", 1)
        m = RandomForestClassifier(
            n_estimators = 100,
            oob_score = True,
            n_jobs = -1,
            class_weight = None,
            verbose = 1,
            random_state = 10,
            **kwargs)
    elif model == "LinearSVC":
        m = LinearSVC(penalty = "l2", C = 1, class_weight = None, random_state = 10, **kwargs)
    elif model == "SVC":
        m = SVC(C = 0.5, class_weight = None, random_state = 10, **kwargs)
    # elif model == "Boost":
    #     m = CatBoostClassifier(iterations = 1000,
    #         learning_rate = 0.03, # default: 0.3
    #         depth = 2, # default: 6
    #         l2_leaf_reg = 5, # default: 3
    #         model_size_reg = 5,
    #         random_seed = 10,
    #         class_weights = None,
    #         silent = True,
    #         **kwargs)
        
    m.fit(X_train, y_train)
    y_pred_reg = m.predict(X_train)
    y_pred_clas = m.predict(X_train).round()
    residuals = y_pred_reg - y_train
    return m, y_pred_reg, y_pred_clas, residuals

def evaluate_regression(m, y_true, y_pred, X_train, y_train, **kwargs):
    print(f"MSE: {round(mean_squared_error(y_pred, y_true), 3)}\n")
    print(f"R2: {round(r2_score(y_true, y_pred), 3)}\n")
    print(f"CrossVal: {np.round(cross_val_score(estimator = m, X = X_train, y = y_train, cv = 5, **kwargs), 3)}\n")
    print(f"CrossVal mean: {round(np.mean(cross_val_score(estimator = m, X = X_train, y = y_train, cv = 5, **kwargs)), 3)}")
    
def evaluate_classification(m, y_true, y_pred, X_train, y_train, **kwargs):
    print(classification_report(y_true, y_pred))
    ax = sns.heatmap(confusion_matrix(y_pred, y_true), cmap = "Greys", annot = True, fmt = "d", xticklabels = "1234", yticklabels = "1234")
    ax.set(xlabel = "ytrue", ylabel = "ypred")
    cv = np.round(cross_val_score(estimator = m, X = X_train, y = y_train, cv = 5, verbose = 0, **kwargs), 3)
    print(f"CrossVal: {cv}\n")
    print(f"CrossVal mean: {np.mean(cv)}")