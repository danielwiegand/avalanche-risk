from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
import pandas as pd
from imblearn.over_sampling import SMOTE
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
    # SMOTE (important: AFTER getting the lags)
    if smote:
        sm = SMOTE(random_state = 10, **kwargs)
        X, y = sm.fit_resample(X, y)
    return X, y

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