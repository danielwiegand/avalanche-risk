## DATA ENGINEERING ###


# Imputation
#! davor: train-test-split!
from sklearn.impute import KNNImputer

data_full_agg.isna().sum()
data_full_agg["Nsgew_86991032"].plot()


imputer = KNNImputer(n_neighbors = 10)

data_full_agg_imputed = pd.DataFrame(imputer.fit_transform(data_full_agg), columns = data_full_agg.columns, index = data_full_agg.index)
data_full_agg_imputed.isna().sum()
data_full_agg_imputed["Nsgew_86991032"].plot()
