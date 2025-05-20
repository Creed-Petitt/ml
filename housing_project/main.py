import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import joblib


area_ix = 1
garage_ix = 27
yr_sold_ix = 36 
yr_built_ix = 7
full_bath_ix = 19
half_bath_ix = 20
bsmt_full_ix = 17
bsmt_half_ix = 18

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        total_area = X[:, area_ix] + X[:, garage_ix] 
        house_age = X[:, yr_sold_ix] - X[:, yr_built_ix] 
        total_bathrooms = (X[:, full_bath_ix]) + (X[:, half_bath_ix]  * 0.5) + (X[:, bsmt_full_ix]) + (X[:,bsmt_half_ix] * 0.5)

        return np.c_[X, total_area, house_age, total_bathrooms]


def main():
    df = pd.read_csv("datasets/ames.csv")
    df = df.drop(columns=["PID", "Alley", "Mas.Vnr.Type", "Pool.QC", "Fence", "Misc.Feature", "Fireplace.Qu"])
    df = df[df["price"] <= 500000]
   
    df["area_cat"] = pd.cut(df["area"],
                                bins=[0, 1000, 2000, np.inf],
                                labels= [1, 2, 3])

    split = StratifiedShuffleSplit(n_splits= 1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df["area_cat"]):
        train_set = df.iloc[train_index]
        test_set = df.iloc[test_index]


    train_set = train_set.drop(columns=["area_cat"])
    test_set = test_set.drop(columns=["area_cat"])
    
    housing = train_set.copy()
    housing_labels = housing["price"].copy()
    housing = housing.drop(columns=["price"])

    housing_num_list = list(housing.select_dtypes(include=["int64", "float64"]).columns)
  
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("feat_eng", FeatureEngineer()), 
        ("std_scalar", StandardScaler())
    ])

    ordinal_attribs = [
    "Exter.Qual",        
    "Exter.Cond",
    "Bsmt.Qual",
    "Bsmt.Cond",
    "Bsmt.Exposure",     
    "BsmtFin.Type.1",    
    "BsmtFin.Type.2",
    "Heating.QC",
    "Kitchen.Qual",
    "Garage.Finish",     
    "Garage.Qual",
    "Garage.Cond",
    "Paved.Drive"]

    one_hot_attribs = [
    "MS.Zoning",
    "Street",
    "Lot.Shape",
    "Land.Contour",
    "Utilities",
    "Lot.Config",
    "Land.Slope",
    "Neighborhood",
    "Condition.1",
    "Condition.2",
    "Bldg.Type",
    "House.Style",
    "Roof.Style",
    "Roof.Matl",
    "Exterior.1st",
    "Exterior.2nd",
    "Foundation",
    "Heating",
    "Central.Air",
    "Electrical",
    "Functional",
    "Garage.Type",
    "Sale.Type",
    "Sale.Condition"
]     
    
    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder())
    ])

    one_hot_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
   
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, housing_num_list),
        ("ord", ordinal_pipeline, ordinal_attribs),
        ("one_hot", one_hot_pipeline, one_hot_attribs)
    ])
        
    housing_prepared = full_pipeline.fit_transform(housing)
    
    #lin_reg = LinearRegression()
    #lin_reg.fit(housing_prepared, housing_labels)

    #scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
    #                     scoring="neg_mean_squared_error", cv=10)
    
    #rmse_scores = np.sqrt(-scores)
    #print("Mean for Linear Regession:", rmse_scores.mean())
    #print("Standard deviation for Linear Regession:", rmse_scores.std())
    #print()

    #tree = DecisionTreeRegressor(max_depth=4)
    
    #tree.fit(housing_prepared, housing_labels)
    #tree_scores = cross_val_score(tree, housing_prepared, housing_labels,
    #                    scoring="neg_mean_squared_error", cv=10)
    
    #rmse_scores_tree = np.sqrt(-tree_scores)
    #print("Mean for Tree:", rmse_scores_tree.mean())
    #print("Standard deviation for Tree:", rmse_scores_tree.std())
    #print()

    param_grid = [
     {'n_estimators': [3, 10, 30, 100], 'max_features': [2, 4, 6, 8, 10]},
     {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

    forest = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(forest, param_grid, cv=5, scoring='neg_mean_squared_error',
                                                return_train_score=True)


    grid_search.fit(housing_prepared, housing_labels)

    final_model = grid_search.best_estimator_

    feature_importances = final_model.feature_importances_
    joblib.dump(feature_importances, "feature_importances.pkl")

    forest_scores = cross_val_score(final_model, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)

    forest_rmse = np.sqrt(-forest_scores)

    #print("Mean for Forest:", forest_rmse.mean())
    #print("Standard deviation for Forest:", forest_rmse.std())
    #print()

    X_test = test_set.drop(columns=["price"])
    y_test = test_set["price"].copy()

    X_test_prepared = full_pipeline.transform(X_test)

    final_predictions = final_model.predict(X_test_prepared)

    final_mse = mean_squared_error(y_test, final_predictions)

    final_rmse = np.sqrt(final_mse)

    #print("Final Model Test. RMSE:", final_rmse)
    

if __name__ == '__main__':
    main()