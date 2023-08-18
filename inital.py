# import libraries
import pandas as pd
import xgboost as xgb
import numpy as np

# import dataset
dataset = pd.read_csv("./dataset/bank-full.csv", sep=";");
dataset.dtypes

# isolate x and y variables
y = dataset.iloc[:, -1].values
X = dataset._get_numeric_data()

# split dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1502)

# transform y factor variables
y_train = np.where(y_train == "yes", 1, 0)
y_test = np.where(y_test == "yes", 1, 0)
np.mean(y_train)
np.mean(y_test)

# create xgboost matrices
Train = xgb.DMatrix(X_train, label = y_train)
Test = xgb.DMatrix(X_test, label = y_test)

# set the parameters
parameters1 = {
    'learning_rate': 0.3,
    'max_depth': 2,
    'colsample_bytree': 1,
    'subsample': 1,
    'min_child_weight': 1,
    'gamma': 0,
    'random_state': 1502,
    'eval_matric': "auc",
    'Objective': "binary:logistic"
    }

# run XGBoost

model1 = xgb.train(params=parameters1, 
                   dtrain=Train,
                   num_boost_round=200,
                   evals=[(Test, "Yes")],
                   verbose_eval=50
                   )