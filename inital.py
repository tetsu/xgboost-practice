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