import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
from sklearn.preprocessing import StandardScaler

# read data
df = pd.read_csv('Hitters.csv').dropna()
df = df.drop(columns=df.keys()[0])
dummies = pd.get_dummies(df[['NewLeague', 'League', 'Division']])
X_ = df.drop(columns=[df.keys()[0], 'NewLeague', 'League', 'Division', 'Salary'])
X = pd.concat([X_, dummies], axis=1)
y = df.Salary

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Ridge regression
ridge_model = Ridge(alpha=10, normalize=True)
ridge_model.fit(X_train, y_train)
y_pred = ridge_model.predict(X_test)

# Evaluate model
error = mean_squared_error(y_test, y_pred)
print(error)
print(ridge_model.score(X_test,y_test))
ridge_model.fit(X, y)
series = pd.Series(ridge_model.coef_, X.columns)
print(series)

# with pd.option_context('display.max_rows', 40, 'display.max_columns', 30, 'display.width', 100):
#     print(y)
