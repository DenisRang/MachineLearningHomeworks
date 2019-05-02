import pandas as pd
import numpy as np
from sklearn.linear_model.coordinate_descent import Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# read data
df = pd.read_csv('Hitters.csv').dropna()
df = df.drop(columns=df.keys()[0])
dummies = pd.get_dummies(df[['NewLeague', 'League', 'Division']])
X_ = df.drop(columns=[df.keys()[0], 'NewLeague', 'League', 'Division', 'Salary'])
X = pd.concat([X_, dummies], axis=1)
y = df.Salary

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Cross validation
alphas = 100 ** np.linspace(10, -2, num=100) * 0.5
lasso_cv = LassoCV(alphas=alphas, normalize=True, cv=3)
lasso_cv.fit(X_train, y_train)

# Lasso regression
lasso_model = Lasso(alpha=lasso_cv.alpha_, normalize=True)
lasso_model.fit(X_train, y_train)
y_pred = lasso_model.predict(X_test)

# Evaluate model
error = mean_squared_error(y_test, y_pred)

print(error)
lasso_model.fit(X, y)
series = pd.Series(lasso_model.coef_, X.columns)
print(series)

# with pd.option_context('display.max_rows', 40, 'display.max_columns', 30, 'display.width', 100):
#     print(y)
