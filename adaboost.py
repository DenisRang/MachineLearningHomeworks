import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# Plot the two-class decision scores
def plot_decision_scores(x, y, model):
    plot_colors = "rg"
    class_names = ["NOT EUROPE", "EUROPE"]
    decisions = model.decision_function(X_train)
    plot_range = (decisions.min(), decisions.max())
    for i, n, c in zip(range(2), class_names, plot_colors):
        plt.hist(decisions[y_train == i],
                 bins=10,
                 range=plot_range,
                 facecolor=c,
                 label='Class %s' % n,
                 alpha=.5,
                 edgecolor='k')

    plt.legend(loc='upper right')
    plt.ylabel('Samples')
    plt.xlabel('Score')
    plt.title('Decision Scores')
    plt.show()


pd.options.display.max_rows = 10
pd.options.display.max_columns = 20

# read data
df = pd.read_csv('countries.csv').dropna().reset_index(drop=True)
X = df.drop(['Country', 'Region'], axis=1)
y = pd.get_dummies(df['Region'])['EUROPE']

X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Create and fit an AdaBoosted decision tree
ESTIMATORS_AMOUNT = 5
model = AdaBoostClassifier(algorithm="SAMME", n_estimators=ESTIMATORS_AMOUNT)
model.fit(X_train, y_train)
plot_decision_scores(X_train,y_train,model)
print("Accuracy with outliers:  {}".format(accuracy_score(y_test, model.predict(X_test))))

# finds alphas
N = len(y_train)
f = model.estimators_
weights = model.estimator_weights_
alphas = [1 / N for i in range(N)]
for i in range(ESTIMATORS_AMOUNT):
    for j in range(N):
        # recompute alphas
        y_temp = f[i].predict(np.reshape(X_train.values[j], (1, -1)))
        if (y_train[j] == y_temp):
            alphas[j] *= np.exp(-weights[i])
        else:
            alphas[j] *= np.exp(weights[i])
    # normalize alphas
    sum_alphas = sum(alphas)
    alphas /= sum_alphas;

# Delete outliers
sorted_index_alphas = sorted(range(len(alphas)), key=alphas.__getitem__, reverse=True)
threshold_index = int(len(sorted_index_alphas) / 10)
X_train = X_train.drop(sorted_index_alphas[:threshold_index], axis=0)
y_train = y_train.drop(sorted_index_alphas[:threshold_index], axis=0)
model = AdaBoostClassifier(algorithm="SAMME", n_estimators=ESTIMATORS_AMOUNT)
model.fit(X_train, y_train)
plot_decision_scores(X_train,y_train,model)
print("Accuracy without outliers:  {}".format(accuracy_score(y_test, model.predict(X_test))))
