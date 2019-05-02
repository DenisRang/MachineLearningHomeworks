import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors.classification import KNeighborsClassifier

# read data
red_df = pd.read_csv("winequality-red.csv", sep=';')
red_df["color"] = 0
white_df = pd.read_csv("winequality-white.csv", sep=';')
white_df["color"] = 1
wine_df = pd.concat([red_df, white_df])

# split data
X_train, X_test, y_train, y_test = train_test_split(wine_df.loc[:, :"color"], wine_df["color"],test_size=0.33)

# # feature scalling
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = pd.DataFrame(scaler.transform(X_train))
# X_test = pd.DataFrame(scaler.transform(X_test))

# knn prediction with cv
error=[]
for k in range(1,40):
    knn_model = KNeighborsClassifier(n_neighbors=k,p=2)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    error.append(np.mean(y_pred!=y_test))

# evalute classifier
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# plotting
plt.plot(range(1,40),
         error,
         color="red",
         linestyle="dashed",
         marker="o")
plt.ylabel("Mean Error")
plt.show()
