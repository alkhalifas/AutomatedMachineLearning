import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

df = pd.read_csv("make_blobs_data_3D.csv")
df = df.drop(columns = ["Unnamed: 0"])

X = df.iloc[:, :3]
print(X.head())
y = df.iloc[:, 3:].values.ravel()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Grid Search
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
svc_clf = SVC(gamma = 'auto')
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 2, 5, 10], "degree" : [1,2,3,4,5], "tol" : [0.001, 0.01, 0.1, 1]}
clf = GridSearchCV(svc_clf, parameters, cv = 5)
clf.fit(X_train, y_train)
print (clf.best_params_)
print("done")