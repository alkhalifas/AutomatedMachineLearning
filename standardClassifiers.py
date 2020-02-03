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


## KNN Classifier:
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=2)
knn_clf.fit(X_train, y_train)

knn_predictions = knn_clf.predict(X_test)
print(classification_report(y_test, knn_predictions))

## SVM Classifier:
from sklearn.svm import SVC
svc_clf = SVC(gamma='auto',
            kernel = "poly",
              C = 0.1,
              degree = 4,
              tol = 0.01,
              random_state=42)
svc_clf.fit(X_train, y_train)

svc_predictions = svc_clf.predict(X_test)
print(classification_report(y_test, svc_predictions))

#Print single accuracy
from sklearn.metrics import accuracy_score
print("SVM Accuracy: ", accuracy_score(y_test, svc_predictions, normalize=True, sample_weight=None)*100)


## SVM Classifier After GridSearch Tuning:
from sklearn.svm import SVC
svc_clf = SVC(gamma='auto',
            kernel = "linear",
              C = 1,
              degree = 1,
              tol = 0.01,
              random_state=42)
svc_clf.fit(X_train, y_train)

svc_predictions = svc_clf.predict(X_test)
print(classification_report(y_test, svc_predictions))

#Print single accuracy
from sklearn.metrics import accuracy_score
print("SVM Accuracy: ", accuracy_score(y_test, svc_predictions, normalize=True, sample_weight=None)*100)

## GaussianNB

from sklearn.naive_bayes import GaussianNB
gnb_clf = GaussianNB(priors = None, var_smoothing = 1e-9)
gnb_clf.fit(X_train, y_train)
gnb_predictions = gnb_clf.predict(X_test)
print(classification_report(y_test, gnb_predictions))
print("GNB Accuracy: ", accuracy_score(y_test, gnb_predictions, normalize=True, sample_weight=None)*100)

