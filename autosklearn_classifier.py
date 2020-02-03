import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

df = pd.read_csv("make_blobs_data.csv")
print(df)

X = df.iloc[:, :3]
y = df.iloc[:, 3:].values.ravel()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

import autosklearn.classification
askl_cls = autosklearn.classification.AutoSklearnClassifier()
askl_cls.fit(X_train, y_train)
predictions = askl_cls.predict(X_test)