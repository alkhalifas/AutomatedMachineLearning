import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

df = pd.read_csv("make_blobs_data.csv")
print(df)

X = df.iloc[:, :3]
y = df.iloc[:, 3:].values.ravel()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle = True)

X_train = pd.DataFrame(X_train).reset_index()
X_test = pd.DataFrame(X_test).reset_index()
y_train = pd.DataFrame(y_train).reset_index()
y_test = pd.DataFrame(y_test).reset_index()

train = pd.concat([X_train, y_train], axis = 1)
train = train.drop(columns = ["index", "Unnamed: 0", "index"])
train.columns = ["col_1", "col_2", "Label"]
test = pd.concat([X_test, y_test], axis =1)
test = test.drop(columns = ["index", "Unnamed: 0", "index"])
test.columns = ["col_1", "col_2", "Label"]


import h2o
from h2o.automl import H2OAutoML, get_leaderboard

h2o.init()

x = train.columns
y = "Label"
x.remove(y)

aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=train)

# AutoML Leaderboard
lb = aml.leaderboard

# Optionally edd extra model information to the leaderboard
lb = get_leaderboard(aml, extra_columns='ALL')

# Print all rows (instead of default 10 rows)
lb.head(rows=lb.nrows)

# The leader model is stored here
aml.leader

# If you need to generate predictions on a test set, you can make
# predictions directly on the `"H2OAutoML"` object, or on the leader
# model object directly

preds = aml.predict(test)

# or:
preds = aml.leader.predict(test)
