import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

df = pd.read_csv("make_blobs_data.csv")
print(df)

X = df.iloc[:, :3]
y = df.iloc[:, 3:].values.ravel()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from tpot import TPOTClassifier

pipeline_optimizer = TPOTClassifier(
    generations=5,  # Number of iterations to the run pipeline optimization process
    population_size=30,  # number of individuals to retain in the genetic programming population every generation.
    cv=5,  # Cross validation generator used for evaluating pipelines
    random_state=42,  # The seed of the pseudo random number generator
    verbosity=3,  # Level of communication of information while running
    scoring="f1_weighted"  # The function that evaluates the quality of the prediction
)

pipeline_optimizer.fit(X_train, y_train)
pipeline_optimizer.score(X_test, y_test)

pipeline_optimizer.export('tpot_exported_pipeline.py')
print(pipeline_optimizer.fitted_pipeline_)
