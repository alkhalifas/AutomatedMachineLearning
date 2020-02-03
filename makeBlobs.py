import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=600, centers=3, n_features=2, random_state=42, cluster_std = 2)

df = pd.DataFrame(X)
df.columns = ["col_1", "col_2"]
df["Label"] = y

df.to_csv("make_blobs_data.csv")

fg = sns.FacetGrid(data = df, hue="Label", aspect = 1.61, height = 5)
fg.map(plt.scatter, "col_1", "col_2").add_legend()

plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.savefig("blobs_scatter_2D.png")
plt.show()

