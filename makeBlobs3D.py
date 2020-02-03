import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=600, centers=4, n_features=3, random_state=42, cluster_std = 2)

df = pd.DataFrame(X)
df.columns = ["col_1", "col_2", "col_3"]
df["Label"] = y

df.to_csv("make_blobs_data_3D.csv")

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df["col_1"], df["col_2"], df["col_3"],  c = df["Label"], cmap='Accent', s=5)

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.savefig("blobs_scatter_3D.png")

plt.show()
