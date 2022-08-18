import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kmeans import KMean


# Importing the dataset
dataset = pd.read_csv('4 Clustering\AI\credit.csv')
X = dataset.iloc[:, [1, 13]].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:13])
X[:, 1:13] = imputer.transform(X[:, 1:13])
X[:, [1, 0]] = X[:, [0, 1]]

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

print(X.shape)

k = KMean(K=5, max_iters=100, plot_steps=False)
y_pred = k.predict(X)

k.plot()