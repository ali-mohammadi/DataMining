import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


def elbow_method(dataFrame):
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, init="k-means++")
        kmeans.fit(dataFrame)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(12, 6))
    plt.grid()
    plt.plot(range(1, 11), wcss, linewidth=2, color="red", marker="8")
    plt.xlabel("K Value")
    plt.xticks(np.arange(1, 11, 1))
    plt.ylabel("WCSS")
    plt.show()


def k_means(df, k):
    km = KMeans(n_clusters=k)
    clusters = km.fit_predict(df.iloc[:, 1:])
    df["label"] = clusters

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['blue', 'red', 'green', 'yellow', 'black']
    for x in range(0, k - 1):
        ax.scatter(df["Annual Income (k$)"][df.label == x], df["Spending Score (1-100)"][df.label == x],
                   df.Age[df.label == x], c=colors[x], s=60)
    ax.view_init(0, 0)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    ax.set_zlabel('Age')
    plt.show()

    colors = ['blue', 'red', 'green', 'yellow', 'black']
    for x in range(0, k - 1):
        plt.scatter(df["Annual Income (k$)"][df.label == x], df["Spending Score (1-100)"][df.label == x], c=colors[x],
                    s=60)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.show()


dataFrame = pd.read_csv('Mall_Customers.csv')
# dataFrame.drop(["CustomerID"], axis=1, inplace=True)

# k_means(dataFrame.iloc[:, 1:], 6)

print(dataFrame.head())