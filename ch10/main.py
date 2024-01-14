from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

X, y = make_blobs(
    n_samples=150,
    n_features=2,
    centers=3,
    cluster_std=0.5,
    shuffle=True,
    random_state=0
)

def plot_blobs_made():
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c='white',
        marker='o',
        edgecolors='black',
        s=50
    )

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_kmeans_clustering_example():
    km = KMeans(
        n_clusters=3,
        init='random',
        n_init=10,
        max_iter=100,
        tol=1e-04,
        random_state=0
    )
    y_km = km.fit_predict(X)

    plt.scatter(
        X[y_km == 0, 0],
        X[y_km == 0, 1],
        s=50, c='lightgreen',
        marker='s', edgecolor='black',
        label='Cluster 1'
    )
    plt.scatter(
        X[y_km == 1, 0],
        X[y_km == 1, 1],
        s=50, c='orange',
        marker='o', edgecolor='black',
        label='Cluster 2'
    )
    plt.scatter(
        X[y_km == 2, 0],
        X[y_km == 2, 1],
        s=50, c='lightblue',
        marker='v', edgecolor='black',
        label='Cluster 3'
    )
    plt.scatter(
        km.cluster_centers_[:, 0],
        km.cluster_centers_[:, 1],
        s=250, c='red',
        marker='*', edgecolor='black',
        label='Cluster 3'
    )
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.tight_layout()
    plt.show()
plot_kmeans_clustering_example()

def elbow_method_example():
    """
    "distortion" = within-cluster sse => SSE = "Sum of squared errors"
    this is a way of finding the "elbow" in the distortion graph,
    as we increase the number of clusters being used we should see a drop
    in distortion. We choose k in correspondence of the "elbow" as it represents
    a good trade-off between number of cluster and value for distortion.
    This example should show an elbow at k = 3.
    """
    distortions = []
    for i in range(1, 11):
        km = KMeans(
            n_clusters=i,
            init='k-means++',
            n_init=10,
            max_iter=100,
            random_state=0
        )
        km.fit(X)
        distortions.append(km.inertia_)
    plt.plot(range(1,11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.tight_layout()
    plt.show()
elbow_method_example()

def silhouette_samples_test():
    from matplotlib import cm
    from sklearn.metrics import silhouette_samples


    # ================== good clustering example ====================
    """
    the silhouette coefficients are not close to 0
    and are approximately equally far away from the average silhouette score, which is, in this case, an
    indicator of good clustering.
    """
    km = KMeans(
        n_clusters=3,
        init='k-means++',
        n_init=10,
        max_iter=300,
        tol=1e-04,
        random_state=0
    )
    y_km = km.fit_predict(X)

    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    silhouette_values = silhouette_samples(
        X, y_km, metric='euclidean'
    )

    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_values = silhouette_values[y_km == c]
        c_silhouette_values.sort()
        y_ax_upper += len(c_silhouette_values)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(
            range(y_ax_lower, y_ax_upper),
            c_silhouette_values,
            height=1.0,
            edgecolor='none',
            color=color
        )
        yticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouette_values)
    silhouette_avg = np.mean(silhouette_values)
    plt.axvline(
        silhouette_avg,
        color="red",
        linestyle="--"
    )
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.tight_layout()
    plt.show()
    # ================== bad clustering example =====================
    """
    the silhouettes now have visibly different lengths and widths, which is
    evidence of a relatively bad or at least suboptimal clustering:
    """

    km = KMeans(n_clusters=2,
        init='k-means++',
        n_init=10,
        max_iter=300,
        tol=1e-04,
        random_state=0)
    y_km = km.fit_predict(X)
    plt.scatter(X[y_km == 0, 0],
        X[y_km == 0, 1],
        s=50, c='lightgreen',
        edgecolor='black',
        marker='s',
        label='Cluster 1')
    plt.scatter(X[y_km == 1, 0],
        X[y_km == 1, 1],
        s=50,
        c='orange',
        edgecolor='black',
        marker='o',
        label='Cluster 2')
    plt.scatter(km.cluster_centers_[:, 0],
        km.cluster_centers_[:, 1],
        s=250,
        marker='*',
        c='red',
        label='Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(
        X, y_km, metric='euclidean'
    )
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper),
        c_silhouette_vals,
        height=1.0,
        edgecolor='none',
        color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouette_vals)
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--")
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.tight_layout()
    plt.show()

silhouette_samples_test()