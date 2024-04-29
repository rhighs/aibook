import pandas as pd

df_wine = pd.read_csv(
    "https://archive.ics.uci.edu/ml/" "machine-learning-databases/wine/wine.data",
    header=None,
)

from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0
)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

import numpy as np

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print("\nEigenvalues \n", eigen_vals)


tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

import matplotlib.pyplot as plt


def plot_individual_exp_cum_exp():
    plt.bar(
        range(1, 14), var_exp, align="center", label="Individual explained variance"
    )
    plt.step(
        range(1, 14), cum_var_exp, where="mid", label="Cumulative explained variance"
    )
    plt.ylabel("Explained variance ratio")
    plt.xlabel("Principal component index")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


# List[(eigen_value, eigen_vec)]
eigen_pairs = [
    (np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))
]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

print("Eigen pairs:\n", eigen_pairs)
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
print("Matrix W:\n", w)

print(
    "Projected vector X_train_std[0] onto a new 2d subspace:\n", X_train_std[0].dot(w)
)
X_train_pca = X_train_std.dot(w)


# Visualizing the transformed dataset
def plot_transorfmed_dataset():
    colors = ["r", "b", "g"]
    markers = ["o", "s", "^"]
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(
            X_train_pca[y_train == l, 0],
            X_train_pca[y_train == l, 1],
            c=c,
            label=f"Class {l}",
            marker=m,
        )
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


from plot_decision_regions import plot_decision_regions
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
lr = LogisticRegression(multi_class="ovr", random_state=1, solver="lbfgs")

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

lr.fit(X_train_pca, y_train)


def plot_decision_regions_custom_pca():
    plot_decision_regions(X_train_pca, y_train, classifier=lr)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
print("Explained vairance ratios:\n", pca.explained_variance_ratio_)

# How does each feature contributes to a given principal component?
# These contributions are called "loadings" and can be calculated by
# scaling the eigenvectors by the square root of the eigenvalues, the
# resulting can be interpreted as the correlation between the original
# features and the principal component


def plot_loadings_custom_pca():
    loadings = eigen_vecs * np.sqrt(eigen_vals)
    fig, ax = plt.subplots()
    ax.bar(range(13), loadings[:, 0], align="center")
    ax.set_ylabel("Loadings for PC 1")
    ax.set_xticks(range(13))
    ax.set_xticklabels(df_wine.columns[1:], rotation=90)
    plt.ylim([-1, 1])
    plt.tight_layout()
    plt.show()


# Same but for PCA components from scikitlearn
def plot_loadings_sklearn_pca():
    sklearn_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    fig, ax = plt.subplots()
    ax.bar(range(13), sklearn_loadings[:, 0], align="center")
    ax.set_ylabel("Loadings for PC 1")
    ax.set_xticks(range(13))
    ax.set_xticklabels(df_wine.columns[1:], rotation=90)
    plt.ylim([-1, 1])
    plt.tight_layout()
    plt.show()


np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print(f"MV {label}: {mean_vecs[label - 1]}\n")

d = 13
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter
print("Within-class scatter matrix: ", f"{S_W.shape[0]}x{S_W.shape[1]}")

print("Class label distribution: ", np.bincount(y_train))

d = 13
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
print("Scaled within-class scatter matrix: ", f"{S_W.shape[0]}x{S_W.shape[1]}")

mean_overall = np.mean(X_train_std, axis=0)
mean_overall = mean_overall.reshape(d, 1)

d = 13
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
print("Between-class scatter matrix: ", f"{S_B.shape[0]}x{S_B.shape[1]}")

"""
Note: some exaplanation here...

Inner class scatter, or within class scatter -> viriability within a class
Between class scatter                        -> variability between classes

S_W = class specific covariance matrix
S_B = class variant covarint matrix, covariance between classes means?
"""

eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [
    (np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))
]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
print("Eigenvalues in descending order:\n")

for eigen_value, eigen_vector in eigen_pairs:
    print(eigen_value)


def plot_cumsum_and_custom_lda():
    tot = sum(eigen_vals.real)
    discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
    cum_discr = np.cumsum(discr)

    plt.bar(range(1, 14), discr, align="center", label="Individual discriminability")
    plt.step(range(1, 14), cum_discr, where="mid", label="Cumulative discriminability")
    plt.xlabel('"Discriminability" ratio')
    plt.ylabel("Linear Discriminants")
    plt.ylim([-0.1, 1.1])
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    W = np.hstack(
        (eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real)
    )
    print("Transformation matrix W:\n", W)

    X_train_lda = X_train_std.dot(W)
    colors = ["r", "b", "g"]
    markers = ["o", "s", "^"]
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(
            X_train_lda[y_train == l, 0],
            X_train_lda[y_train == l, 1],
            c=c,
            label=f"Class {l}",
            marker=m,
        )

    plt.xlabel("LD 1")
    plt.xlabel("LD 2")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# LDA via scikitlearn
def plot_sklearn_lda():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    lda = LDA(n_components=2)
    X_train_lda = lda.fit_transform(X_train_std, y_train)

    lr = LogisticRegression(multi_class="ovr", random_state=1, solver="lbfgs")
    lr = lr.fit(X_train_lda, y_train)
    plot_decision_regions(X_train_lda, y_train, classifier=lr)
    plt.xlabel("LD 1")
    plt.ylabel("LD 2")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

    X_test_lda = lda.transform(X_test_std)
    plot_decision_regions(X_test_lda, y_test, classifier=lr)
    plt.xlabel("LD 1")
    plt.ylabel("LD 2")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


# t-SNE section from here
from sklearn.datasets import load_digits

digits = load_digits()

fig, ax = plt.subplots(1, 4)
for i in range(4):
    ax[i].imshow(digits.images[i], cmap="Greys")
plt.show()
print("Digits dataset shape:\n", digits.data.shape)

y_digits = digits.target
X_digits = digits.data

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, init="pca", random_state=123)
X_digits_tsne = tsne.fit_transform(X_digits)


def plot_projection(x, colors):
    import matplotlib.patheffects as PathEffects

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect="equal")
    for i in range(10):
        plt.scatter(x[colors == i, 0], x[colors == i, 1])

    for i in range(10):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects(
            [
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal(),
            ]
        )


plot_projection(X_digits_tsne, y_digits)
plt.show()

import sys
if __name__ == '__main__':
    globals()[sys.argv[1]]()
