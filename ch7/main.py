from scipy.special import comb

import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

df_wine = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
    header=None,
)
df_wine.columns = [
    "Class label",
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline",
]

df_wine = df_wine[df_wine["Class label"] != 1]
y = df_wine["Class label"].values
X = df_wine[["Alcohol", "OD280/OD315 of diluted wines"]].values

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)


def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.0))
    probs = [
        comb(n_classifier, k) * error**k * (1 - error) ** (n_classifier - k)
        for k in range(k_start, n_classifier + 1)
    ]
    return sum(probs)


def plot_ensembles():
    error_range = np.arange(0.0, 1.01, 0.01)
    ens_errors = [ensemble_error(n_classifier=11, error=error) for error in error_range]
    plt.plot(error_range, ens_errors, label="Ensemble error", linewidth=2)
    plt.plot(error_range, error_range, linestyle="--", label="Base error", linewidth=2)
    plt.xlabel("Base error")
    plt.ylabel("Base/Ensemble error")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.5)
    plt.show()


def majority_voting():
    vote = np.argmax(np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6]))
    print(f"Ready-made predictions majority voting: {vote}")

    # probability based majority vote
    ex = np.array([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6]])
    p = np.average(ex, axis=0, weights=[0.2, 0.2, 0.6])
    vote = np.argmax(p)
    print(f"Ensemble class labels probabilities: {p}")
    print(f"Probability based majority voting: {vote}")


def majority_voting_classifier():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    iris = datasets.load_iris()
    X, y = iris.data[50:, [1, 2]], iris.target[50:]
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=1, stratify=y
    )

    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from maj_vote import MajorityVoteClassifier

    clf1 = LogisticRegression(penalty="l2", C=0.001, solver="lbfgs", random_state=1)

    clf2 = DecisionTreeClassifier(max_depth=1, criterion="entropy", random_state=0)

    clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric="minkowski")

    pipe1 = Pipeline([["sc", StandardScaler()], ["clf", clf1]])
    pipe3 = Pipeline([["sc", StandardScaler()], ["clf", clf3]])

    mv_clf = MajorityVoteClassifier([pipe1, clf2, pipe3])

    clfs = [pipe1, clf2, pipe3, mv_clf]

    clf_labels = [
        "LogisticRegression",
        "DecisionTreeClassifier",
        "KNeighborsClassifier",
        "MajorityVoteClassifier",
    ]
    for clf, label in zip(clfs, clf_labels):
        scores = cross_val_score(
            estimator=clf, X=X_train, y=y_train, cv=10, scoring="roc_auc"
        )
        print(f"ROC AUC: {scores.mean():.2f} (+/- {scores.std():.2f} [{label}]")

    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc

    colors = ["black", "orange", "blue", "green"]
    linestyles = [":", "--", "-.", "-"]

    for clf, label, clr, ls in zip(clfs, clf_labels, colors, linestyles):
        y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
        fpr, tpr, threshholds = roc_curve(y_true=y_test, y_score=y_pred)
        roc_auc = auc(x=fpr, y=tpr)
        plt.plot(
            fpr, tpr, color=clr, linestyle=ls, label=f"{label} (auc  {roc_auc:.2f})"
        )

    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=2)
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid(alpha=0.5)
    plt.xlabel("False positives rate (FPR)")
    plt.xlabel("True positives rate (TPR)")
    plt.show()

    # Plotting decision regions
    # Standard scaling is used to make it easier to visualize and compare regions between classfiers
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)

    x_min = X_train_std[:, 0].min() - 1
    x_max = X_train_std[:, 0].max() + 1
    y_min = X_train_std[:, 1].min() - 1
    y_max = X_train_std[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    f, axarr = plt.subplots(
        nrows=2, ncols=2, sharex="col", sharey="row", figsize=(7, 5)
    )

    from itertools import product

    for idx, clf, tt in zip(product([0, 1], [0, 1]), clfs, clf_labels):
        clf.fit(X_train_std, y_train)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
        axarr[idx[0], idx[1]].scatter(
            X_train_std[y_train == 0, 0],
            X_train_std[y_train == 0, 1],
            c="blue",
            marker="^",
            s=50,
        )
        axarr[idx[0], idx[1]].scatter(
            X_train_std[y_train == 1, 0],
            X_train_std[y_train == 1, 1],
            c="green",
            marker="o",
            s=50,
        )
        axarr[idx[0], idx[1]].set_title(tt)
    plt.text(
        -3.5,
        -5.0,
        s="Sepal width [Standardized]",
        ha="center",
        va="center",
        fontsize=12,
    )
    plt.text(
        -12.5,
        4.5,
        s="Sepal length [standardized]",
        ha="center",
        va="center",
        fontsize=12,
        rotation=90,
    )
    plt.show()

    # Apply grid search to inner params of the ensemble classifier
    from sklearn.model_selection import GridSearchCV

    params = {
        "decisiontreeclassifier__max_depth": [1, 2],
        "pipeline-1__clf__C": [0.001, 0.1, 100.0],
    }
    grid = GridSearchCV(estimator=mv_clf, param_grid=params, cv=10, scoring="roc_auc")
    grid.fit(X_train, y_train)

    for r, _ in enumerate(grid.cv_results_["mean_test_score"]):
        mean_score = grid.cv_results_["mean_test_score"][r]
        std_dev = grid.cv_results_["std_test_score"][r]
        params = grid.cv_results_["params"][r]
        print(f"{mean_score:.3f} +/- {std_dev:.2f} {params}")
    print(f"Best parameters: {grid.best_params_}")
    print(f"ROC AUC : {grid.best_score_:.2f}")


def bagging_ensembles_classifier():
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(criterion="entropy", random_state=1, max_depth=None)
    bag = BaggingClassifier(
        estimator=tree,
        n_estimators=500,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        n_jobs=1,
        random_state=1,
    )

    from sklearn.metrics import accuracy_score

    # Fit a single tree
    tree = tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)
    tree_train = accuracy_score(y_train, y_train_pred)
    tree_test = accuracy_score(y_test, y_test_pred)
    print(f"Decision tree train/test accuracies" f"{tree_train:.3f}/{tree_test:.3f}")

    # Fit the entire bag
    bag = bag.fit(X_train, y_train)
    y_train_pred = bag.predict(X_train)
    y_test_pred = bag.predict(X_test)
    bag_train = accuracy_score(y_train, y_train_pred)
    bag_test = accuracy_score(y_test, y_test_pred)
    print(f"Bagging train/test accuracies " f"{bag_train:.3f}/{bag_test:.3f}")

    x_min = X_train[:, 0].min() - 1
    x_max = X_train[:, 0].max() + 1
    y_min = X_train[:, 1].min() - 1
    y_max = X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    f, axarr = plt.subplots(
        nrows=1, ncols=2, sharex="col", sharey="row", figsize=(8, 3)
    )

    for idx, clf, tt in zip([0, 1], [tree, bag], ["Decision tree", "Bagging"]):
        clf.fit(X_train, y_train)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        axarr[idx].contourf(xx, yy, Z, alpha=0.3)
        axarr[idx].scatter(
            X_train[y_train == 0, 0], X_train[y_train == 0, 1], c="blue", marker="^"
        )
        axarr[idx].scatter(
            X_train[y_train == 1, 0], X_train[y_train == 1, 1], c="green", marker="o"
        )
        axarr[idx].set_title(tt)
    axarr[0].set_ylabel("Alcohol", fontsize=12)
    plt.tight_layout()
    plt.text(
        0,
        -0.2,
        s="OD280/OD315 of diluted wines",
        ha="center",
        va="center",
        fontsize=12,
        transform=axarr[1].transAxes,
    )
    plt.show()


def ada_boost_classification():
    y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
    yhat = np.array([1, 1, 1, -1, -1, -1, -1, -1, -1, -1])
    correct = y == yhat
    weights = np.full(10, 0.1)
    print("weights =", weights)
    epsilon = np.mean(~correct)
    print("epsilon =", epsilon)
    alpha_j = 0.5 * np.log((1 - epsilon) / epsilon)
    print("alpha_j =", alpha_j)
    update_if_correct = 0.1 * np.exp(-alpha_j * 1 * 1)
    print("updated correct =", update_if_correct)
    update_if_wrong = 0.1 * np.exp(-alpha_j * 1 * -1)
    print("updated wrong =", update_if_wrong)
    updated_weights = np.multiply(
        weights, np.exp(np.multiply(-alpha_j, np.multiply(y, yhat)))
    )
    print("updated weights =", updated_weights)
    norm_w = updated_weights / np.sum(updated_weights)
    print("normalized weights =", norm_w)

    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier

    tree = DecisionTreeClassifier(criterion="entropy", random_state=1, max_depth=1)
    ada = AdaBoostClassifier(
        estimator=tree, n_estimators=500, learning_rate=0.1, random_state=1
    )

    tree = tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)
    tree_train = accuracy_score(y_train, y_train_pred)
    tree_test = accuracy_score(y_test, y_test_pred)

    print()
    print(f"Decision tree train/test accuracies {tree_train:.3f}/{tree_test:.3f}")
    print()

    ada = ada.fit(X_train, y_train)
    y_train_pred = ada.predict(X_train)
    y_test_pred = ada.predict(X_test)
    ada_train = accuracy_score(y_train, y_train_pred)
    ada_test = accuracy_score(y_test, y_test_pred)

    print()
    print(f"AdaBoost train/test accuracies {ada_train:.3f}/{ada_test:.3f}")
    print()

    x_min = X_train[:, 0].min() - 1
    x_max = X_train[:, 0].max() + 1
    y_min = X_train[:, 1].min() - 1
    y_max = X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    f, axarr = plt.subplots(1, 2, sharex="col", sharey="row", figsize=(8, 3))
    for idx, clf, tt in zip([0, 1], [tree, ada], ["Decision tree", "AdaBoost"]):
        clf.fit(X_train, y_train)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        axarr[idx].contourf(xx, yy, Z, alpha=0.3)
        axarr[idx].scatter(
            X_train[y_train == 0, 0], X_train[y_train == 0, 1], c="blue", marker="^"
        )
        axarr[idx].scatter(
            X_train[y_train == 1, 0], X_train[y_train == 1, 1], c="green", marker="o"
        )
        axarr[idx].set_title(tt)
        axarr[0].set_ylabel("Alcohol", fontsize=12)
    plt.tight_layout()
    plt.text(
        0,
        -0.2,
        s="OD280/OD315 of diluted wines",
        ha="center",
        va="center",
        fontsize=12,
        transform=axarr[1].transAxes,
    )
    plt.show()


ada_boost_classification()
