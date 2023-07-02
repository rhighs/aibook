import pandas as pd

df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases"
    "/breast-cancer-wisconsin/wdbc.data",
    header=None,
)

from sklearn.preprocessing import LabelEncoder

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
print("l.classes_:", le.classes_)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=1
)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression())

pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
test_acc = pipe_lr.score(X_test, y_test)
print(f"Test accuracy: {test_acc:.3f}")

import numpy as np
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print(
        f"Fold: {k+1:02d}, ",
        f"Class distr.: {np.bincount(y_train[train])}",
        f"Acc.: {score:.3f}",
    )
mean_acc = np.mean(scores)
std_acc = np.std(scores)
print(f"\nCV Accuracy: {mean_acc:.3f} +/- {std_acc:.3f}")

# Less verbose way of the above implementation of cross validation score evaluation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=1)
print(f"\nCV Accuracy scores: {scores}")
print(f"CV accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")

# Learning curves and validation curves
# Refresher here:
# https://datascience.stackexchange.com/questions/62303/difference-between-learning-curve-and-validation-curve?newreg=9c1e5ec325444eaf96b167dd2b554902
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve

pipe_lr = make_pipeline(
    StandardScaler(), LogisticRegression(penalty="l2", max_iter=10000)
)


def plot_learning_curve():
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=pipe_lr,
        X=X_train,
        y=y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=10,
        n_jobs=1,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(
        train_sizes,
        train_mean,
        color="blue",
        marker="o",
        markersize=5,
        label="Training accuracy",
    )
    plt.fill_between(
        train_sizes,
        train_mean + train_std,
        train_mean - train_std,
        alpha=0.15,
        color="blue",
    )
    plt.plot(
        train_sizes,
        test_mean,
        color="green",
        linestyle="--",
        marker="s",
        markersize=5,
        label="Validation accuracy",
    )
    plt.fill_between(
        train_sizes,
        test_mean + test_std,
        test_mean - test_std,
        alpha=0.15,
        color="green",
    )

    plt.grid()
    plt.xlabel("Number of training examples")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.ylim([0.8, 1.03])
    plt.show()


def plot_validation_curve():
    param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    train_scores, test_scores = validation_curve(
        estimator=pipe_lr,
        X=X_train,
        y=y_train,
        param_name="logisticregression__C",
        param_range=param_range,
        cv=10,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(
        param_range,
        train_mean,
        color="blue",
        marker="o",
        markersize=5,
        label="Training accuracy",
    )
    plt.fill_between(
        param_range,
        train_mean + train_std,
        train_mean - train_std,
        alpha=0.15,
        color="blue",
    )
    plt.plot(
        param_range,
        test_mean,
        color="green",
        linestyle="--",
        marker="s",
        markersize=5,
        label="Validation accuracy",
    )
    plt.fill_between(
        param_range,
        test_mean + test_std,
        test_mean - test_std,
        alpha=0.15,
        color="green",
    )
    plt.grid()
    plt.xscale("log")
    plt.xlabel("Param C")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.ylim([0.8, 1.03])
    plt.show()


# Choose combinations of hyperparam values and choose the best one
def show_grid_search():
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [
        {"svc__C": param_range, "svc__kernel": ["linear"]},
        {"svc__C": param_range, "svc__gamma": param_range, "svc__kernel": ["rbf"]},
    ]
    gs = GridSearchCV(
        estimator=pipe_svc,
        param_grid=param_grid,
        scoring="accuracy",
        cv=10,
        refit=True,
        n_jobs=-1,
    )
    gs = gs.fit(X_train, y_train)
    print(gs.best_score_)
    print(gs.best_params_)

    clf = gs.best_estimator_
    clf.fit(X_train, y_train)
    print(f"test accuracy: {clf.score(X_test, y_test):.3f}")


def randomized_search():
    import scipy.stats
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.svm import SVC

    param_range = scipy.stats.loguniform(
        0.0001, 1000.0
    )  # -> [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    np.random.seed(1)
    print(param_range.rvs(10))

    pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
    param_grid = [
        {"svc__C": param_range, "svc__kernel": ["linear"]},
        {"svc__C": param_range, "svc__gamma": param_range, "svc__kernel": ["rbf"]},
    ]

    rs = RandomizedSearchCV(
        estimator=pipe_svc,
        param_distributions=param_grid,
        scoring="accuracy",
        refit=True,
        n_iter=20,
        cv=10,
        random_state=1,
        n_jobs=-1,
    )

    rs = rs.fit(X_train, y_train)
    print(rs.best_score_)
    print(rs.best_params_)


randomized_search()
