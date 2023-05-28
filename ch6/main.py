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

# Less verbose way of the abovei implementation of cross validation score evaluation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=1)
print(f"\nCV Accuracy scores: {scores}")
print(f"CV accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
