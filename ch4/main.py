import pandas as pd
from io import StringIO

csv_data = """
A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,
"""

df = pd.read_csv(StringIO(csv_data))


def data_remove_ex():
    print(df)
    print(df.isnull().sum())
    print(df.dropna(axis=0))
    print(df.dropna(axis=1))

    # drop rows containing all NaN values
    print(df.dropna(how="all"))

    # drop rows that have number of NaNs under a certain threshold (here 4)
    print(df.dropna(thresh=4))

    # drop rows where NaN appears in specific colums
    print(df.dropna(subset=["C"]))


def data_imputing():
    from sklearn.impute import SimpleImputer
    import numpy as np

    imr = SimpleImputer(missing_values=np.nan, strategy="mean")
    imr = imr.fit(df.values)
    imputed_data = imr.transform(df.values)
    print(imputed_data)

    # pd version
    print(df.fillna(df.mean()))


df = pd.DataFrame(
    [
        ["red", "L", 13.5, "class1"],
        ["green", "M", 10.1, "class2"],
        ["blue", "XL", 15.3, "class2"],
    ]
)
df.columns = ["color", "size", "price", "classlabel"]
size_mappings = {
    "XL": 3,
    "L": 2,
    "M": 1,
}
df["size"] = df["size"].map(size_mappings)


def categorical_data_encoding():
    from sklearn.preprocessing import LabelEncoder
    import numpy as np

    inv_size_mappng = {v: k for k, v in size_mappings.items()}
    # df['size'] = df['size'].map(inv_size_mappng)

    class_mappings = {label: i for i, label in enumerate(np.unique(df["classlabel"]))}
    # Use sklearn label encoder instead
    # df['classlabel'] = df['classlabel'].map(class_mappings)

    class_le = LabelEncoder()
    y = class_le.fit_transform(df["classlabel"].values)
    print(df, y, sep="\n")


def one_hot_encoding():
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    import numpy as np

    X = df[["color", "size", "price"]].values
    color_le = LabelEncoder()
    X[:, 0] = color_le.fit_transform(X[:, 0])
    print(X)

    X = df[["color", "size", "price"]].values
    color_ohe = OneHotEncoder()
    print(color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray())

    from sklearn.compose import ColumnTransformer

    X = df[["color", "size", "price"]].values
    c_transf = ColumnTransformer(
        [
            ("onehot", OneHotEncoder(), [0]),
            ("nothing", "passthrough", [1, 2]),
        ]
    )
    print(c_transf.fit_transform(X).astype(float))

    dummies = pd.get_dummies(df[["price", "color", "size"]])
    print(dummies)

    color_ohe = OneHotEncoder(categories="auto", drop="first")
    c_transf = ColumnTransformer(
        [("onehot", color_ohe, [0]), ("nothing", "passthrough", [1, 2])]
    )
    print(
        c_transf.fit_transform(X).astype(float)
    )  # Color code 0 0 implies a blue color


def features_onto_same_scale():
    from sklearn.preprocessing import MinMaxScaler

    mms = MinMaxScaler()
    X_train_norm = mms.fit_transform(X_train)
    X_test_norm = mms.transform(X_test)

    from sklearn.preprocessing import StandardScaler

    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.fit_transform(X_test)

    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(penalty="l1", C=1.0, solver="liblinear", multi_class="ovr")
    lr.fit(X_train_std, y_train)
    print("Training accuracy:", lr.score(X_train_std, y_train))
    print("Test accuracy:", lr.score(X_test_std, y_test))


def sequential_feature_selection():
    from sklearn import datasets
    from sklearn.base import clone
    from itertools import combinations
    import numpy as np
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y
    )

    from sklearn.preprocessing import StandardScaler

    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.fit_transform(X_test)

    class SBS:
        def __init__(
            self,
            estimator,
            k_features,
            scoring=accuracy_score,
            test_size=0.25,
            random_state=1,
        ):
            self.scoring = scoring
            self.estimator = clone(estimator)
            self.k_features = k_features
            self.test_size = test_size
            self.random_state = random_state

        def fit(self, X, y):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            dim = X_train.shape[1]
            self.indices_ = tuple(range(dim))
            self.subsets_ = [self.indices_]
            score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)

            self.scores_ = [score]
            while dim > self.k_features:
                scores = []
                subsets = []

                # Try every combination without a certain feature column x'
                for p in combinations(self.indices_, r=dim - 1):
                    score = self._calc_score(X_train, y_train, X_test, y_test, p)
                    scores.append(score)
                    subsets.append(p)

                best = np.argmax(scores)
                self.indices_ = subsets[best]
                self.subsets_.append(self.indices_)
                dim -= 1

                self.scores_.append(scores[best])
            self.k_score_ = self.scores_[-1]
            return self

        def transform(self, X):
            return X[:, self.indices_]

        def _calc_score(self, X_train, y_train, X_test, y_test, indices):
            self.estimator.fit(X_train[:, indices], y_train)
            y_pred = self.estimator.predict(X_test[:, indices])
            score = self.scoring(y_test, y_pred)
            return score

    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=5)
    sbs = SBS(knn, k_features=1)
    sbs.fit(X_train_std, y_train)
    k_feat = [len(k) for k in sbs.subsets_]
    plt.plot(k_feat, sbs.scores_, marker="o")
    plt.ylim([0.7, 1.02])
    plt.ylabel("Accuracy")
    plt.xlabel("Number of features")
    plt.grid()
    plt.tight_layout()
    plt.show()

import sys
if __name__ == '__main__':
    globals()[sys.argv[1]]()