import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mlxtend.plotting import scatterplotmatrix
from mlxtend.plotting import heatmap
from scipy.integrate import quad

columns = [
    "Overall Qual",
    "Overall Cond",
    "Gr Liv Area",
    "Central Air",
    "Total Bsmt SF",
    "SalePrice",
]

df = pd.read_csv(
    "http://jse.amstat.org/v19n3/decock/AmesHousing.txt", sep="\t", usecols=columns
)
df = df.dropna(axis=0)
df["Central Air"] = df["Central Air"].map({"N": 0, "Y": 1})

print(f"dataframe shape = {df.shape}")
print(df.isnull().sum())

scatterplotmatrix(df.values, figsize=(12, 10), names=df.columns, alpha=0.5)
plt.tight_layout()
plt.show()

cm = np.corrcoef(df.values.T)
hm = heatmap(cm, row_names=df.columns, column_names=df.columns)
plt.tight_layout()
plt.show()

X = df[["Gr Liv Area"]].values
y = df["SalePrice"].values
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()


class LinearRegressionGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.array([0.0])
        self.losses_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return self.net_input(X)


def lin_regplot(X, y, model):
    plt.scatter(X, y, c="steelblue", edgecolors="white", s=70)
    plt.plot(X, model.predict(X), color="black", lw=2)


def simple_handmade_lr():
    lr = LinearRegressionGD(eta=0.1)
    lr.fit(X_std, y_std)
    plt.plot(range(1, lr.n_iter + 1), lr.losses_)
    plt.ylabel("MSE")
    plt.xlabel("Epoch")
    plt.show()

    lin_regplot(X_std, y_std, lr)
    plt.xlabel("Living area above ground (standardized)")
    plt.ylabel("Sale price (standardized)")
    plt.show()

    # obtain the original price from a std predicted price
    feature_std = sc_x.transform(np.array([[2500]]))
    target_std = lr.predict(feature_std)
    target_reverted = sc_y.inverse_transform(target_std.reshape(-1, 1))
    print(f"Sales price: ${target_reverted.flatten()[0]:.2f}")


from sklearn.linear_model import LinearRegression


def sklearn_builtin_lr():
    # using sklearn lr implementation
    slr = LinearRegression()
    slr.fit(X, y)
    y_pred = slr.predict(X)
    print(f"Slope (param w): {slr.coef_[0]:.3f}")
    print(f"Slope (bias unit b): {slr.intercept_:.3f}")

    lin_regplot(X, y, slr)
    plt.xlabel("Living area above ground in square foot")
    plt.ylabel("Sale price in U.S. Dollars")
    plt.tight_layout()
    plt.show()


def analytical_lr(X, y):
    """
    w = (XT*X)^-1 * XT*y
    """
    Xb = np.hstack((np.ones((X.shape[0], 1)), X))
    w = np.zeros(X.shape[1])
    z = np.linalg.inv(np.dot(Xb.T, Xb))
    w = np.dot(z, np.dot(Xb.T, y))
    print(f"(analytical lr) Slope {w[1]:.3f}")
    print(f"(analytical lr) Intercept {w[0]:.3f}")


analytical_lr(X, y)


def test_ransac_regressor():
    from sklearn.linear_model import RANSACRegressor

    ransac = RANSACRegressor(
        LinearRegression(),
        max_trials=100,
        min_samples=0.95,
        residual_threshold=None,
        random_state=123,
    )
    ransac.fit(X, y)

    def plot_fitted_ransac(ransac):
        # Show inliers vs outliers (that are computed via MAD)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        line_X = np.arange(3, 10, 1)
        line_y_ransac = ransac.predict(line_X[:, np.newaxis])
        plt.scatter(
            X[inlier_mask],
            y[inlier_mask],
            c="steelblue",
            edgecolors="white",
            marker="o",
            label="inliers",
        )
        plt.scatter(
            X[outlier_mask],
            y[outlier_mask],
            c="limegreen",
            edgecolor="white",
            marker="s",
            label="Outliers",
        )
        plt.plot(line_X, line_y_ransac, color="black", lw=2)
        plt.xlabel("Living area above ground in square feet")
        plt.ylabel("Sale price in U.S. dollars")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    plot_fitted_ransac(ransac)
    print(f"Slope: {ransac.estimator_.coef_[0]:.3f}")
    print(f"Intercept: {ransac.estimator_.intercept_:.3f}")

    def median_absolute_deviation(data):
        return np.median(np.abs(data - np.median(data)))

    print(f"median absolute value for dataset y = {median_absolute_deviation(y)}")

    # this one should show a greater range of inliners
    ransac = RANSACRegressor(
        LinearRegression(),
        max_trials=100,
        min_samples=0.98,
        residual_threshold=65000,
        random_state=123,
    )
    ransac.fit(X, y)

    plot_fitted_ransac(ransac)
    print(f"Slope: {ransac.estimator_.coef_[0]:.3f}")
    print(f"Intercept: {ransac.estimator_.intercept_:.3f}")


"""
Model performance evluation
"""

from sklearn.model_selection import train_test_split

target = "SalePrice"
features = df.columns[df.columns != target]
X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)


def plot_slr_residuals():
    # min-max range for plotting...
    x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
    x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    test_redisiduals = y_test_pred - y_test
    ax1.scatter(
        y_test_pred,
        test_redisiduals,
        c="limegreen",
        marker="s",
        edgecolors="white",
        label="Test data",
    )
    train_residuals = y_train_pred - y_train
    ax2.scatter(
        y_train_pred,
        train_residuals,
        c="steelblue",
        marker="o",
        edgecolors="white",
        label="Training data",
    )
    ax1.set_ylabel("Residuals")

    for ax in (ax1, ax2):
        ax.set_xlabel("Predicted values")
        ax.legend(loc="upper left")
        ax.hlines(y=0, xmin=x_min - 100, xmax=x_max + 100, color="black", lw=2)
    plt.tight_layout()
    plt.show()


from sklearn.metrics import mean_squared_error

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f"MSE train: {mse_train:.2f}")
print(f"MSE test: {mse_test:.2f}")

from sklearn.metrics import mean_absolute_error

mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f"MAE train: {mae_train:.2f}")
print(f"MAE test: {mae_test:.2f}")

from sklearn.metrics import r2_score

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f"R^2 train: {train_r2:.2f}")
print(f"R^2 test: {test_r2:.2f}")


def polynomial_features_examples():
    from sklearn.preprocessing import PolynomialFeatures

    X = np.array(
        [258.0, 270.0, 294.0, 320.0, 342.0, 368.0, 396.0, 446.0, 480.0, 586.0]
    )[:, np.newaxis]
    y = np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 360.8, 368.0, 391.2, 390.8])
    lr = LinearRegression()
    pr = LinearRegression()
    quadratic = PolynomialFeatures(degree=2)
    X_quad = quadratic.fit_transform(X)
    lr.fit(X, y)
    X_fit = np.arange(250, 600, 10)[:, np.newaxis]
    y_lin_fit = lr.predict(X_fit)
    pr.fit(X_quad, y)
    y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))
    plt.scatter(X, y, label="Training points")
    plt.plot(X_fit, y_lin_fit, label="Linear fit", linestyle="--")
    plt.plot(X_fit, y_quad_fit, label="Quadratic fit")
    plt.xlabel("Explanatory variable")
    plt.ylabel("Predicted or known target values")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


polynomial_features_examples()
