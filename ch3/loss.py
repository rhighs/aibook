import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def loss_1(z):
    return -np.log(sigmoid(z))


def loss_0(z):
    return -np.log(1 - sigmoid(z))


z = np.arange(-10, 10, 0.1)
sigma_z = sigmoid(z)
c1 = [loss_1(x) for x in z]
plt.plot(sigma_z, c1, label="L(W, b) if y=1")
c0 = [loss_0(x) for x in z]
plt.plot(sigma_z, c0, label="L(W, b) if y=0")
plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel("$\sigma(z)$")
plt.ylabel("L(w, b)")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
