# iris 数据集二分类问题
import time

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class Perceptron(object):

    def __init__(self, eta, n_iters):
        self.eta = eta
        self.n_iters = n_iters
        self.b = 0

    def sign(self, x, w, b):
        return np.dot(x, w) + b

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)

        for _ in range(self.n_iters):
            for xi, yi in zip(X, y):
                if yi * self.sign(xi, self.w, self.b) <= 0:
                    self.w += self.eta * np.dot(yi, xi)
                    self.b += self.eta * yi
        return self.w, self.b

    def predict(self, X):
        predict = []
        for testx in X:
            result = np.dot(testx, self.w) + self.b
            result = np.where(result <= 0, -1, 1)
            predict.append(result)
        return np.array(predict)

    def score(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)


if __name__ == '__main__':

    time1 = time.time()
    X, y = load_iris(return_X_y=True)
    X = X[:100, [0, 2]]
    # print(X.shape)
    # print(y.shape)
    y = np.where(y == 1, 1, -1)[:100]
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3)
    perceptron = Perceptron(eta=0.1, n_iters=150)
    w, b = perceptron.fit(X_train, y_train)
    y_pred = perceptron.predict(X_test)
    print(perceptron.score(y_test, y_pred))
    time2 = time.time()
    print("训练且预测花费时间：%.2f" % (time2-time1))

    x_points = np.linspace(4, 7, 10)
    y_points = -(w[0]*x_points + b)/w[1]
    plt.plot(x_points, y_points)
    pos = X_test[y_test == 1]
    neg = X_test[y_test == -1]
    plt.scatter(pos[:, 0], pos[:, 1])
    plt.scatter(neg[:, 0], neg[:, 1])
    plt.plot()
    plt.show()

    # pos = X_train[y_train == 1]
    # neg = X_train[y_train == -1]
    # plt.scatter(pos[:, 0], pos[:, 1])
    # plt.scatter(neg[:, 0], neg[:, 1])
    # plt.show()