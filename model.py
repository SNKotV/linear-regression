import numpy as np


class Model:
    def __init__(self, dimension, alpha):
        self.dimension = dimension
        self.alpha = alpha
        self.theta = np.zeros(shape=self.dimension)

    def predict(self, x):
        return self.theta.dot(x)

    def train(self, x, y, show_info=False):
        eps = 1e-6
        prev_cost = self.cost(x, y)
        cost = 1 / eps

        while abs(cost - prev_cost) > eps:
            self.gd(x, y)
            prev_cost = cost
            cost = self.cost(x, y)

            if show_info:
                print('cost: ' + str(cost))

    def cost(self, x, y):
        m = len(x)
        return sum((self.predict(x[i]) - y[i]) ** 2 for i in range(m)) / (2 * m)

    def gd(self, x, y):
        m = len(x)
        theta = self.theta
        for j in range(self.dimension):
            theta[j] = self.theta[j] - self.alpha / m * \
                       sum((self.predict(x[i]) - y[i]) * x[i][j] for i in range(m))

        self.theta = theta
