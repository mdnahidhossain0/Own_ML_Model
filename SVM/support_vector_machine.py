import numpy as np


class SupportVectorMachine:

    def __init__(self, learning_rate, iteration_num, lamda):
        self.learning_rate = learning_rate
        self.iteration_num = iteration_num
        self.lamda = lamda

    def fit(self, X, y):

        self.m, self.n = X.shape

        self.weights = np.zeros(self.n)
        self.bias = 0
        y = np.where(y <= 0, -1, 1)

        for _ in range(self.iteration_num):
            for idx, x_i in enumerate(X):

                condition = y[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1

                if condition:
                    dw = 2 * self.lamda * self.weights
                    db = 0
                else:
                    dw = 2 * self.lamda * self.weights - y[idx] * x_i
                    db = y[idx]

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X):

        linear_output = np.dot(X, self.weights) - self.bias
        return np.where(linear_output >= 0, 1, -1)

    def accuracy(self, X, y):

        prediction = self.predict(X)
        y = np.where(y <= 0, -1, 1)
        return np.mean(prediction == y)