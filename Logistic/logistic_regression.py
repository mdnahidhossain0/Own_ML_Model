import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate, iteration_num):
        self.learning_rate = learning_rate
        self.iteration_num = iteration_num

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.iteration_num):
            self.updateWeights()

    def updateWeights(self):
        y_hat = 1 / (1 + np.exp(-(self.X.dot(self.w) + self.b)))

        dw = (1 / self.m) * np.dot(self.X.T, (y_hat - self.Y))
        db = (1 / self.m) * np.sum(y_hat - self.Y)

        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def predict(self, X):
        y_pred = 1 / (1 + np.exp(-(X.dot(self.w) + self.b)))
        return np.where(y_pred >= 0.5, 1, 0)


