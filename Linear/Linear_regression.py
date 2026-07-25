import numpy as np

class LinearRegression:

    def __init__(self, learning_rate ,iteration_num):
        self.learning_rate = learning_rate
        self.iteration_num = iteration_num

    def fit(self ,X , Y):
        self.m , self.n = X.shape
        self.w = np.zeros(self.n)
        self.b =0
        self.X = X
        self.Y = Y

        for i in range(self.iteration_num):
            self.updateWeights()


    def updateWeights(self):
        Y_pred = self.predict(self.X)

        dw = -(2/self.m) * np.dot(self.X.T , (self.Y - Y_pred))
        db = -(2/self.m) * np.sum(self.Y - Y_pred)
        
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db
        

    def predict(self, X):
        return X.dot(self.w) + self.b

