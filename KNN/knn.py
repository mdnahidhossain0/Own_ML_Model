import numpy as np
from collections import Counter

class KNN:
    def __init__(self,k):
        self.k =k

    def fit(self,X,Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self,X):
        prediction = [self._predict(x) for x in X]
        return prediction
    
    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        
        k_nearest_labels = [self.Y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
