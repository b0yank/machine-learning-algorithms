import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from abc import abstractmethod

class ModelNotTrainedException(Exception):
    pass

class LinearSolver:
    @abstractmethod
    def __init__(self, C):
        if C < 0:
            raise ValueError('C must be a non-negative number.')
        
        self.C = C
        self._is_trained = False
        
    def _check_trained(self):
        if self._is_trained == False:
            raise ModelNotTrainedException('Model needs to be trained first.')

class LinearRegression(LinearSolver):    
    def __init__(self, C = 1e2):        
        super().__init__(C = C)
        
    def fit(self, X, y, epochs = 100, alpha = 0.1):
        n_samples, n_features = X.shape
        
        y = y.copy().reshape(-1, 1)
        
        if len(y) != n_samples:
            raise ValueError('Invalid y shape.')
            
        w = np.random.normal(size = (n_features, 1))
        b = 1
        
        for i in range(epochs):
            h = X.dot(w) + b
            dw = (1 / n_samples) * (X.T.dot(h - y)) + (1 / self.C) * w
            db = (1 / n_samples) * np.sum(h - y)

            w = w - alpha * dw
            b = b - alpha * db

        self.coef_ = w
        self.intercept_ = b
        self._is_trained = True
        
    def predict(self, X):
        super()._check_trained()
            
        return X * self.coef_ + self.intercept_
    
    def score(self, X, y):
        super()._check_trained()
        
        f = self.predict(X)
        y_mean = (1 / len(y)) * np.sum(y)
        ss_res = np.sum(np.square(y - f))
        ss_tot = np.sum(np.square(y - y_mean))
        
        return 1 - ss_res / ss_tot

class LogisticRegression(LinearSolver):    
    def __init__(self, C = 1e2):        
        super().__init__(C = C)
        
    def fit(self, X, y, epochs = 100, alpha = 0.1):
        n_samples, n_features = X.shape
        
        y = y.copy()
        
        if len(y) != n_samples:
            raise ValueError('Invalid y shape.')
            
        self.__labels = np.unique(y)
        self.__is_binary = len(self.__labels) == 2
        
        n_weights = 1 if self.__is_binary else len(self.__labels)
        
        # one hot encode y if not binary
        if not self.__is_binary:
            zeros = np.zeros((n_samples, n_weights))
            zeros[np.arange(n_samples), y] = 1
            y = zeros
            
        w = np.random.normal(size = (n_features, n_weights))
        b = np.ones((1, n_weights))
        
        for i in range(epochs):
            h = X.dot(w) + b
            a = self.__logistic(h)
            dw = (1 / n_samples) * (X.T.dot(a - y)) + (1/self.C) * w
            db = (1 / n_samples) * np.sum(a - y)
            
            w = w - alpha * dw
            b = b - alpha * db
        
        self.coef_ = w
        self.intercept_ = b
        self._is_trained = True
        
    def predict(self, X):
        super()._check_trained()
            
        probabilities = self.__logistic(X.dot(self.coef_) + self.intercept_)
        
        predictions = []
        for probability in probabilities:
            predictions.append(self.__labels[np.argmax(probability)])
        
        return np.array(predictions)
    
    def score(self, X, y):
        super()._check_trained()
            
        predictions = self.predict(X)
        diff = predictions - y
        
        return 1 - (np.count_nonzero(diff) / len(y))
    
    def __logistic(self, z):
        if self.__is_binary:
            return 1 / (1 + np.exp(-z))
        
        return np.array([np.exp(z[:, i]) / np.sum(np.exp(z), axis = 1) for i in range(z.shape[1])]).T
