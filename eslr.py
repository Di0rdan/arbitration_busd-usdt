import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import numpy.linalg as lg
from data_manager import *

class ExpSmoothLinearRegression:
    
    def __init__(self, n, alpha=0.5):
        self.alpha = alpha
        self.XTX = np.eye(n + 1)
        self.XTy = np.zeros((n + 1, 1))
        self.predictions = []
        
    def update(self, X, y):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        y = y.reshape((X.shape[0], 1))
        
        self.XTX = self.XTX * (1 - self.alpha) + X.T @ X * self.alpha
        self.XTy = self.XTy * (1 - self.alpha) + X.T @ y * self.alpha
        
    def predict(self, X):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        pred = (X @ lg.inv(self.XTX) @ self.XTy).reshape(X.shape[0])
        self.predictions.append(pred)
        
        return pred
    
    def coef_(self):
        return (lg.inv(self.XTX) @ self.XTy).reshape((self.XTX.shape[0],))
    
class ExpSmoothLinearRegression2:
    
    def __init__(self, x_size, y_size, reg_alpha=1.0, alpha=0.5):
        self.alpha = alpha
        self.XTX = np.eye(x_size + 1) * reg_alpha
        self.XTy = np.zeros((x_size + 1, y_size))
        self.predictions = []
        
    def update(self, X, y):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        y = y.reshape((X.shape[0], self.XTy.shape[1]))
        
        self.XTX = self.XTX * (1 - self.alpha) + X.T @ X * self.alpha
        self.XTy = self.XTy * (1 - self.alpha) + X.T @ y * self.alpha
        
    def predict(self, X):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        pred = (X @ lg.inv(self.XTX) @ self.XTy).reshape((X.shape[0], self.XTy.shape[1]))
        self.predictions.append(pred)
        
        return pred
    
    def coef_(self):
        return (lg.inv(self.XTX) @ self.XTy).reshape((self.XTX.shape[0], self.XTy.shape[1]))