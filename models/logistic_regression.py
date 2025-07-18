import numpy as np
import sys,os 
sys.path.insert(0, os.path.abspath("."))
from core.activations import sigmoid

class LogisticRegression:
    def __init__(self,alpha=0.01,epochs=1000):
        self.alpha=alpha
        self.epochs=epochs
        self.weights=None
    def fit(self,X,y):
        self.weights=np.zeros(X.shape[1]+1)#for bias
        X=np.hstack((np.ones((X.shape[0],1)),X))#adding bias term in the matrix itself instead of updating it separately
        for i in range(self.epochs):
            z= X @ self.weights
            g=sigmoid(z)
            dw=(1/X.shape[0])*X.T@(g-y)
            self.weights-=self.alpha*dw #using loss as a metric to get the best weight
    def predict(self,X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        prob=sigmoid(X@self.weights) #using our optimal weight to get our probability
        return (prob>=0.5).astype(int)

