import numpy as np
import sys,os 
sys.path.insert(0, os.path.abspath("."))
from core.activations import softmax
class SoftmaxRegression:
    def __init__(self,learning_rate = 0.01,epochs = 1000):
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.weights=None
    def fit(self,X,y):#We need y_k which is the boolean y=k
        X=np.hstack((np.ones((X.shape[0],1)),X))
        classes=np.max(y)+1 #assuming classes were mapped to numbers beforehand
        Y=np.zeros((len(y),classes))# make a matrix of y with 0s and 1s instead of class numbers
        Y[np.arange(len(y)),y] = 1 # y is the correct col, i.e. the correct class in that observation
        for i in range(epochs):
            z= X @ self.weights
            g=softmax(z)
            dw = (1/X.shape[0])*X.T @ (g - Y)
            self.weights-=self.alpha*dw
    def predict(self,X):
        X=np.hstack((np.ones((X.shape[0],1)),X))
        g=softmax(X @ self.weights)
        return np.argmax(g,axis=1)#returning the maximum probability for each observation
    
