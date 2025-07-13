#Understanding the mechanics of linear regression through code
#Normal Equation:
import numpy as np
class LinearRegression:
    def __init__(self,method = 'gr',learning_rate = 0.01, epochs=1000):
        self.method = method
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.weights=None
        self.bias=None
    def fit(self,X,y):
        if self.method =='norm':#norm = Normal Equation
            X_fit=np.hstack((np.ones((X.shape[0],1)),X))#join series of ones
            self.weights=(np.linalg.inv(X_fit.T@X_fit))@(X_fit.T@y)
            self.bias=0
        elif self.method == 'gr':
            self.weights = np.zeros(X.shape[1])
            self.bias = 0
            for i in range(self.epochs):
                yhat= X @ self.weights + self.bias
                error=yhat-y
                dw=(2/X.shape[0])*X.T@error
                db=(2/X.shape[0])*np.sum(error)
                self.weights-=self.learning_rate*dw
                self.bias-=self.learning_rate*db
        else:
            raise ValueError("Method must be 'gr' or 'norm'")

        
    def predict(self,X):
        if self.method =='norm':
            X=np.hstack((np.ones((X.shape[0],1)),X))
        return X @ self.weights + self.bias
    





