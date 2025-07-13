import numpy as np
def mean_squared_error(y_actual,y_pred):
    return np.mean((y_actual-y_pred)**2)
def root_mean_squared_error(y_actual,y_pred):
    return np.sqrt(np.mean((y_actual-y_pred)**2))
def mean_absolute_error(y_actual,y_pred):
    return np.mean(np.abs(y_actual-y_pred))
def r2_score(y_actual,y_pred):
    SSe=np.sum((y_actual-y_pred)**2)
    SSt=np.sum((y_actual-np.mean(y_actual))**2)
    return 1 - (SSe/SSt)
def r2_adjusted(y_actual,y_pred, X):
    n=len(y_actual)
    p=X.shape[1]
    return 1-((1 -r2_score(y_actual,y_pred)) * (n-1)/(n-p-1))
