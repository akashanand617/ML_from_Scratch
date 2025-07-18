import numpy as np
#linear regression
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
#logistic regression
def accuracy(y_actual,y_pred):
    return np.sum(y_actual == y_pred)/len(y_actual)
def precision(y_actual, y_pred):
    TP=np.sum((y_actual==1)&(y_pred==1))
    FP=np.sum((y_actual==0)&(y_pred==1))
    return TP/(TP+FP+1e-8)#for 0 positives.
def recall(y_actual, y_pred):
    TP=np.sum((y_actual==1)&(y_pred==1))
    FN=np.sum((y_actual==1)&(y_pred==0))
    return TP/(TP+FN+1e-8)#for 0 positives.
def f1_score(y_actual, y_pred):
    pre=precision(y_actual, y_pred)
    rec=recall(y_actual, y_pred)
    return (2*pre*rec)/(pre+rec)
def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])