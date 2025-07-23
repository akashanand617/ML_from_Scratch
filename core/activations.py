import numpy as np
def sigmoid(x):
    #Making it robust/resistant to overflow from exponent
    positive = x>=0
    negative = ~positive#faster than setting another condition
    res=np.empty_like(x)#faster to add to empty array than array of 0s
    n_exp=np.exp(x[negative])#save computation time
    res[positive]= 1 / (1 + np.exp(-x[positive]))
    res[negative]= n_exp / (1 + n_exp)
    return res
def softmax(x):
    x_m=x-np.max(x)#stability
    res=np.exp(x_m)
    return res/np.sum(res,axis=1)#sum across features
