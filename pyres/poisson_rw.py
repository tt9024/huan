import numpy as np

def get_po( t, lam ) :
    ret = []
    kf = 1.0
    ef = np.exp(-t*lam)
    for k in np.arange(t*2.0).astype(float) + 1:
        kf0 = k * kf
        ret.append((( t * lam ) ** k )/kf0)
        kf = kf0
    return np.array(ret) * ef

def get_normal(t, lam) :
    """
    get a normal distribution out of poisson
    parameters, t * lam is the mean and var.
    """
    d = np.sqrt(t*lam)
    u = t * lam
    ret = []
    c0 =  1/ (np.sqrt( np.pi * 2.0 ) * d)
    for k in np.arange( t*2.0).astype(float) + 1 :
        ed = ((k - u)**2.0)/ (2*t*lam)
        ret.append(np.exp(-ed))
    return np.array(ret) * c0
