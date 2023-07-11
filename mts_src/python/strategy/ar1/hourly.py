import numpy as np

def get_hourly(pnl, hours) :
    """pnl is shape of [nd, n]
    """
    nd, n=pnl.shape
    pc = np.hstack((np.zeros((nd,1)), np.cumsum(pnl, axis=1)))
    ix = np.r_[np.arange(0,n,hours*12),n]
    return pc[:,ix[1:]]-pc[:,ix[:-1]]


