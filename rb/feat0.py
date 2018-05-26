import numpy as np
import getdata as gd
from scipy import linalg

def daily_ret(dd) :
    ism = np.log(gd.array_from_bdict(dd, 'ism')) #[days, bars]
    lr = ism[:, 1:]-ism[:, :-1]
    lr = np.vstack( (np.zeros(ism.shape[0]), lr.T) ).T
    return lr

def daily_cov(px, a = 0.01) :
    lpx = np.log(px)
    lr = lpx[:, 1:]-lpx[:, :-1]
    k = lr.shape[1] # bars per day
    v = np.zeros((k,k))
    scl = 0
    for lr0 in lr :
        v = (1-a)*v + a*np.outer(lr0, lr0)
        scl = (1-a)*scl + a
    v /= scl
    v0 = np.sqrt(v[np.arange(k), np.arange(k)])
    v /= v0
    v = (v.T/v0).T
    return v, scl

def ret_vol_anlysis(px, steps = [1, 3, 6, 12, 30, 60]) :
    """ 
        Input: px, shape [Days, bars] of log prices observed at begining
               of each bar
        output: for each step specified, return: 
                standard deviation of logret in the future step for each bar
                mean size of abs logret in the future step for each bar
                sum of abs logret/per bar - abs sum logret/bar
    """
    k = px.shape[1]
    st = []
    abs_lr = []
    sad_abs_lr = []
    for s in steps :
        dv = []
        ddv = []
        for p0 in px :
            # for each day
            dp = np.r_[p0[s:]-p0[:-s], p0[-1] - p0[-s:]] # future s steps
            # get a hankel for length of rope
            if s == 1:
                adp = np.abs(dp)
            else :
                p00 = np.r_[ p0[1:] - p0[:-1], np.zeros(1+s) ]
                vp = linalg.hankel( p00[:-s+1], p00[-s:] )
                adp = np.sum(abs(vp),axis=1)[:k]
            dv.append(dp)
            ddv.append(adp - np.abs(dp))
        st.append(np.std(dv, axis = 0))
        abs_lr.append(np.mean(np.abs(dv), axis = 0))
        sad_abs_lr.append(np.mean(ddv, axis = 0))
    return np.array(st), np.array(abs_lr), np.array(sad_abs_lr)


