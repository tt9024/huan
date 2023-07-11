import numpy as np
import copy

"""
1. make the transaction cost small and scale with size
2. try to use C++
"""

def gradient(p0,p,f,v,t):
    # p is length nf vector
    F=f - v*p
    pp = np.r_[p0, p, 0] # shape of nf+2
    return F - (np.sign(pp[1:-1]-pp[2:])*t[1:] + np.sign(pp[:-2]-pp[1:-1])*t[:-1])

def _get_t(dp, bbo_size, one_tick_return):
    # getting a trade cost estimation given the 
    # trade size dp (with sign), estimated bbo_size and
    # the one_tick_return
    nlevel= np.round(p_abs)//bbo_size
    return  (nlevel+1)/2*one_tick_return

def _get_t_continous(dp, bbo_size, one_tick_return):
    # this is more smooth
    return ((dp**2)/bbo_size + abs(dp))*one_tick_return/2

def _get_t_vector(dp_arr, bbo_size_arr, one_tick_return):
    return np.sum(((dp_arr**2)/bbo_size_arr + np.abs(dp_arr)))*one_tick_return/2

def _pop(p0, f, v, t, p00):
    p = copy.deepcopy(p00)
    J0 = -1e+10
    cnt = 0
    while True:
        pp = np.r_[p0, p, 0]
        J = np.dot(f,p) - 0.5*np.dot(p**2,v) - np.dot(np.abs(pp[:-1]-pp[1:]),t)
        #if np.abs(J0-J) < np.abs(J)*0.00001:
        if J0-J > 0:
            break
        J0 = J
        g = f -  v*p - (np.sign(pp[1:-1]-pp[2:])*t[1:] - np.sign(pp[:-2]-pp[1:-1])*t[:-1])
        gm = np.max(np.abs(g))
        step = 1
        step= 1.0/gm*step
        p += g*step
        cnt+=1
    return p, J0, g

def _gen_DtD(bbo_size_arr, one_tick_return):
    nf = len(bbo_size_arr)-1
    D = np.zeros((nf+1,nf+2))
    ix = np.arange(nf+1).astype(int)
    D[ix,ix+1]=-1
    D[ix,ix]=1
    DtD = np.dot(np.dot(D.T,np.eye(nf+1)/bbo_size_arr*(one_tick_return/2)),D)
    return DtD

def _pop_bs(p0, f, v, p00, bbo_size_arr, one_tick_return, p_last=0):
    p = copy.deepcopy(p00)
    J0 = -1e+10
    cnt = 0
    DtD = _gen_DtD(bbo_size_arr, one_tick_return)
    while True:
        pp = np.r_[p0, p, p_last]
        dp = pp[:-1]-pp[1:]
        dbabs = np.abs(dp)
        J = np.dot(f,p) - 0.5*np.dot(p**2,v) - np.sum((dp**2)/bbo_size_arr + dbabs)*one_tick_return/2
        if J0-J > 0:
            break
        J0 = J
        pps = np.sign(dp)
        g = f -  v*p - 2*np.dot(DtD[1:-1,:],pp) - (pps[1:] - pps[:-1])*one_tick_return/2
        gm = np.max(np.abs(g))
        step = 1
        step= 1.0/gm*step
        p += g*step
        cnt+=1
    return p, J0, g

def _grad_dtd(p0, p, p_last, DtD):
    pp = np._r[p0, p, p_last]
    return np.dot(DtD[1:-1,:],pp)*2

def pop(p0, f, v, t):
    rcnt = 1000
    nf = len(f)
    J = []
    P = []
    p = f/v
    p00, j00, g00 = _pop(p0, f, v, t, p)
    j0 = -1e+10
    j = j0+1
    for c in np.arange(rcnt):
        #scl = (np.abs(p) + 1)*2
        #p00 = p + 0.001*(np.random.rand(nf)*200-100) * (scl/np.std(scl))
        p00 = p + 0.001*(np.random.rand(nf)*200-100)
        p,j,g00 = _pop(p0,f,v,t,p00)
        #p,j = _pop_dumb(p0,f,v,t,p00)

        P.append(p)
        J.append(j)

        # early stop on avg of 20
        lb = 10
        if len(J)>2*lb and np.mean(J[-lb:]) - np.mean(J[-2*lb:-lb]) < 1e-5:
            break
    ixj = np.argsort(J)
    return np.array(P)[ixj][-1], np.array(J)[ixj][-1], J, P

def pop_bs(p0, f, v, bbo_size_arr, one_tick_return, p00=None):
    # This works well!
    # setting the lb to be 5 and purturb to be 0.001*(np.random.rand(nf)*200-100)
    # good 
    rcnt = 1000
    nf = len(f)
    J = []
    P = []
    if p00 is None:
        p = f/v
    else :
        p = p00
    p00, j00, g00 = _pop_bs(p0, f, v, p, bbo_size_arr, one_tick_return)
    j0 = -1e+10
    j = j0+1
    for c in np.arange(rcnt):
        #scl = (np.abs(p) + 1)*2
        #p00 = p + 0.001*(np.random.rand(nf)*200-100) * (scl/np.std(scl))
        p00 = p + 0.001*(np.random.rand(nf)*200-100)
        p,j,g00 = _pop_bs(p0,f,v,p00, bbo_size_arr, one_tick_return)
        #p,j = _pop_dumb(p0,f,v,t,p00)

        P.append(p)
        J.append(j)

        # early stop on avg of 20
        lb = 5
        if len(J)>2*lb and np.mean(J[-lb:]) - np.mean(J[-2*lb:-lb]) < 1e-5:
            break
    ixj = np.argsort(J)
    return np.array(P)[ixj][-1], np.array(J)[ixj][-1], J, P


def pop_1(p0, f, v, p1, t0, t1):
    # run a pop for a single hop
    pp = []
    jj = []
    plo = min(p0, p1)
    phi = max(p0, p1)

    # lowest point
    pp0 = min(plo, (f+t0+t1)/v)
    if pp0 < plo:
        jj0 = f*pp0 - 0.5*v*pp0**2 - (p0-pp0)*t0 - (p1-pp0)*t1
        pp.append(pp0)
        jj.append(jj0)

    # highest point
    pp0 = max(phi, (f - (t0+t1))/v)
    if pp0 > phi:
        jj0 = f*pp0 - 0.5*v*pp0**2 - (pp0-p0)*t0 - (pp0-p1)*t1
        pp.append(pp0)
        jj.append(jj0)

    # in between 
    if p0 != p1:
        if p0 < p1:
            pp0 = np.clip((f-(t0-t1))/v, p0, p1)
            jj0 = f*pp0 - 0.5*v*pp0**2 - (pp0-p0)*t0 - (p1-pp0)*t1
        else :
            pp0 = np.clip((f-(t1-t0))/v, p1, p0)
            jj0 = f*pp0 - 0.5*v*pp0**2 - (p0-pp0)*t0 - (pp0-p1)*t1
        pp.append(pp0)
        jj.append(jj0)

        # add p0 and p1
        pp.append(p0)
        jj.append(f*p0 - 0.5*v*p0**2 -  (phi-plo)*t1)

    pp.append(p1)
    jj.append(f*p1 - 0.5*v*p1**2 -  (phi-plo)*t0)

    ix = np.argsort(jj)
    return pp[ix[-1]], jj[ix[-1]]

def pop_all_steps(p0, f, v, t, p_last=0, pp0=None):
    ### iterates from first to last back and forth
    ## this doesn't work, need perturb. Even with perturband, the converge is not linear
    ## not as good as pop()
    ## need a dynamic table
    nf = len(f)
    if pp0 is None:
        pp0=np.r_[p0, f/v, p_last]
    J = -1e+10
    while True:
        for i, fx, vx in zip(np.arange(nf, 0, -1), f[::-1], v[::-1]):
            px, jx = pop_1(pp0[i-1], fx, vx, pp0[i+1], t[i-1],t[i])
            pp0[i]=px
        p = pp0[1:-1]
        J0 = np.dot(f,p) - 0.5*np.dot(p**2,v) - np.dot(np.abs(pp0[:-1]-pp0[1:]),t)
        #print(p, J0)
        if J0 <= J:
            break
        J = J0
    return p


def _eval(p0, p, f, v, t, p_last=0):
    pp = np.r_[p0, p, p_last]
    J = np.dot(f,p) - 0.5*np.dot(p**2,v) - np.dot(np.abs(pp[:-1]-pp[1:]),t)
    return J

def grad(p0, p, f, v, t, p_last=0):
    pp = np.r_[p0, p, p_last]
    g = f - v*p - (np.sign(pp[1:-1]-pp[2:])*t[1:] - np.sign(pp[:-2]-pp[1:-1])*t[:-1])
    return g

def _pop_dumb(p0, f, v, t, p00):
    p = p00
    #p = f/v
    J0 = -1e+10
    J = _eval(p0, p, f, v, t)
    nf = len(f)

    # create an array of d
    da = []
    for i in np.arange(nf).astype(int):
        d = np.zeros(nf)
        d[i] = 1
        da.append(d[i])
        d[i] = -1
        da.append(d[i])

    while J > J0:
        J0 = J
        # find the best direction to go
        ja = []
        pa = []
        for d0 in da:
            p0_ = p+d0
            j0_ = _eval(p0, p0_, f, v, t)
            ja.append(j0_)
            pa.append(p0_)
        ix = np.argsort(ja)
        p = np.array(pa)[ix][-1]
        J = np.array(ja)[ix][-1]

    return p, J0

def find_step(p0, p, f, v, t,delta=1e-4):
    g1 = gradient(p0, p, f,v,t)
    nf = len(f)
    step = np.ones(nf)

    while True:
        pp = p0+g1*step
        g2 = gradient(p0,pp,f,v,t)
        # compare g1 and g2, if |g2[i]|_2 > |g1[i]|_2, that means we are further away from 0, set the step
        # to half
        # if |g2[i]| < |g1[i]| then we are closer, double the step
        # the goal is to find a new point that
        # np.sum(np.sign(g1*g2)==-nf
        if np.sum(np.sign(g1*g2))==-nf:
            pass

def find_pi(p1, p2, t1, t2, f, v):
    # find best position p with previous position of p1 and next position of p2
    # and tcost of t1, t2 respectively.  It's forecast/variance is f/v
    pass

