import numpy as np

class rate :
    """
    The sum over 't_i' is the sum over all the events that happened prioir to 't=0'
    'mu0(t)' is defined by :
    'mu0(t) = sum_{j=1,...p} a_j * e^{-t*lambda_j)'
    
    r(t)
    returns the value of the rate 'mu(t)'.

    r.update(t)
    updates the rate function by adding the contribution of an event that happened at time 't'
    """
    def __init__(self,u,a,l):
        if len(a)!=len(l):
            raise runtimeError("a and l must be of same length")
        if np.sum(a/l)>=1:
            raise RuntimeError("process unstable: must have np.sum(a/l)<1.0")
        self.u=u
        self.a=a
        self.l=l
        self.ueff=u/(1-np.sum(a/l))
        self.b=np.zeros(len(a))

    def update(self, t):
        self.b=self.a+self.b*np.exp(-t*self.l)

    def __call__(self,t):
        return self.u+np.sum(self.b*np.exp(-t*self.l))

def generate_data(rate,T):
    """
    t=generate_data(rate.T)
    generates a random sample of event times in '[0,T]' using the rate model 'rate'
    """

    ts=[]
    tsl=0.0
    while tsl < T:
        t=0.0
        s0=rate(t)
        s=2*s0
        while s > rate(t) :
            x,y=np.random.rand(2)
            t+=-(1.0/s0)*np.log(1-x)
            s=y*s0
        rate.update(t)
        tsl+=t
        ts.append(tsl)

    return np.array(ts[:-1])


