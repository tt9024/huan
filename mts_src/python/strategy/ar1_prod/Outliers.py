import numpy as np

def soft1(y,sd,ym,scl) :
    """
    ysft=soft1(y,sd,ym,scl)
    Softly truncates the values 'y' to be
    no bigger in magnitude than 'ym' 
    standard deviations 'sd'
    Inputs :
    'y' = raw return
    'sd' = standard deviation
    'ym' = max abs value for truncation, in terms of sd
    'scl' = scale controling the sharpness
            and softness of truncation
            typically 'scl'=1
            'scl' > 1 for sharper truncation
            'scl' < 1 for softer truncation
    """
    yc=np.clip(y,-30*sd,30*sd)  # this is hard limit, more for wrong numbers
    return np.log(np.cosh((yc/sd+ym)*scl)/np.cosh((yc/sd-ym)*scl))*sd/(2*scl)

def soft2(y,sd,ym,scl) :
    """
    might modify some points to smooth it.  
    """
    yc=np.clip(y,-30*sd,30*sd)
    #return (np.tanh((yc/sd+ym*1.2)*scl*1.2)-np.tanh((yc/sd-ym*1.2)*scl*1.2))*y/2.0
    # should it be yc or y
    return (np.tanh((yc/sd+ym*1.2)*scl*1.2)-np.tanh((yc/sd-ym*1.2)*scl*1.2))*yc/2.0

def soft3(y,sd_mul) :
    """
    yc = ym + tanh ((y-ym)/ub) * ub
    Note between [-1, 1], tanh is almost linear, 
    and starts to significantly throttle for more than 2.
    and flattens out after 3, which approaches to 1.
    """
    mu = np.mean(y)
    ym = np.std(y)*sd_mul
    return mu+np.tanh((y-mu)/ym)*ym
