import numpy as np
import copy

class ExponentialFit :
    """
    Class for fitting y = a exp(-b * x ) + c
    fit_diff uses dy to get rid of c, but could be highly 
       unstable with noise
    fit_grad fit a linear model on log(y) with given c 
       and then do a gradient search on c. 
       It uses fit_diff to find initial c
    fit0 fits a basic model given c. It has an option
       for reweighting of linear regression using the 
       decay factor b, but the benefit has to be 
       shown.

    Overall impression: linear regression on log(y)
    could be highly susceptible with even the
    iid additive gaussian noise. 

    It needs more data points to fit. a and b are more stable
    than c, which is the equillibrium point.  Fitting 
    less than 10 samples could be tough.

    Maybe certain prior knowledge could be used
    to imporve
    """

    def __init__(self, x, y) :
        """
        fit an exponential model for input data
        (x_i, y_i) with a model 
        y_i = a exp ( - b * x_i ) + c + e
        so that dot (e.T,e) is minimized
        It assumes x_i >= 0 and y_i >= 0.
        """
        self.x = np.array(x).copy()
        self.y = np.array(y).copy()
        assert len(np.nonzero(y<=1e-10)[0]) == 0, "y value too small!"

    def fit_diff(self,x,y) :
        """
        A quick way for using diff of y for estimation
        dy_i/dx_i = -a b exp ( -b x_i )

        log(-d_i) = log(a*b) - b x_i
        This regress gives b, use this back to original 
        function to get a and c.
        """
        d = (y[1:]-y[:-1])/(x[1:]-x[:-1])
        d = np.r_[d, 0]
        d = (d[1:]+d[:-1])/2
        Y = np.log(np.clip(-d, 1e-10,1e+10))
        X = np.vstack((np.ones(len(d)), x[1:])).T
        b = -np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T), Y)[1]
        # then regress with previous function
        Y = y
        X = np.vstack((np.ones(len(y)),np.exp(-b * x))).T
        c,a = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T), Y)
        e = np.dot(X,np.array([c,a])) - Y
        return [a,b,c],Y+e,np.sqrt(np.mean(e**2))


    def fit0(self,x,y,c=0,rewt=True):
        """
        This fits a submodel where c = 0
        It uses a linear regression on log(y) = log(a) - b * x
        """
        Y = np.log(np.clip(y-c,1e-10,1e+10))
        X = np.vstack((np.ones(len(y)),x)).T
        a,b = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T), Y)
        a=np.exp(a)
        b=-b
        Yb=a*np.exp(-b*x)+c
        e=Yb-y

        if rewt :
            while True: 
                # let w decay similarly with b
                #w = np.exp(-b*x)[::-1]
                w = np.exp(-b*x)
                W = np.diag(w/np.sum(w))
                a0,b0 = np.dot(np.dot(np.linalg.inv(np.dot(X.T,np.dot(W,X))),np.dot(X.T,W)), Y)
                a0=np.exp(a0)
                b0=-b0
                if np.abs(b0-b) < 1e-5 :
                    Yb=a*np.exp(-b*x)+c
                    e=Yb-y
                    break
                #print "b ", b, "b0 ", b0
                b = b*0.5 + b0*0.5

        return [a,b],Yb,np.sqrt(np.mean(e**2))

    def fit_grad(self,x,y,rewt=False) :
        """
        This tries to search for c, assuming it's convex w.r.t. 
        mean sqared error.
        Starts from c0 given by fit_diff, and c0_ the absolute
        minimum of y, it tries to work with three points 
        that traps the minimum c.
        """
        param0,_,e0=self.fit_diff(x,y)
        c0 = param0[-1]

        # try to find another c that traps a minimum e
        # since true c > c0_, this serves as a reference of lower bound
        c0_ = 1e-10
        param0_,_,e0_=self.fit0(x,y,c=c0_,rewt=rewt)

        print "getting initial trap set up"
        if e0_ > e0 :
            # find c1 that has e1 greater than e0
            print "Case 1: e0_ larger than e0 ",
            dc = max(c0-c0_, c0/3)
            e1 = 0
            c1 = c0
            while e1 < e0:
                c1 = c1+dc
                param1_,_,e1=self.fit0(x,y,c=c1,rewt=rewt)
            p=[c0_,c0,c1]
            e=[e0_,e0,e1]
        else :
            print "Case 2: e0_ less than e0 ",
            # c0_ is less than c0, just 
            # search for a point that is lower than c0_
            dt = (c0 + c0_)/2 - c0_
            e1 = e0
            while e1 > e0_ and dt>1e-6:
                c1 = c0_ + dt
                dt/=2
                param1,_,e1=self.fit0(x,y,c=c1,rewt=rewt)

            if dt<1e-6 :
                print "Case 3: e0_ is the boundary minimum!"
                param1,yb1,e1=self.fit0(x,y,c=c0_,rewt=rewt)
                param1.append(c0_)
                return param1, yb1, e1

            p = [c0_, c1, c0]
            e = [e0_, e1, e0]

        print "Starting with TRAP " , p , " Error ", e

        # now find the minimum amon the three poits
        err = np.mean(e)
        old_err = np.max(e)
        tol = 1e-9
        while np.abs(old_err - err) > tol :
            if p[1]-p[0] > p[2] - p[1] :
                c2 = (p[1]+p[0])/2
                param2,_,e2=self.fit0(x,y,c=c2,rewt=rewt)
                if e2 > e[1] :
                    # remove p0
                    p = [c2,p[1],p[2]]
                    e = [e2,e[1],e[2]]
                else :
                    # remove p2
                    p = [p[0],c2,p[1]]
                    e = [e[0],e2,e[1]]
            else :
                c2 = (p[1]+p[2])/2
                param2,_,e2=self.fit0(x,y,c=c2,rewt=rewt)
                if e2 > e[1] :
                    # remove p2
                    p = [p[0],p[1],c2]
                    e = [e[0],e[1],e2]
                else :
                    # remove p0
                    p = [p[1],c2,p[2]]
                    e = [e[1],e2,e[2]]
            old_err = err
            err = np.mean(e)
            #print " TRAP " , p , " Error ", e, " old_err ", old_err, " cur_err", err, " diff_err", np.abs(old_err - err)

        c = p[1]
        param,yb1,e1=self.fit0(x,y,c=c,rewt=rewt)
        param.append(c)
        print "DONE! Got c ", c, " error: ", e1
        return param,yb1,e1

    def fit(self, ifplot=False, rewt=True) :
        param,yb,e = self.fit_grad(self.x,self.y,rewt=rewt)
        if ifplot :
            import matplotlib.pylab as pl
            pl.figure()
            pl.plot(self.x,self.y,'.',label='data')
            pl.plot(self.x,yb, '-',label='fit ' + str(param) + ' e:' + str(e))
            pl.legend(loc='best')
            pl.grid()
            pl.show()
        return param,yb,e


