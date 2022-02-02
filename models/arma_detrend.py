import numpy as np
import ean

class ArmaResidual :
    def __init__(self,y,in_th=3,out_th=10,wt=None) :
        y0=y.copy()
        self.raw0=y0
        self.n=len(y)
        if wt is None :
            wt=np.ones(self.n).astype(float)
        assert np.min(wt) > 0, 'wt not positive'
        wt/=np.sum(wt)
        self.wt=wt.copy()
        mu=np.dot(y0,wt)
        sd=np.sqrt(np.dot((y0-mu)**2,wt))
        y1=ean.outlier(y0,mu,sd,in_th=in_th,out_th=out_th)
        self.raw=y1.copy() # before whiten

        X0=np.vstack((np.ones(self.n).astype(float),np.arange(self.n).astype(float))).T
        b0=np.dot(np.dot(np.linalg.inv(np.dot(X0.T,X0)),X0.T),y1)
        self.trend=None
        if b0[1]>2/np.sqrt(self.n) :
            # detrend
            print 'detrended with ', b0
            self.trend=b0.copy()
            y1-=np.dot(X0,b0)

        # whiten and detrend
        self.mu=np.dot(y1,self.wt)
        self.sd=np.sqrt(np.dot((y1-mu)**2,self.wt))
        y1=(y1-self.mu)/self.sd
        self.y=y1.copy()

    def raw_from_y(self,y,yix) :
        raw=y.copy()
        if self.trend is not None :
            assert len(yix)==len(y),'index length not same as y'
            X0=np.vstack((np.ones(len(y)).astype(float),yix.astype(float))).T
            raw+=np.dot(X0,self.trend)
        raw*=self.sd
        raw+=self.mu
        return raw

    def remove_arma(self,ix,max_ar=10,max_ma=4) :
        ord,params=self._est_arma(ix,max_ar=max_ar,max_ma=max_ma)
        self._set_ord(ord,params)
        self.yres=self._y_to_arma_res(self.y)
        res=self.yres[ix].copy()
        return res

    def raw_from_yres(self,yres,yresix) :
        y=self._y_from_arma_res(yres,yresix)
        return self.raw_from_y(y, yresix)

    def _y_from_arma_res(self,res,resix) :
        res_t=self.yres[resix].copy()
        return self.y[resix]-(res_t-res)

    def _est_arma(self,ix,max_ar=10,max_ma=4) :
        print 'estimating arma params: max_ar/max_ma:', max_ar,max_ma
        ord=self._est_ord(ix,max_ar=max_ar,max_ma=max_ma)
        params=self._fit(ord,ix,if_plot=False)
        return ord, params

    def _y_to_arma_res(self,y,ax=None) :
        y0=np.r_[np.zeros(self.arord),y]
        e0=np.zeros(len(y)+self.maord)
        ar=self.ar[::-1]
        ma=self.ma[::-1]
        yh=[]
        for i, yt in enumerate(y) :
            yh0=np.dot(y0[i:i+self.arord],ar)+np.dot(e0[i:i+self.maord],ma)
            e0[i+self.maord]=yt-yh0
            yh.append(yh0)
        if ax is not None:
            ax.plot(y, '.-', label='y')
            ax.plot(yh, '.-', label='fitted')
            ax.grid() ; ax.legend()
        return e0[self.maord:]

    def _set_ord(self,ord,params) :
        print 'setting arma order: ', ord, params
        self.arord=ord[0]
        self.maord=ord[1]
        self.ar=params[:self.arord]
        self.ma=params[self.arord:]

    def _est_ord(self,ix,max_ar=10,max_ma=4) :
        from statsmodels.tsa import stattools
        assert len(self.y[ix]) > 2*(max_ar + max_ma), 'not enough data to estimate'
        res=stattools.arma_order_select_ic(self.y[ix],max_ar=max_ar,max_ma=max_ma,fit_kw={'method':'css'})
        return res['bic_min_order']

    def _fit(self,ord,ix,if_plot=False) :
        assert len(self.y[ix]) > 2*(ord[0]+ord[1]), 'not enough data to fit'
        from statsmodels.tsa import arima_model
        ar=arima_model.ARMA(self.y[ix],ord)
        param=ar.fit(trend='nc')
        if if_plot :
            param.plot_predict()
        return param.params

