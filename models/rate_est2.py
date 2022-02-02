import DailyData as DD
import corr4 as corr
import numpy as np
import datetime
import exe_sim3 as es
import sklearn as sk
import matplotlib.pylot as pl
import traceback
import pandas
import event_time as evt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import gzip
import cPickle
import copy

########################
# impact model
# first fit a logit model to see how parameters decay
########################
def getMidUpDn(mid0, upsample=10,tick_sz=0.125):
    """
    For each of mid up/dn side, get the count of mid_up and mid_dn in integer
    (tick_sz) in each upsample interval mid_tot is the total up/dn in each upsample interval
    """
    print 'getMidUpDn', len(mid0)
    mid_up=[]
    mid_dn=[]
    mid=mid0[1:]-mid0[:-1]
    tot = len(mid)/upsample*upsample
    diffidx=np.arange(0,tot,upsample)
    midu=mid.copy()
    didx=np.nonzero(midu<0)[0]
    midu[didx]=0
    midusum=np.cumsum(midu)[diffidx]
    mid_up=midusum[1:]-midusum[:-1]
    mid_up=(mid_up / tick_sz + 0.5).astype(int)

    midd=mid.copy()
    uidx=np.nonzero(midd>0)[0]
    midd[uidx]=0
    middsum=np.cumsum(midd)[diffidx]
    mid_dn=middsum[1:]-middsum[:-1]
    mid_dn=(mid_dn/(-tick_sz)-0.5).astype(int)

    mid_tot=mid_up-mid_dn


def getMidUpDnContinuous(mid0,sse,tick_sz=0.125,min_sec_dt=0.001):
    """
    the process removes the continous changes in the mid_cnt and adding
    all such continous change counts to the last such sequence
    """
    print "getMidUpDn Continuous", len(mid0)

    mid_diff=np.abs(mid0[1:]-mid0[:-1])
    idx=np.nonzero(mid_diff>0)[0]
    mid_cnt=(mid_diff[idx]/tick_sz+0.5).astype(int)
    sse_idx=idx+1

    sse0=sse[sse_idx]
    mid_cnt_cum=np.cumsum(mid_cnt)
    sse_idx_nz=np.nonzero(sse0[1:]-sse0[:-1]>=(min_sec_dt-1e-7))[0]+1
    sse_idx_nz=np.append(0,sse_idx_nz)
    z_cnt=np.append(mid_cnt_cum[sse_idx_nz[1:]-1]-mid_cnt_cum[sse_idx_nz[:-1]],0)
    mid_cnt_nz=mid_cnt[sse_idx_nz]+z_cnt
    return mid_cnt_nz, sse_idx[sse_idx_nz]

def getMidChangeSlot(mid, sse, tick_sz=0.125, slot_sec=0) :
    if slot_sec != 0 :
        """
        slot the sse into slot_sec
        considering the fillings as well
        starting from the first index
        """
        slot_idx = es.sampleTSIdx(sse, slot_sec).astype(int)
    else :
        slot_idx = np.arange(len(sse))

    sse0=sse[slot_idx]
    mid0=mid[slot_idx]
    mid_diff=np.abs(mid0[1:]-mid0[:-1])
    idx=np.nonzero(mid_diff>tick_sz-(1e-10))[0]
    mid_cnt=(mid_diff[idx]/tick_sz).astype(int)
    sse_idx=idx+1
    return mid_cnt, slot_idx[sse_idx]


class VolImpactEstimationMultiDecay:
    """
    this is different with the previous two models in that
    1. change the notation from a, b to U, a
    2. add constrain that U=(N-aS)/T, so only needs to fit a 
    3. decay can be aggregation of multiple decay functions (lambda_i)
    therefore needs to fit a vector a_i for each lambda_i
    """

    def __init__(self, dd, dt, tick_size, lam_list, slot_sec=0.03, sse=None, mid=None):
        lam_list=np.array(lam_list)
        self.num_lam=len(lam_list)
        self.lam_list=lam_list
        self.dd=dd
        self.slot_sec=slot_sec

        if sse is None:
            sse, bp, bsz, ap, asz = dd.getQuotes()
            sse=sse.astype(float)/1000.0
            mid=(bp[0,:]+ap[0,:])/2
            mid_cnt,idx=getMidUpDnContinuous(mid, sse, tick_sz=tick_size, min_sec_dt=slot_sec)
        else :
            idx=np.arange(len(sse)).astype(int)
            mid_cnt=np.ones(len(sse))

        self.nk=np.sum(mid_cnt).astype(int)
        self.mid=mi[idx]
        self.sse=sse[idx]
        self.mid_cnt=mid_cnt
        print 'len(mid_cnt): ', len(mid_cnt), ' nk: ', self.nk

        #prepare for ct
        self.ct,self.ct_sse=self.fillCt()

        # second derivative stuff
        self.T=sse[-1]-sse[0]
        self.S=(self.nk-self.ct[:,-1])/self.lam_list

    def fillCt(self):
        if self.nk > len(self.mid_cnt) :
            return self.fillCt_mid_cnt_not_one()
        total_t=len(self.mid)
        ct = np.zeros((self.num_lam, total_t))
        tdiff=np.zeros(total_t)
        tdiff[1:]=self.sse[1:]-self.sse[:-1]
        tdiff_cum=np.cumsum(tdiff)

        min_add=1e-10
        d=1
        max_add=1
        lix=np.argsort(self.lam_list)[::-1]
        while max_add > min_add:
            td=self.sse[d:]-self.sse[:-d]
            for j in lix :
                this_add = np.exp(-self.lam_list[j]*td)
                ct[j,d:]+=this_add
            max_add=np.max(this_add)
            d+=1
            if d%10==0:
                print d, max_add
        return ct, self.sse

    def fillCt_mid_cnt_not_one(self):
        total_t=len(self.mid)
        ct=np.zeros((self.num_lam, total_t))
        tdiff=np.zeros(total_t)
        tdiff[1:]=self.sse[1:]-self.sse[:-1]
        tdiff_cum=np.cumsum(tdiff)

        min_add=1e-10
        elam=np.e**(-self.lam_list)

        d=1
        max_add=1
        lix=np.argsort(self.lam_list)[::-1]
        while max_add > min_add :
            td=self.sse[d:]-self.sse[:-d]
            mc=self.mid_cnt[:-d]
            for j in lix :
                this_add=np.exp(-self.lam_list[j]*td)*mc
                ct[j,d:]+=this_add
            max_add=np.max(this_add)
            d+=1
            if d%10 == 0:
                print d, max_add, lix

        print 'done'

        ct_exp=np.tile(np.nan, self.nk*self.num_lam).reshape(self.num_lam,self.nk)
        ct_idx=np.cumsum(self.mid_cnt).astype(int)
        ct_exp[:,0]=ct[:,0]
        ct_exp[:,ct_idx][:-1]]=ct[:,1:]
        pandas.Series(ct_exp.reshape(self.nk*self.num_lam)).fillna(method='ffill',inplace=True)
        ct_sse=np.tile(np.nan,self.nk)
        ct_sse[0]=self.sse[0]
        ct_sse[ct_idx[:-1]]=self.sse[1:]
        pandas.Series(ct_sse).fillna(method='ffill',inplace=True)
        return ct_exp, ct_sse

    def findUt(self,a) :
        U=(self.nk-np.dot(a,self.S))/self.T
        self.U=U
        return U+np.dot(self.ct.T,a)

    def evalLogProb(self,a=None):
        ret_a=False
        if a is None:
            print 'fitting...'
            lgp,a=self.fit()
            print 'fit got log_prob ', lgp, ', a= ', a
            ret_a=True
        ut=self.findUt(a)
        if np.min(ut)<=0 :
            raise ValueError('negative log')
        log_ut=np.log(ut)
        if ret_a:
            return [np.sum(log_ut)-self.nk,a,self.U]
        return np.sum(log_ut)

    def evalProb(self,a):
        ut=self.findUt(a)
        return np.prod(ut)

    def first_second_derivative_continuous(self,a):
        ut=self.findUt(a)
        glist=[]
        for i in np.arange(self.num_lam):
            gi=(self.ct[i,:]-self.S[i]/self.T)/ut
            glist.append(gi)

        G=np.empty(self.num_lam)
        H=np.empty((self.num_lam,self.numlam))
        for i in np.arange(self.num_lam):
            G[i]=np.sum(glist[i])
            for j in np.arange(i,self.num_lam):
                H[i,j]=-np.dot(glist[i],glist[j])
                H[j,i]=H[i,j]

        return G,H

    def fit(self,ma_iter=100,step=0.01,min_diff=1e-5):
        log_prob,a=fitLogProbNewton(self,max_iter,step,min_diff)
        return log_prob[-1],a[-1]

def fitLogProbNewton(imp, max_iter=1000,step=0.01,min_diff=1e-12,a_init=None):
    """
    this is a general version of multiple lambda with U=(N-aS)/T
    """

    if a_init != None:
        a=a_init
    else :
        while True:
            a=np.random.rand(imp.num_lam)/imp.num_lam*2*imp.lam_list
            val=np.dot(a,1.0/imp.lam_list)
            if val<1 :
                break
            print val
    print 'using initial a ', a

    log_prob=[]
    param=[]

    print 'starting with ', a
    this_iter=0
    prev_val=-1e+10
    while this_iter<max_iter:
        G,H=imp.first_second_derivative_continuous(a)
        try :
            Hinv=np.linalg.inv(H)
        except Exception as e:
            print 'probelm inverting H ', H
            print e.message
            break

        dv_adj=np.dot(Hinv,G)
        while True:
            a0=a-dv_adj
            try:
                this_val=imp.evalLogProb(a0)
            except Exception as e:
                print e.message
                print 'half dv for logProb not exist', dv_adj
                dv_adj/=2.0
                continue

            if this_val<prev_val-min_diff :
                print 'half dv for decreasing logProb ', dv_adj, a0
                dv_adj/=2.0
                continue

            a=a0
            break
        log_prob.append(this_val)
        param.append(a)
        print this_iter, ': a= ', a
        print '    log prob val: ', this_val, ' diff: ', this_val - prev_val
        print '    dv_adj: ', dv_adj
        if np.abs(this_val - prev_val)<min_diff :
            print ' done!'
            break

        prev_val=this_val
        this_iter+=1

    return log_prob, param

def getImpSurface2lam(imp, lam1_rng, lam2_rng, max_step=200) :
    a_min,a_max=lam1_rng
    b_min,b_max=lam2_rng
    a_arr=np.arange(a_min,a_max,(a_max-a_min)/max_step)
    b_arr=np.arange(b_min,b_max(b_max-b_min)/max_step)
    a_cnt=int(len(a_arr)*0.9+0.5)
    b_cnt=int(len(b_arr)*0.9+0.5)
    A,B=np.meshgrid(a_arr[:a_cnt],b_arr[:b_cnt])

    Z=np.zeros((a_cnt,b_cnt))
    for i in np.arange(a_cnt):
        for j in np.arange(b_cnt) :
            (a,b)=(A[i,j],B[i,j])
            try :
                Z[i,j]=imp.evalLogProb([a,b])
            except :
                Z[i,j]=np.nan
    return A,B,Z

def plotImpSurface(A,B,Z):
    fig=pl.figure()
    ax=fig.gca(projection='3d')
    suf=ax.plot_surface(A,B,Z,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0,antialiased=False)
    ax.zaxis=set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf,shrink=0.5,aspect=5)
    pl.show()

def eval_lam(lam,sse,mid) :
    imp=VolImpactEstimationMultiDecay(None,0,1,lam,slot_sec=0.001,sse=np.array(sse),mid=np.array(mid))
    lgp,a,U=imp.evalLogProb()
    return lgp,a,U

def pick_lam(param):
    param=np.array(param)
    lam=param[:,0]
    lgp=param[:,1]
    a=param[:,2]
    U=param[:,3]
    aU=a/U

    lgp-=np.min(lgp)
    lgp/=np.max(lgp)
    aU-=np.min(aU)
    aU/=np.max(aU)
    dlgp=lgp[1:]-lgp[:-1]
    daU=aU[1:]-aU[:-1]

    ix=np.nonzero(dlgp<daU)[0]
    l0=lam[ix[0]+1]
    return l0

def search_lam_double(lam0, lam_min, lam_max,sse,mid,dl_chg_pct=10,num_jobs=4):
    from multiprocessing.pool import ThreadPool
    pool=ThreadPool(processes=num_jobs)
    res=[]
    lam=lam_min
    lam_list=[]
    while lam<lam_max:
        l0=max(lam0,lam)
        l1=min(lam0,lam)
        if l0/l1>=1.5:
            lam_list.append(lam)
            res.apend(pool.appy_async(eval_lam,([lam0,lam],sse,mid)))
        lam*=(1.0+float(dl_chg_pct)/100.0)

    param=[]
    for lam0,res0 in zip(lam_list,res):
        lgp,a,U=res0.get()
        param.append([lam0,lgp,a,U])
    return param

