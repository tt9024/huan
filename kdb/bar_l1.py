import l1
import numpy as np
import os
import sys
import datetime
import glob

def gen_repo(symbol, year_s, year_e, contract_list) :
    """
    year_s, year_e are integers
    contract_list could be l1.MonthlyFrontContract for CL
    """
    bar_lr=np.array([]).reshape(0,4)
    N = len(contract_list)
    ric=l1.FC(symbol, str(year_s)+'0101')[:-2]
    while year_s <= year_e :
        print 'getting year ', year_s 
        for n in np.arange(N) :
            ct = contract_list[n]
            print 'getting contract ', ct
            glob_str=symbol+'/'+ric+ct+'?_'+str(year_s)+'*.csv'
            fn=glob.glob(glob_str)
            #assert len(fn) == 1 , glob_str+' returned error '+str(fn)
            if len(fn) == 1 :
                print 'reading bar file ', fn[0]
                b=l1.bar_by_file(fn[0])
                ba, sd, ed = l1.write_daily_bar(b)
                bt=ba[:,0]
                lr=ba[:,1]
                vl=ba[:,5]
                vb=ba[:,6]-ba[:,7]
                bar_lr=np.r_[bar_lr, np.vstack((bt,lr,vl,vb)).T]
            else :
                print glob_str, ' returned error ', str(fn), ' continue...'
        year_s+=1
    return bar_lr

cl_monthly=  ['H','J','K','M','N','Q','U','V','X','Z','F','G']
cl_quartly=  ['M','U','Z','H']
cl_oddmonthly= ['K','N','U','Z','H']
cl_gcmonthly = ['J','M','Q','Z','G']
def gen_bar(symbol,cl, year_s=1998, year_e=2018) :
    ba=[]
    for y in np.arange(year_s, year_e+1) :
        ba.append(gen_repo(symbol,y,y, cl))
    np.savez_compressed(symbol+'_bar_'+str(year_s)+'_'+str(year_e),bar=ba)

class BarRepo :
    def __init__(self, symbol, repo_dir, raw_dir=None) :
        """"
        This loads the bar file for the symbol from repo
        If the bar file is not found, generate from the
        kdb's raw bars, period is 5 sec.
        """
        self.sym=symbol
        self.repo_dir=repo_dir
        self.raw_dir=raw_dir

        bf=repo_dir+'/'+symbol+'*.npz'
        if len(glob.glob(bf)) == 0 :
            self.gen_repo()
        self.bar=np.load(bf)['bar']

    def gen_repo(self) :
        # get all the 5 second bars from repo
        # generate a daily bar file
        pass

        

