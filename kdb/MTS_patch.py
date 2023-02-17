import numpy as np
import repo_dbar as repo
import l1
import os
import glob
import dill

# patch vbs of various barseconds


def patch_vbs(dbar, day, utc, vbs, barsec):
    bar, col, bs = dbar.load_day(day)
    if bar is None or len(bar)==0:
        print('problem getting bars from repo on ', day)
        return
    # make sure it's a multiple
    bs_mul = barsec//bs
    if bs_mul*bs != barsec:
        print('barsec ', barsec, ' is not a multiple of repo barsec ', bs, ' on ', day)
        return

    utc_bs = dbar._make_daily_utc(day, barsec)
    nbar = len(utc)
    ix = np.clip(np.searchsorted(utc, utc_bs),0,nbar-1)
    ixz = np.nonzero(utc[ix] == utc_bs)[0]
    if len(ixz) == 0:
        print('nothing found in repo on ', day)
        return

    # reuse the existing if not provided, but aggregated at barsec
    #vbs_bs = np.zeros(len(utc_bs))
    vbs_bs = np.sum(bar[:,repo.vbsc].reshape((len(utc_bs),bs_mul)),axis=1)
    vbs_bs[ixz] = vbs[ix][ixz]

    # calculate the weight to be vol within the barsec
    vbs0 = bar[:,repo.volc].reshape((len(utc_bs),bs_mul))
    vbs0 = (vbs0.T/np.sum(vbs0,axis=1)).T
    vbs0[np.isinf(vbs0)] = 1.0/bs_mul
    vbs0[np.isnan(vbs0)] = 1.0/bs_mul

    vbs_bs0 = (vbs0.T*vbs_bs).T.reshape((len(utc_bs)*bs_mul,1))

    # write this day back
    dbar.overwrite([vbs_bs0], [day], [[repo.vbsc]], bs)
    print('!!DONE ', day)

def update_array(dbar, vbs_array, barsec):
    """
    vbs_array shape [nndays, 2], of utc and vbs
    """
    nndays, nc = vbs_array.shape
    assert nc == 2, 'vbs_array expected shape 2 (utc,vbs)'

    utc=vbs_array[:,0]
    vbs=vbs_array[:,1]
    assert utc[1]-utc[0] == barsec, 'barsec mismatch! ' + str((utc[1]-utc[0],barsec))
    start_day = l1.trd_day(vbs_array[0,0])
    end_day = l1.trd_day(vbs_array[-1,0])
    tdi = l1.TradingDayIterator(start_day)
    day = tdi.yyyymmdd()
    while day != end_day:
        patch_vbs(dbar, day, utc, vbs, barsec)
        tdi.next()
        day = tdi.yyyymmdd()

def update_array_path(array_path='/home/bfu/kisco/kr/vbs/2021_1125_2022_0114', barsec=15, repo_path = '/home/bfu/kisco/kr/repo'):
    os.system('gunzip ' + os.path.join(array_path,'*.npy.gz'))
    fn = glob.glob(os.path.join(array_path, '*.npy'))
    for f in fn:
        print('processing ', f)
        # expect file name as CL.npy
        symbol = f.split('/')[-1].split('.')[0]
        vsarr = np.load(open(f,'rb'))
        dbar = repo.RepoDailyBar(symbol, repo_path=repo_path)
        update_array(dbar, vsarr, barsec)

def update_dict(dict_file, barsec, repo_path='/home/bfu/kisco/kr/repo', symbol_list=None):
    """dict: {symbol : { 'utc': shape [ndays,2], 'vbs': shape [ndays, n] } }
    where utc has each day's first/last utc
    the barsec is given for verification purpose: barsec = (utc1-utc0)/n 
    """
    d = dill.load(open(dict_file, 'rb'))
    for symbol in d.keys():
        if symbol_list is not None:
            if symbol not in symbol_list:
                continue
        utc=d[symbol]['utc']
        vbs=d[symbol]['vbs']
        ndays, nc = utc.shape
        assert nc==2, 'utc shape not 2 for ' + symbol
        print('got ',ndays,' for ', symbol)
        dbar = repo.RepoDailyBar(symbol, repo_path=repo_path)
        for u, v in zip(utc, vbs):
            (u0,u1)=u
            day = l1.trd_day(u0)
            # LCO could have utc up until 18:00
            # turn it on when fixed in mts_repo
            #assert day == l1.trd_day(u1), 'not same trade day for %s on %d: %f-%f'%(symbol, day, u0, u1)
            utc0 = np.arange(u0,u1+barsec,barsec).astype(int)
            n = len(v)
            assert len(utc0)==n, 'vbs shape mismatch with utc for %s on %s: %d-%d'%(symbol, day, (u1-u0)//barsec,n)
            print('process %s on %s'%(symbol, day))
            patch_vbs(dbar, day, utc0, v, barsec)

def update_dict_all():
    # a scripted update, modify as needed

    # the 2 _N1 from 20220223 to 20220415 with barsec=5
    path = '/home/bfu/kisco/kr/vbs/update_0415'
    dict_files = ['0223_0302_5s.dill', '0303_0415_5s.dill']
    barsec=5
    repo_path = '/home/bfu/kisco/kr/repo'
    for df in dict_files:
        update_dict(os.path.join(path, df), barsec, repo_path=repo_path)

    # the _N2 from 20211125 to 20220415 with barsec=30
    dict_files = ['20211125_2022_0415_N2_30s.dill']
    barsec=30
    repo_path = '/home/bfu/kisco/kr/repo_nc'
    for df in dict_files:
        update_dict(os.path.join(path, df), barsec, repo_path=repo_path)


