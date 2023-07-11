#!/usr/bin/python3

import xml.etree.ElementTree as ET
import numpy as np
import datetime
import copy
import traceback
import mts_util
import subprocess

TradableKeys = ["symbol", "exch_symbol", "venue", "tick_size", \
                "point_value", "px_multiplier", "type", \
                "mts_contract", "contract_month", "mts_symbol", \
                "N", "expiration_days", "tt_security_id", \
                "tt_venue", "currency", "expiration_date","bbg_id",\
                "bbg_px_multiplier", "tickdata_id", \
                "tickdata_px_multiplier", "tickdata_timezone", \
                "lotspermin", "start_time", "end_time"]

TradableKeysSpread = ["S"]

def gen_asset_root(xml_path, assets_xml) :
    xml_cfg = xml_path + "/" + assets_xml
    root = ET.parse(xml_cfg).getroot()
    return root

def get_symbol_spread_set(cfg = '/home/mts/run/config/main.cfg'):
    mts_cfg = mts_util.MTS_Config(cfg)
    spread_dict = {}
    try:
        keys = mts_cfg.listSubKeys('Spreads')
        for k in keys:
            kl = int(mts_cfg.arrayLen('Spreads.'+k))
            sa = []
            for i in np.arange(kl).astype(int):
                sa.append(tuple(mts_cfg.getArr('Spreads.%s[%d]'%(k,i), int)))
            spread_dict[k] = copy.deepcopy(sa)
    except:
        #traceback.print_exc()
        print('no symbol spread definitions found')
    return spread_dict

def get_max_N(cfg = '/home/mts/run/config/main.cfg'):
    mts_cfg = mts_util.MTS_Config(cfg)
    return mts_cfg.get('MaxN',int)

def update_symbol_map(today_yyyy_mm_dd, xml_path, max_N = 2, assets_xml = "assets.xml", mts_symbols=None, root = None, symbol_spread_dict={}, add_prev_day_symbols=False):
    """
    Input: 
        today_yyyy_mm_dd: today's date as yyyy-mm-dd
        xml_path: the path to the mappings xml, i.e. config/symbol
        max_N: number of contracts to include for each future symbol
        assets_xml: the xml file, ie. assets.xml
        mts_symbols: optional list of mts_symbols (taken without _N, otherwise ignores _N) 
                     to generate mappings for. If None then generates all.
        root:        optional xml root if None then generates from xml_path
        spread_list:    optional dictionary of per-symbol list of N1-N2 pairs to be added.
                     i.e. {'WTI':[(1,6), (1,12)]}, to add spreads N1-N6 and N1-N12 (in addition to defaults)
                     Note, if not specified, it adds [(0,1),(0,2),(1,2)] by default
        add_prev_day_symbols: in case today's holiday (and half day) for some symbols, they won't be 
                              in today's symbol_map and therefore the bpmain/tpmain won't 
                              subscribe them.  This could lead to issues such as next day's open referring
                              to the previous day's close, making the overnight return bigger than reality.
                              Set this true to include previous day's symbols if they are not in today's map.
                              Symbols are 'WTI','Brent', etc.

    return :
        it reads assets.xml and roll_schedules/* 
        and parse into a list of tradable and a dict of venues for current day:
        tradable = [ 
            {
              tradable = ESH1
              symbol = SPX
              exch_symbol = ES
              venue = CME
              currency = USD
              tick_size = 0.25
              point_value = 50
              px_multiplier = 0.01
              type = FUT

              mts_contract = SPX_202103
              contract_month = 202103
              mts_symbol = SPX_N1
              N = 1
              expiration_days = 4
              tt_security_id = 0123455
              tt_venue = CME
              subscribed = 0
              bbg_id = ESH1 INDEX
              bbg_px_multiplier = 1.0
              tickdata_id = ESH21
              tickdata_px_multiplier = 0.01
              lotspermin = 40

              start_time = 20210214-18:00:00
              end_time   = 20210214-17:00:00
            }
            ...
          ]

        venue = {
          CME = {
                hours = [ -6, 0, 17, 0 ]
          }
        }
    """

    if mts_symbols is not None:
        # remove "_N?" from mts_symbols if exists
        symbols0 = set()
        for symbol in mts_symbols :
            symbols0.add(symbol.split('_')[0])
        mts_symbols = symbols0

    today = today_yyyy_mm_dd
    if '-' not in today_yyyy_mm_dd and len(today_yyyy_mm_dd) == 8 :
        today = today_yyyy_mm_dd[:4] + '-' + today_yyyy_mm_dd[4:6] + '-' + today_yyyy_mm_dd[6:]
    #today += " 00:00:00"

    if root is None :
        root = gen_asset_root(xml_path, assets_xml)

    tradable = []
    venues = {}

    max_Nsp = max_N
    default_spread_list = [(0,1),(0,2),(1,2)]

    for assets in root :
        sym = {}
        max_Nsp = max_N
        spread_set = set(default_spread_list)
        for asset in assets :
            if asset.tag == "symbol" :
                if mts_symbols is not None and len(mts_symbols)>0 and asset.text not in mts_symbols :
                    break
                sym["symbol"] = asset.text
                if asset.text in symbol_spread_dict.keys():
                    spread_set=set(default_spread_list+symbol_spread_dict[asset.text])
                    for (n1,n2) in spread_set:
                        max_Nsp=max(max_Nsp,n2)
            elif asset.tag == "exchange_symbol":
                sym["exch_symbol"] = asset.text
            elif asset.tag == "currency" :
                sym["currency"] = asset.text
            elif asset.tag == "exchange" :
                sym["venue"] = asset.text
            elif asset.tag == "ticksize" :
                sym["tick_size"] = asset.text
            elif asset.tag == "spread_ticksize" :
                sym["spread_tick_size"] = asset.text
            elif asset.tag == "currency" :
                sym["currency"] = asset.text
            elif asset.tag == "calendar" :
                cal_file = xml_path + "/calendars/" + asset.text

                sym["cal"] = {"file": cal_file}
                # getting the calendar for first max_N contract of today,
                # include N0, N1, etc
                c = np.genfromtxt(cal_file, delimiter = ",", dtype = "|S32")
                try :
                    ct = c[np.nonzero(c[:,0] == str.encode(today))[0][0]]
                    s = ct[2].decode()
                    e = ct[3].decode()
                    if s == "" or e == "" :
                        # not a trading day
                        raise RuntimeError(today + " is not a trading day from calendar")
                    sym["cal"]["s"] = s
                    sym["cal"]["e"] = e
                    sym["expiration_days"] = int(ct[6])
                    # read upto max_N contracts into "contracts"
                    # for N = 0, 1, .., max_Nsp
                    sym["contracts"] = []
                    for n in np.arange(max_Nsp+1).astype(int) :
                        sym["contracts"].append(ct[7+n].decode())
                    sym["start_time"] = s.split(' ')[1]
                    sym["end_time"] = e.split(' ')[1]

                except :
                    #traceback.print_exc()
                    #raise TradingDayError( today + " is not a trading day from " + cal_file)
                    #print ( today + " is not a trading day from " + cal_file)
                    break

                # set the venue
                if "venue" in sym.keys() :
                    v = sym["venue"]
                    if v in venues.keys() :
                        if venues[v]["s"] != s or venues[v]["e"] != e :
                            #print ("venue hour update for " + v)
                            if venues[v]["s"] > s :
                                venues[v]["s"] = s
                            if venues[v]["e"] < e :
                                venues[v]["e"] = e
                    else :
                        venues[v] = {"s": s, "e": e}

            elif asset.tag == "providers" :
                # getting the tt's px mul
                # and tickdata's px_multiplier and timezone
                for  p in asset :
                    if p.tag == "tt" :
                        for p0 in p :
                            if p0.tag == "px_multiplier" :
                                sym["px_multiplier"] = p0.text
                            if p0.tag == "exchange":
                                sym["tt_venue"] = p0.text
                    elif p.tag == "tickdata" :
                        for p0 in p :
                            if p0.tag == "px_multiplier" :
                                sym["tickdata_px_multiplier"] = p0.text
                            elif p0.tag == "timezone" :
                                sym["tickdata_timezone"] = p0.text
                    elif p.tag == "bbg" :
                        for p0 in p :
                            if p0.tag == "px_multiplier" :
                                sym["bbg_px_multiplier"] = p0.text

            elif asset.tag == "execution" :
                # getting the twap lotspermin
                for  p in asset :
                    if p.tag == "twap" :
                        for p0 in p :
                            if p0.tag == "lotspermin":
                                sym["lotspermin"] = p0.text

            elif asset.tag == "contracts" :
                # getting the tradable name
                for c in asset :
                    con0 = {'type':'FUT'}
                    for f in c :
                        if f.tag == "pointvalue" :
                            con0["point_value"] = f.text
                        elif f.tag == "expiry" :
                            expiry = f.text
                            if expiry not in sym["contracts"] :
                                con0 = {}
                                break
                            n = np.nonzero(np.array(sym["contracts"])==expiry)[0][0]
                            con0["contract_month"] = expiry
                            con0["N"] = n
                        elif f.tag == "tt_exchange_id" :
                            con0["tradable"] = f.text
                        elif f.tag == "tt_security_id" :
                            con0["tt_security_id"] = f.text
                        elif f.tag == "expiration_date" :
                            con0["expiration_date"] = f.text
                        elif f.tag == "bbg_id" :
                            con0["bbg_id"] = f.text
                        elif f.tag == "tickdata_id" :
                            con0["tickdata_id"] = f.text
                        elif f.tag == "symbol" :
                            con0["mts_contract"] = f.text

                    if len(con0.keys()) >= 4 :
                        sym[n] = copy.deepcopy(con0)

            elif asset.tag == "spreads":
                # getting the tradable name
                for c in asset :
                    con0 = {'type':'MLEG'}
                    for f in c :
                        if f.tag == "pointvalue" :
                            con0["point_value"] = f.text
                        elif f.tag == "expiry" :
                            expiry1,expiry2 = f.text.split('-')
                            if expiry1 not in sym["contracts"] or expiry2 not in  sym["contracts"]:
                                con0 = {}
                                break
                            n1 = np.nonzero(np.array(sym["contracts"])==expiry1)[0][0]
                            n2 = np.nonzero(np.array(sym["contracts"])==expiry2)[0][0]
                            if (n1,n2) not in spread_set:
                                break
                            con0["contract_month"] = "%s-%s"%(expiry1,expiry2)
                            con0["N"] = n1
                            con0["S"] = n2

                        elif f.tag == "tt_exchange_id" :
                            con0["tradable"] = f.text
                        elif f.tag == "tt_security_id" :
                            con0["tt_security_id"] = f.text
                        elif f.tag == "bbg_id" :
                            con0["bbg_id"] = f.text
                        elif f.tag == "tickdata_id" :
                            con0["tickdata_id"] = f.text
                        elif f.tag == "symbol" :
                            con0["mts_contract"] = f.text

                    if len(con0.keys()) >= 4 and 'S' in con0.keys():
                        if n1 not in sym.keys():
                            raise RuntimeError("spread contract defined before underlying")
                        if 'spreads' not in sym[n1].keys():
                            sym[n1]['spreads'] = []
                        sym[n1]['spreads'].append(copy.deepcopy(con0))

        # finish parsing this asset into sym
        # write maxN tradable into trd 
        # 
        for n in np.arange(max_Nsp+1).astype(int) :
            if n not in sym.keys() :
                continue

            sym0 = copy.deepcopy(sym)
            sym0.update(sym[n])

            # for underlying types (FUT)
            if n <= max_N :
                sym0["mts_symbol"] = sym["symbol"] + "_N"+str(n)
                trd0 = {}
                for k in TradableKeys :
                    # write to the files
                    try :
                        trd0[k] = sym0[k]
                    except :
                        if 'tickdata' not in k:
                            print (k + " is not defined " + str(sym0["symbol"]))
                        trd0[k] = 'None'

                trd0["tradable"] = sym0["tradable"]
                if trd0['expiration_date'] < today :
                    # only add contracts expires later than today
                    continue
                tradable.append(copy.deepcopy(trd0))

            # for the spread with n1=n
            if 'spreads' not in sym0.keys():
                continue
            for spd_con in sym0['spreads']:
                # we don't need this check, all spd_con should be in the set
                #if (n, spd_con['S']) not in spread_set:
                #    continue
                trd0 = {}

                # for N0, only add spread of N0-N1, nothing else
                if str(n) == '0' and str(spd_con['S']) != '1':
                    continue
                
                sym1=copy.deepcopy(sym0)
                sym1.update(spd_con)
                for k in TradableKeys + TradableKeysSpread:
                    # write to the files
                    try :
                        trd0[k] = sym1[k]
                    except :
                        if 'tickdata' not in k :
                            print (k + " is not defined " + str(sym0["symbol"]))
                        trd0[k] = 'None'

                trd0["mts_symbol"] = sym1["symbol"] + "_N"+str(n)+'-'+sym1['symbol']+'_N'+str(spd_con['S'])
                trd0["tradable"] = sym1["tradable"]
                trd0['tick_size'] = sym1['spread_tick_size']
                tradable.append(copy.deepcopy(trd0))


    if add_prev_day_symbols:
        # called from launch to add previous day's symbol in the symbol map for subscription/reference purpose for MTS engine
        tdi = mts_util.TradingDayIterator(today_yyyy_mm_dd)
        tdi.begin()
        prev_day=tdi.prev()
        tp, vp = update_symbol_map(prev_day, xml_path, max_N = max_N, assets_xml = assets_xml, mts_symbols=mts_symbols, root = root, symbol_spread_dict=symbol_spread_dict, add_prev_day_symbols=False)
        # merge symbols that are in tp into tradable
        cur_symbols = []
        for td in tradable:
            cur_symbols.append(td['symbol'])
        for td in tp:
            if td['symbol'] not in cur_symbols:
                tradable.append(td)
        # merge venue
        for vk in vp.keys():
            if vk not in venues.keys():
                venues[vk] = vp[vk]

    return tradable, venues


def writeToConfigure(tradable, venues, cfg_path, map_file = "symbol_map.cfg") :
    """
    persist the two dictionaries to cfg_path/map_file
    """

    with open(cfg_path + "/" + map_file, "w") as f :
        f.write("tradable = {\n")
        for trd in tradable :
            f.write("    " + trd["tradable"] + " = {\n")
            for key in TradableKeys+TradableKeysSpread:
                if key in trd.keys():
                    f.write("        " + key + " = " + str(trd[key])+ "\n")
            f.write("    " + "}\n")

        f.write("}\n")
        f.write("venue = {\n")
        for key in venues.keys() :
            f.write("    " + key + " = {\n")
            s = venues[key]["s"]
            e = venues[key]["e"]

            sh = int(s.strip().split(" ")[1].split(":")[0])
            sm = int(s.strip().split(" ")[1].split(":")[1])
            eh = int(e.strip().split(" ")[1].split(":")[0])
            em = int(e.strip().split(" ")[1].split(":")[1])

            if sh > eh or (sh == eh and sm >= em) :
                sh = sh - 24

            f.write("        hours = [ " + str(sh) + ", " + str(sm) + ", " + str(eh) + ", " + str(em) + " ]\n")
            f.write("    }\n")
        f.write("}\n")

class SymbolMap :
    def __init__(self, xml_path = '/home/mts/run/config/symbol', max_N = 2, assets_xml = "assets.xml", default_symbol_spread_dict={}, main_cfg_fn = None):
        self.xml_path = xml_path
        assert max_N <= 6, "max_N more than 6 - calendar file limit 6, run twice to get 12"
        self.max_N = max_N
        self.assets_xml = assets_xml
        self.assets_root = None
        self.default_symbol_spread_dict = default_symbol_spread_dict 
        if main_cfg_fn is not None:
            self.default_symbol_spread_dict = get_symbol_spread_set(main_cfg_fn)
            self.max_N = get_max_N(main_cfg_fn)

    def update_config(self, today_yyyy_mm_dd, cfg_path = 'config', map_file = 'symbol_map.cfg', mts_symbols = None, symbol_spread_dict={}):
        if len(symbol_spread_dict) == 0:
            symbol_spread_dict = self.default_symbol_spread_dict
        self._get_assets_root()
        t,v = update_symbol_map(today_yyyy_mm_dd, self.xml_path, self.max_N, self.assets_xml, mts_symbols = mts_symbols, root = self.assets_root, symbol_spread_dict=symbol_spread_dict)
        writeToConfigure(t, v, cfg_path, map_file)

    def get_tradable_map(self, today_yyyy_mm_dd, mts_key = False, mts_symbols = None, symbol_spread_dict={}, add_prev_day=False, optional_key_name=None) :
        """
        if mts_key is True, map have key on mts_symbol, otherwise, key is tradable
        return :
            dict with key being either tradable or mts_symbol, value being the tradable dict
        """
        if len(symbol_spread_dict) == 0:
            symbol_spread_dict = self.default_symbol_spread_dict
        self._get_assets_root()
        t,v = update_symbol_map(today_yyyy_mm_dd, self.xml_path, self.max_N, self.assets_xml, mts_symbols=mts_symbols, root=self.assets_root, symbol_spread_dict=symbol_spread_dict)
        smap = {} # a map from mts_symbol to a tradable
        key_name = 'mts_symbol'
        if not mts_key:
            if optional_key_name is not None:
                key_name = optional_key_name
            else:
                key_name = 'tradable'
        for t0 in t :
            #k = t0["mts_symbol"] if mts_key else t0["tradable"]
            k = t0[key_name]
            smap[k] = copy.deepcopy(t0)

        if add_prev_day:
            # add symbols that were tradable on previous day but not on today
            # to the smap
            tdi = mts_util.TradingDayIterator(today_yyyy_mm_dd)
            tdi.begin()
            prev_day = tdi.prev()
            smap_prev = self.get_tradable_map(prev_day, mts_key= mts_key, mts_symbols=mts_symbols, \
                             symbol_spread_dict=symbol_spread_dict, add_prev_day = False, optional_key_name=optional_key_name)
            for k in smap_prev.keys():
                if k not in smap.keys():
                    smap[k] = smap_prev[k]
        return smap

    @staticmethod
    def is_mts_symbol(mts_or_tradable):
        """look for pattern that either has '_N[012...9]' or '_yyyymm'
        """
        tf = mts_or_tradable.split('_')
        if len(tf)>1:
            tf0 = tf[-1]
            if tf0[0]=='N':
                try:
                    n_contract = int(tf0[1:])
                    return True
                except:
                    pass
            elif len(tf0)==6:
                try:
                    contract_month = int(tf0)
                    return True
                except:
                    pass
        return False

    def get_tinfo(self, mts_or_tradable, yyyymmdd = None, is_mts_symbol = False, symbol_spread_dict={}, add_prev_day=False, optional_key_name=None) :
        if len(symbol_spread_dict) == 0:
            symbol_spread_dict = self.default_symbol_spread_dict
        if yyyymmdd is None :
            yyyymmdd = datetime.datetime.now().strftime('%Y%m%d')

        is_mts_symbol0 = SymbolMap.is_mts_symbol(mts_or_tradable)
        if is_mts_symbol is None:
            is_mts_symbol=is_mts_symbol0
        elif is_mts_symbol0 != is_mts_symbol:
            print('got %s, which %s a MTS symbol, but was given otherwise, please fix'%(mts_or_tradable, 'is' if is_mts_symbol0 else 'is NOT'))
        try :
            # try it as mts symbol, 
            mts_symbols = [mts_or_tradable] if is_mts_symbol else None

            # this uses the mts_symbol as key, i.e. WTI_N1
            smap = self.get_tradable_map(yyyymmdd, mts_key = True, mts_symbols = mts_symbols, symbol_spread_dict=symbol_spread_dict, add_prev_day=add_prev_day, optional_key_name=optional_key_name)
            return smap[mts_or_tradable]
        except :
            try :
                if is_mts_symbol0 and optional_key_name is None:
                    # this uses the mts_contract as key, i.e. WTI_202209
                    smap = self.get_tradable_map(yyyymmdd, mts_key = False, mts_symbols = mts_symbols, symbol_spread_dict=symbol_spread_dict,add_prev_day=add_prev_day, optional_key_name='mts_contract')
                else:
                    # this uses the 'exchange symbol as key, i.e. CLU2
                    smap = self.get_tradable_map(yyyymmdd, mts_key = False, mts_symbols = mts_symbols, symbol_spread_dict=symbol_spread_dict,add_prev_day=add_prev_day, optional_key_name=optional_key_name)
                return smap[mts_or_tradable]
            except :
                raise KeyError("failed to get tinfo " + mts_or_tradable)
            
    def list_symbol(self, today_yyyymmdd = None, mts_symbol_list=None, add_prev_day=False) :
        """
        return a list of symbol, such as WTI, SPX, from mts_symbol, such as WTI_N1. 
        if None then list all defined in assets.xml for today_yyyymmdd
        """
        if today_yyyymmdd is None :
            today_yyyymmdd = datetime.datetime.now().strftime('%Y%m%d')
        tdi = mts_util.TradingDayIterator(today_yyyymmdd)
        today_yyyymmdd=tdi.begin()
        smap = self.get_tradable_map(today_yyyymmdd, mts_key = True, mts_symbols=mts_symbol_list, add_prev_day=add_prev_day)
        if mts_symbol_list is None:
            mts_symbol_list = smap.keys()

        symbols = []
        for k in mts_symbol_list:
            sym = smap[k]['symbol']
            if sym not in symbols: 
                symbols.append(sym)
        return symbols

    def get_tradable_from_mts_symbol(self, mts_symbol_arr, trade_day) :
        """
        get an array of tradables given mts_symbols and a trading day
        Note the returned array has same length with mts_symbol_arr.  If not a trading day
        for the corresponding mts_symbol, a 'None' is put in place
        """
        tm = self.get_tradable_map(trade_day, mts_key=True, mts_symbols = mts_symbol_arr)
        ret = []
        for ms in mts_symbol_arr:
            if ms not in tm.keys() :
                ret.append(None)
            else :
                ret.append(tm[ms]["tradable"])
        return ret

    def get_contract_from_symbol(self, mts_symbol_no_N, trade_day, add_prev_day=False, include_spread=True, extra_N=[]):
        """
        mts_symbol_no_N: mts_symbol that doesn't have '_N', i.e. WTI
        trade_day: a trading day in YYYYMMDD
        include_spread: if True, includes all spreads within the limit of maxn, 
                        i.e. if max=2, N0-N1, N0-N2, N1-N2
                        if False, no spread is included
        extra_N: list of N beyond the max_N, i.e [6,9,12] for max_N=2 (cannot include spread)
        return: a list of all contracts that matches with the symbol on
                the day, upto max_N defined in self.max_N
        NOTE - since the calendar file has only contracts upto N_6, in order to extend to N_12
               it runs again at half year earlier
        """
        assert (not include_spread) or (len(extra_N) == 0)
        # handle the extra_N here
        max_N = self.max_N
        max_NE = max_N if len(extra_N)==0 else max(max_N, np.max(extra_N))
        if len(extra_N) > 1:
            en = np.array(extra_N)
            assert np.min(en[1:]-en[:-1]) > 0

        if max_NE > 6:
            assert max_NE <= 12
            self.max_N = 6
        else:
            self.max_N = max_NE

        tm = self.get_tradable_map(trade_day, mts_key=True, mts_symbols = [ mts_symbol_no_N ], add_prev_day=add_prev_day)
        assert len(tm.keys()) > 0, 'no contracts found on ' + trade_day + ', a holiday?'
        Ns = []
        Contracts = []
        for ms in tm.keys():
            if not include_spread and ('S' in tm[ms].keys()):
                continue
            symbol = tm[ms]['symbol']
            if symbol == mts_symbol_no_N :
                Ns.append(tm[ms]['N'])
                Contracts.append(tm[ms]['contract_month'])
        nix = np.argsort(Ns)
        Ns = list(np.array(Ns)[nix])
        Contracts = list(np.array(Contracts)[nix])

        if max_NE > 6:
            # get a day of 6 months ago and get a list of 6 contracts to append to
            day = trade_day
            year = int(day[:4])
            month = int(day[4:6])
            if month < 7:
                year -= 1
                month += 6
            else:
                month -= 6
            day = '%04d%02d'%(year,month)+'07' #a lucky day
            tdi = mts_util.TradingDayIterator(day)
            day = tdi.begin()

            # This is quite a hack - 
            # if the current day has just rolled, i.e. N6
            # is a new contract, make sure the dates half 
            # year earlier starts from 'N7'
            if np.min(Ns) == 0:
                day = tdi.next(10)

            Ns2 = []
            Contracts2 = []
            tm2 = self.get_tradable_map(day, mts_key=True, mts_symbols = [ mts_symbol_no_N ], add_prev_day=True)
            # generate all the contracts
            for ms in tm2.keys() :
                if not include_spread and ('S' in tm2[ms].keys()):
                    continue
                symbol = tm2[ms]["symbol"]
                if symbol == mts_symbol_no_N :
                    Ns2.append(tm2[ms]['N'])
                    Contracts2.append(tm2[ms]['contract_month'])
            nix = np.argsort(Ns2)
            Ns2 = list(np.array(Ns2)[nix])
            Contracts2 = list(np.array(Contracts2)[nix])
            
            # populate the Ns
            for con in Contracts2 :
                year = int(con[:4])+1
                con = '%04d'%(year)+con[-2:]
                if con > Contracts[-1]:
                    Ns.append(Ns[-1]+1)
                    Contracts.append(con)

        ix = np.searchsorted(np.array(Ns), max_N)
        assert Ns[ix] == max_N
        ret = Contracts[:ix+1]

        # make Ns (the contract in the future) to Ms (the months in the future)
        Ms = np.array(Contracts).astype(int)//100*12+(np.array(Contracts).astype(int)%100)
        Ms = Ms - Ms[0] + Ns[0]
        for en in extra_N:
            if en <= max_N:
                continue
            ix = np.clip(np.searchsorted(np.array(Ms), en),0,len(Ns)-1)
            if Ns[ix] == en:
                ret.append(Contracts[ix])

        self.max_N=max_N
        return ret

    def get_symbol_contract_from_tradable(self, tradable, trade_day = None, is_mts_symbol=False, add_prev_day=False) :
        """
        return mts_symbol, symbol and  contract_month
        """

        max_N0 = self.max_N
        max_N = max_N0
        # for MTS non-spread symbols, N can be entertained
        if is_mts_symbol and len(tradable.split('_')) == 2:
            sym = tradable.split('_')[0]
            N = tradable.split('_')[1]
            assert N[0] == 'N', 'unknown format of MTS symbol: ' + tradable
            n = int(N[1:])
            assert n >= 0 and n <= 12, 'N out of range: ' + tradable
            if n > self.max_N:
                if n <= 6:
                    max_N = 6
                else:
                    if trade_day is None :
                        trade_day = datetime.datetime.now().strftime('%Y%m%d')
                    con = self.get_contract_from_symbol(sym, trade_day, add_prev_day=add_prev_day, include_spread=False, extra_N=[n])
                    return tradable, sym, con[-1]

        self.max_N = max_N
        tinfo = self.get_tinfo(tradable, trade_day, is_mts_symbol=is_mts_symbol,add_prev_day=add_prev_day)
        self.max_N = max_N0
        return tinfo['mts_symbol'], tinfo['symbol'], tinfo['contract_month']

    def get_mts_symbol_from_tt_venue(self, tt_venue_arr, trade_day, tt_venue_map_cfg='config/venue_map.cfg', N=1) :
        ttvm = self._load_tt_venue_map(tt_venue_map_cfg)
        tm = self.get_tradable_map(trade_day, mts_key=True)
        ret = []
        for ms in tm.keys() :
            ttv = ttvm[tm[ms]["venue"]]
            if tt_venue_arr is None or ttv in tt_venue_arr :
                if ms[-1] != str(N) :
                    continue
                ret.append([tm[ms]["symbol"], ttv])
        return ret

    def get_mts_symbol_from_mts_venue(self, mts_venue_arr, trade_day) :
        tm = self.get_tradable_map(trade_day, mts_key=True)
        ret = []
        for ms in tm.keys() :
            venue = tm[ms]["venue"]
            if venue in mts_venue_arr :
                ret.append(ms)
        return ret

    def get_mts_symbol_from_field(self, trade_day, field_values, field_name) :
        tm = self.get_tradable_map(trade_day, mts_key=True)
        ret = []
        for ms in tm.keys() :
            if field_name not in tm[ms].keys() :
                continue
            val = tm[ms][field_name]
            if val in field_values: 
                ret.append(ms)
        return ret

    def getSubscriptionList(self, mts_cfg, trade_day) :
        # read "MTSVenue" and "MTSSymbol" from config and return a list of mts_symbol
        # as subscription list
        sub_venue0 = mts_cfg.listSubKeys('MTSVenue')
        sub_venue = []
        for v in sub_venue0 :
            if len(mts_cfg.getArr('MTSVenue.'+v)) > 0 :
                sub_venue.append(v)

        sub_symbol0 = mts_cfg.listSubKeys('MTSSymbol')
        sub_symbol = []
        for s in sub_symbol0 :
            if len(mts_cfg.getArr('MTSSymbol.'+s)) > 0 :
                sub_symbol.append(s)

        sub_symbol= set(self.get_mts_symbol_from_field(trade_day,  sub_symbol, 'symbol'))
        sub_symbol.update(self.get_mts_symbol_from_mts_venue(sub_venue, trade_day))
        return sub_symbol

    def _load_tt_venue_map(self, tt_venue_map_cfg) :
        vm = {}
        try :
            with open(tt_venue_map_cfg, "r") as f :
                while True :
                    l = f.readline()
                    if len(l) ==0 :
                        break
                    l = l.strip()
                    if l[0] == '#' :
                        continue
                    la = l.split('=')
                    if len(la) == 2 :
                        vm[la[0].strip()] = la[1].strip()
        except :
            traceback.print_exc()
            print ("failed to load tt venue map!")
        return vm

    def _get_assets_root(self) :
        if self.assets_root is None :
            try :
                self.assets_root = gen_asset_root(self.xml_path, self.assets_xml)
            except :
                traceback.print_exc()
                raise RuntimeError("Cannot get symbol map from " + self.xml_path + ", " + self.assets_xml)


class FXMap :
    def __init__(self, fx_filename='/home/mts/run/config/symbol/fx_1700.txt'):
        """
        days, utcs: array of yyyymmdd (utc) of the days of rates
        syms:       array of symbols of columns of rates
        rates: shape [ndays,nfx] rates x, where x is number of USD needed to for 1 FX
        """
        self.fn = fx_filename
        self.syms, self.days, self.utcs, self.rates = self._load()

    def _load(self):
        """load historical FX from file into fxdict, i.e. fx_1500.txt
        File format:
            Header:
            trade_date,AUD,BRL,CAD,CHF,CLP,CNH,CNY,COP,CZK,DKK,EUR,GBP,HKD,HUF,IDR,ILS,INR,ISK,JPY,KRW,MXN,MYR,NOK,NZD,PHP,PLN,RUB,SAR,SEK,SGD,THB,TRY,TWD,ZAR
            Body:
            2007-02-28,0.7879,,,,,,,,,,1.3231,1.9636,,,,,,,,,,,,0.7016,,,,,,,,,,
        """
        syms = subprocess.check_output(['head','-1',self.fn]).decode().replace('\r','').replace('\n','').split(',')[1:]
        rates = np.genfromtxt(self.fn, delimiter=',',skip_header=1,dtype='str')
        days = []
        utcs = []
        for d in rates[:,0]:
            dt = datetime.datetime.strptime(d, '%Y-%m-%d')
            days.append(dt.strftime('%Y%m%d'))
            utcs.append(int(dt.strftime('%s'))+17*3600)  # utc at 17:00 of the day

        rates[np.nonzero(rates=='')]='1'
        rates=rates[:,1:].astype(float)
        return np.array(syms), np.array(days), np.array(utcs), rates

    def get(self, sym, day, use_prev_day=True, default=None):
        """
        sym: str, i.e. 'EUR', 'GBP'
        day: yyyymmdd
        use_prev_day: use previous day if day is not found
        default: if sym not found, throw if None, otherwise use the default
        """
        six = np.nonzero(self.syms==sym)[0]
        if len(six) == 0:
            if default is not None:
                return default
            raise RuntimeError('unknown FX: %s'%(sym))
        six=six[0]

        dix = max(np.searchsorted(self.days, str(int(day)+1))-1,0)
        if self.days[dix] != day:
            if not use_prev_day:
                raise RuntimeError('%s not found on %s'%(sym,day))
        return self.rates[dix,six]

    def get_by_utc(self, sym, utc, use_prev_day=True, default=None):
        six = np.nonzero(self.syms==sym)[0]
        if len(six) == 0:
            if default is not None:
                return default
            raise RuntimeError('unknown FX: %s'%(sym))
        six=six[0]

        dix = max(np.searchsorted(self.utcs, utc+1)-1,0)
        if not use_prev_day:
            utc0=self.utcs[dix]
            ymd0 = datetime.datetime.fromtimestamp(utc0).strftime('%Y%m%d')
            ymd =  datetime.datetime.fromtimestamp(utc).strftime('%Y%m%d')
            if ymd != ymd0:
                raise RuntimeError('%s not found on %s'%(sym,ymd))
        return self.rates[dix,six]



