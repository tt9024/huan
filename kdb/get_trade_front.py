import numpy as np
import sys
import os
import datetime
import traceback
import pdb
import l1

def get_future_trade_front(symbol_list, start_date, end_date, kdb_util='bin/get_trade', mock_run=False) :
    for symbol in symbol_list :
        bar_dir = symbol
        os.system(' mkdir -p ' + bar_dir)
        ti = l1.TradingDayIterator(start_date)
        day=ti.yyyymmdd()
        while day <= end_date :
            fc=l1.FC(symbol, day)
            # for each day, get trades for FC, FC+, FC/FC+, FC+/FC++
            #fc_next, roll_day=FC_next(symbol, day)
            #fc_next_next, roll_day=FC_next(symbol, roll_day)

            for c in [fc ] :
                fn=bar_dir+'/'+c+'_trd_'+day+'.csv'
                print 'checking ', c, fn
                # check if the file exists and the size is small
                if l1.get_file_size(fn) < 10000 and l1.get_file_size(fn+'.gz') < 10000 :
                    os.system('rm -f ' + fn + ' > /dev/null 2>&1')
                    os.system('rm -f ' + fn + '.gz' + ' > /dev/null 2>&1')
                    cmdline=kdb_util + ' ' + c + ' ' + day + ' > ' + fn
                    print 'running ', cmdline
                    if not mock_run :
                        os.system( cmdline )
                        os.system( 'gzip ' + fn )
                        os.system( 'sleep 5' )
            ti.next()
            day=ti.yyyymmdd()

if __name__ == '__main__' :
    import sys
    symbol_list= sys.argv[1:-2]
    sday=sys.argv[-2]
    eday=sys.argv[-1]
    get_future_trade_front(symbol_list, sday, eday)

