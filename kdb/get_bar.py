import numpy as np
import sys
import os
import datetime
import traceback
import pdb
import l1

if __name__ == '__main__' :
    import sys
    symbol_list= sys.argv[1:-2]
    sday=sys.argv[-2]
    eday=sys.argv[-1]
    l1.get_future_bar_fix(symbol_list, sday, eday)
