import numpy as np
import sys
import os
import datetime
import traceback
import pdb
import glob
import l1
import repo_dbar as repo

#############
# This module deals with the 1second bars collected at run time
# with the columes as
# UTC         bs    bp         as            ap  bv  sv  utc_at_collect   qbc qac bc sc ism_avg
# --------------------------------------------------------------------------------------------------
# 1535425169, 5, 2901.5000000, 2901.7500000, 135, 5, 17, 1535425169000056, 1, 2, 1, 2, 2901.5062609
# ...
# Where
# UTC is the bar ending time
# QBC is best bid change count
# QAC is best ask change count
# bc  is buy counts
# sc  is sell counts
#
# Parser will get from the file in 
# bar/NYM_CL_B1S.csv
# NOTE 1: utc offset:
# From 201805301800 to 201806261700, utc + 1 matches with history
# From 201806261800 to 201808171700, utc + 2 matches with history
# Good afterwards
#
# NOTE 2:
# Extended columns starts from 20180715-20:39:55, but may have problem
# for first few days
#
# NOTE 3:
# Next contract bar starts from 20180802-18:12:30
#
# NOTE 4:
# Be prepared for any data losses and errors!
# zero prices, zero sizes


def l1_bar(symbol, bar_path) :

    b = np.genfromtxt(bar_path, delimiter=',', use_cols=[0,1,2,3,4,5,6])
    # I need to get the row idx for each day for the columes of vbs and ism
    # which one is better?
    # I could use hist's trade for model, and l1/tick for execution
    pass

