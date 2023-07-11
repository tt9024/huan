import numpy as np
import pandas
import datetime
import repo_util

def test_normalize():
    cols = ['utc', 'open', 'high', 'low', 'close', 'vol', 'lpx', 'ltm', 'vbs', 'bsz','bqd','opt_v2']

    # missing
    bar0 = [ [100, 10,  10.1, 9.9,  9.9, 2,  9.9,  100, -2, 5.5, -3, 2],\
             [102, 9.9, 10.0, 10.0, 10.0,4,  10.0, 101, 2,  5.2, 1,  2],\
             [104, 9.8, 9.9,  9.8,  9.7, 11, 9.75,  103, -10,2.3, -2, -5]]

    dbar  = repo_util.daily1s(np.array(bar0), 101, 105, cols=cols)
    dbar0 = [[102,    9.9,  10,   9.9,  10,    4,   10,  101,    2,    5.2,   1,    2, ],\
             [103,    10,   10,   9.8,  9.8,   0,   10,  101,    0,    5.2,   0,    0, ],\
             [104,    9.8,  9.9,  9.7,  9.7,  11,   9.75, 103,  -10,   2.3,  -2,   -5, ],\
             [105,    9.7,  9.7,  9.7,  9.7,   0,   9.75, 103,    0,   2.3,   0,    0, ]]
    assert np.max(np.abs(dbar-np.array(dbar0))) == 0

    # first bar missing, use prev
    dbar  = repo_util.daily1s(np.array(bar0), 100, 105, cols=cols)
    dbarr = [[101,    9.9,  9.9,  9.9,  9.9,   0,   9.9, 100,    0,    5.5,   0,    0, ],\
             [102,    9.9,  10,   9.9,  10,    4,   10,  101,    2,    5.2,   1,    2, ],\
             [103,    10,   10,   9.8,  9.8,   0,   10,  101,    0,    5.2,   0,    0, ],\
             [104,    9.8,  9.9,  9.7,  9.7,  11,   9.75, 103,  -10,   2.3,  -2,   -5, ],\
             [105,    9.7,  9.7,  9.7,  9.7,   0,   9.75, 103,    0,   2.3,   0,    0, ]]
    assert np.max(np.abs(dbar-np.array(dbarr))) == 0

    # no first
    try:
        dbar  = repo_util.daily1s(np.array(bar0), 98, 103, cols=cols)
        assert False, 'should fail on first bar missing and not backward fill'
    except:
        pass
    dbar  = repo_util.daily1s(np.array(bar0), 99, 103, cols=cols)
    dbarr = [[100,    10,   10.1, 9.9,  9.9,   2,   9.9, 100,    -2,   5.5,   -3,   2 ],\
             [101,    9.9,  9.9,  9.9,  9.9,   0,   9.9, 100,    0,    5.5,   0,    0 ],\
             [102,    9.9,  10,   9.9,  10,    4,   10,  101,    2,    5.2,   1,    2 ],\
             [103,    10,   10,   10,   10,    0,   10,  101,    0,    5.2,   0,    0 ]]
    assert np.max(np.abs(dbar-np.array(dbarr))) == 0

    # bad value
    bar1 = np.array(bar0.copy())
    bar1[2,1:4] = 0 # fill by close
    dbar  = repo_util.daily1s(np.array(bar1), 99, 105, cols=cols)
    dbarr = [[100,    10,   10.1, 9.9,  9.9,   2,   9.9, 100,    -2,   5.5,   -3,   2 ],\
             [101,    9.9,  9.9,  9.9,  9.9,   0,   9.9, 100,    0,    5.5,   0,    0 ],\
             [102,    9.9,  10,   9.9,  10,    4,   10,  101,    2,    5.2,   1,    2 ],\
             [103,    10,   10,   10,   10,    0,   10,  101,    0,    5.2,   0,    0 ],\
             [104,    10,   10,   9.7,  9.7,  11,   9.75, 103,  -10,   2.3,  -2,   -5 ],\
             [105,    9.7,  9.7,  9.7,  9.7,   0,   9.75, 103,    0,   2.3,   0,    0 ]]
    assert np.max(np.abs(dbar-np.array(dbarr))) == 0

    bar1 = np.array(bar0.copy())
    bar1[1,2:5] = 0 # fill by open of the bar
    dbar  = repo_util.daily1s(np.array(bar1), 99, 105, cols=cols)
    dbarr = [[100,    10,   10.1, 9.9,  9.9,   2,   9.9, 100,    -2,   5.5,   -3,   2 ],\
             [101,    9.9,  9.9,  9.9,  9.9,   0,   9.9, 100,    0,    5.5,   0,    0 ],\
             [102,    9.9,  9.9,  9.9,  9.9,   4,   10,  101,    2,    5.2,   1,    2 ],\
             [103,    9.9,  9.9,  9.8,  9.8,   0,   10,  101,    0,    5.2,   0,    0 ],\
             [104,    9.8,  9.9,  9.7,  9.7,  11,   9.75, 103,  -10,   2.3,  -2,   -5, ],\
             [105,    9.7,  9.7,  9.7,  9.7,   0,   9.75, 103,    0,   2.3,   0,    0, ]]
    assert np.max(np.abs(dbar-np.array(dbarr))) == 0


    bar0 = [[99, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0]] + bar0
    dbar  = repo_util.daily1s(np.array(bar0), 99, 103, cols=cols)
    dbarr = [[100,   10,   10.1, 9.9,  9.9,   2,   9.9,  100,  -2,    5.5,  -3,    2, ],\
	 [101,   9.9,  9.9,  9.9,  9.9,   0,   9.9,  100,   0,    5.5,   0,    0, ],\
	 [102,   9.9,  10,   9.9,  10,    4,   10,   101,   2,    5.2,   1,    2, ],\
	 [103,   10,   10,   10,   10,    0,   10,   101,   0,    5.2,   0,    0, ]]
    assert np.max(np.abs(dbar-np.array(dbarr))) == 0

    dbar  = repo_util.daily1s(np.array(bar0), 98, 103, cols=cols, backward_fill=True)
    dbarr = [[ 99,  10,  10,    10,   10,   0,  9.9, 100,   0,   5.5,   0,   0,],\
	     [100,  10,  10.1,  9.9,  9.9,  2,  9.9, 100,  -2,   5.5,  -3,   2,],\
	     [101,  9.9, 9.9,   9.9,  9.9,  0,  9.9, 100,   0,   5.5,   0,   0,],\
	     [102,  9.9, 10,    9.9,  10,   4,  10,  101,   2,   5.2,   1,   2,],\
	     [103,  10,  10,    10,   10,   0,  10,  101,   0,   5.2,   0,   0,]]
    assert np.max(np.abs(dbar-np.array(dbarr))) == 0, print(dbar-np.array(dbarr))

    bar0 = [[99, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0]] + bar0[:2] + bar0
    dbar  = repo_util.daily1s(np.array(bar0), 98, 103, cols=cols, backward_fill=True, allow_non_increasing=True)
    assert np.max(np.abs(dbar-np.array(dbarr))) == 0
    
def test_normalize_ohlc():
    bar = np.array([[100, 10,  0,    0,   9.9,  2,  9.9,  100, -2, 5.5,  -3,  2],\
                    [102, 9.9, 0,    0,   10.0, 4,  10.0, 101, 2,  5.2,  1,   2],\
                    [104, 9.8, 9.9,  9.8, 9.7,  11, 9.75,  103, -10,2.3, -2, -5]])

    cols0 = ['utc', 'open', 'high', 'close']
    bar0 = bar[:,np.array([0,1,2,4]).astype(int)]
    dbar  = repo_util.daily1s(np.array(bar0), 98, 105, cols=cols0, backward_fill=True)
    dbarr = np.array([[99, 10,  10,  10,],\
		     [100,  10,  10,  9.9],\
		     [101,  9.9, 9.9, 9.9],\
		     [102,  9.9, 10,  10,],\
		     [103,  10,  10,  9.8],\
		     [104,  9.8, 9.9, 9.7],\
		     [105,  9.7, 9.7, 9.7]])
    assert np.max(np.abs(dbar-dbarr)) == 0

    cols0 = ['utc', 'high', 'close']
    bar0 = bar[:,np.array([0,2,4]).astype(int)]
    dbar  = repo_util.daily1s(np.array(bar0), 98, 105, cols=cols0, backward_fill=True)
    dbarr = np.array([[99,   9.9,  9.9],\
		     [100,   9.9,  9.9],\
		     [101,   9.9,  9.9],\
		     [102,   10,   10,],\
		     [103,   10,   10,],\
		     [104,   9.9,  9.7],\
		     [105,   9.7,  9.7]])
    assert np.max(np.abs(dbar-dbarr)) == 0
    


    cols0 = ['utc', 'open', 'low']
    bar0 = bar[:,np.array([0,1,3]).astype(int)]
    dbar  = repo_util.daily1s(np.array(bar0), 98, 105, cols=cols0, backward_fill=True)
    dbarr = np.array([[99,  10,  10,],\
		     [100,  10,  10,],\
		     [101,  10,  10,],\
		     [102,  9.9, 9.9],\
		     [103,  9.9, 9.9],\
		     [104,  9.8, 9.8],\
		     [105,  9.8, 9.8]])
    assert np.max(np.abs(dbar-dbarr)) == 0

    cols0 = ['utc', 'high', 'lpx']
    bar0 = bar[:,np.array([0,2,6]).astype(int)]
    dbar  = repo_util.daily1s(np.array(bar0), 98, 105, cols=cols0, backward_fill=True)
    dbarr = np.array([[99,   9.9, 9.9 ],\
		     [100,   9.9, 9.9 ],\
		     [101,   9.9, 9.9 ],\
		     [102,   10,  10, ],\
		     [103,   10,  10, ],\
		     [104,   9.9, 9.75],\
		     [105,   9.75, 9.75]])
    assert np.max(np.abs(dbar-dbarr)) == 0

