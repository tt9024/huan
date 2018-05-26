"Contains functions for the fast solving of block banded problems \n\
Solves for 'y' in 'M y = x' where 'M' is the symetrix block tridiagnoal matrix\n\
possibly with corners "

import numpy as np
import scipy.linalg as lg
from scipy.sparse.linalg import aslinearoperator as L0

def py_sym_block_tri_solve(d0,d1,x):
    """y=py_sym_block_tri_solve(d0,d1,x)
       python code for solving for 'y' in 'M y = x'
       where 'M' is the symmetric block tridia matrix
       Inputs :
       'd0' = 3d numpy array for the main diagonal with d0[0,:,:]
       being the [0,0] block
       'd1' = length 'n' -1 list of 2d numpy arrays for the
       sub and super diagonal with d1[0] being the [1,0]
       and [0,1] blocks
       'x' = numpy vector (or array)
       Outputs:
       'y' = solution of equation 'M y = x' where
       'M' is the symmetric block tridiagonal matrix
    """
    pass
