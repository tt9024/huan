import numpy as np

def f1 (d0, d1, d2):
    n = len(d0)
    # init E
    E = np.ones((3,3))
    E[np.arange(3),np.arange(3)]=d0[:3]
    E[np.arange(2)+1,np.arange(2)] = d1[:2]
    E[np.arange(2),np.arange(2)+1] = d1[:2]
    E[2,0] = d2[0]
    E[0,2] = d2[0]

    dd = [np.array([d0[0],d1[0]])]

    ff = []
    # need big L
    #L = np.eye(n)

    for i in np.arange(n-3):
        f = E[0,1:]/E[0,0]
        E[0,:2] = E[1,1:]-f*E[0,1]
        E[1,:2] = E[2,1:]-f*E[0,2]
        E[2,:] = np.array([d2[i+1],d1[i+2],d0[i+3]])
        E[:2,2]= np.array([d2[i+1],d1[i+2]])
        dd.append(E[0,:2].copy())
        ff.append(f)

        #L[i+1,:i+1]-=L[i,:i+1]*f[0]
        #L[i+2,:i+1]-=L[i,:i+1]*f[1]

    # the last one
    i = n-3
    f = E[0,1:]/E[0,0]
    E[0,:2] = E[1,1:]-f*E[0,1]
    E[1,:2] = E[2,1:]-f*E[0,2]
    dd.append(E[0,:2].copy())
    ff.append(f)

    #L[i+1,:i+1]-=L[i,:i+1]*f[0]
    #L[i+2,:i+1]-=L[i,:i+1]*f[1]

    i = n-2
    f = E[0,1:]/E[0,0]
    E[0,:2] = E[1,1:]-f*E[0,1]
    dd.append(E[0,:2].copy())
    ff.append(f)

    # done with dd
    dd=np.array(dd)

    # create a big eye
    L = np.eye(n)
    for i, f in enumerate(ff[:-1]):
        L[i+1,:i+1]-=L[i,:i+1]*f[0]
        L[i+2,:i+1]-=L[i,:i+1]*f[1]

    i+=1
    f = ff[-1]
    L[i+1,:i+1]-=L[i,:i+1]*f[0]

    return dd[:,0], dd[:-1,1], d2, L

def f2 (d0, d1, d2):
    # diag the lower banded
    # note d0 cannot have very small numbers, assuming
    # the matrix is invertable

    n = len(d0)
    dd1 = d1/d0[:-1]
    dd2 = d2/d0[:-2]

    # create a big eye
    L = np.eye(n)
    for i, (f0, f1) in enumerate(zip(dd1[:-1], dd2)):
        L[i+1,:i+1]-=L[i,:i+1]*f0
        L[i+2,:i+1]-=L[i,:i+1]*f1
    i+=1
    L[i+1,:i+1]-=L[i,:i+1]*dd1[-1]
    return L

def ti(d0, d1, d2):
    n=len(d0)
    dd0, dd1, d2, L1 = f1(d0,d1,d2)
    L2 = f2(dd0,dd1,d2)
    B = np.zeros((n,n))
    B[np.arange(n),np.arange(n)]=1.0/dd0

    print(L1)
    print(L2)
    print(dd0)

    return np.trace(np.dot(np.dot(L1.T, B),L2))

def test_f1(d0,d1,d2):
    dd0, dd1, dd2, L = f1(d0,d1,d2)
    n = len(dd0)

    Z = np.eye(n)*d0
    Z[np.arange(n-1)+1,np.arange(n-1)]=d1
    Z[np.arange(n-1),np.arange(n-1)+1]=d1
    Z[np.arange(n-2)+2,np.arange(n-2)]=d2
    Z[np.arange(n-2),np.arange(n-2)+2]=d2

    B = np.eye(n)*dd0
    B[np.arange(n-1),np.arange(n-1)+1] = dd1
    B[np.arange(n-2),np.arange(n-2)+2] = dd2
    assert np.max(np.abs(np.dot(L,Z)-B)) < 1e-8

def test_f2(d0,d1,d2):
    n = len(d0)
    L = f2(d0,d1,d2)

    Z = np.eye(n)
    Z[np.arange(n),np.arange(n)] = d0
    Z[1+np.arange(n-1),np.arange(n-1)] = d1
    Z[2+np.arange(n-2),np.arange(n-2)] = d2

    B1 = np.dot(L,Z)
    B = np.eye(n)
    B[np.arange(n),np.arange(n)] = d0
    assert np.max(np.abs(B1-B)) < 1e-8

def test_ti(d0,d1,d2):
    n=len(d0)
    Z = np.eye(n)*d0
    Z[np.arange(n-1)+1,np.arange(n-1)]=d1
    Z[np.arange(n-1),np.arange(n-1)+1]=d1
    Z[np.arange(n-2)+2,np.arange(n-2)]=d2
    Z[np.arange(n-2),np.arange(n-2)+2]=d2

    assert abs(np.trace( np.linalg.inv(Z) )-ti(d0,d1,d2))<1e-8

    return Z

