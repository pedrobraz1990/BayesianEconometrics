import numpy as np
# cimport numpy as np
from numpy.linalg import inv
from numpy.linalg import det
import pandas as pd
import datetime as dt

DTYPE = np.float64

# def KalmanFilter(np.ndarray[double, ndim = 2] y,
#                 int nStates,
#                 np.ndarray[double, ndim = 2] Z,
#                 np.ndarray[double, ndim = 2] H,
#                 np.ndarray[double, ndim = 2] T,
#                 np.ndarray[double, ndim = 2] Q,
#                 np.ndarray[double, ndim = 1] a1,
#                 np.ndarray[double, ndim = 2] P1,
#                 np.ndarray[double, ndim = 2] R):
def KalmanFilter(y, nStates, Z, H, T, Q, a1, P1, R):


    # cdef int t, dim
    # cdef double ll = 0
    #
    # cdef int p = y.shape[1]
    # cdef int n = y.shape[0]
    # cdef int m = nStates
    #
    # cdef np.ndarray yhat = np.empty((n, p), dtype=DTYPE)
    # cdef np.ndarray ind = np.zeros((n, p), dtype=DTYPE)
    # cdef np.ndarray a = np.empty((m, n), dtype=DTYPE)
    # cdef np.ndarray P = np.empty((m, m, n), dtype=DTYPE)
    # a[:, 0] = a1
    # P[:, :, 0] = P1
    # cdef np.ndarray vt = np.zeros((n, p), dtype=DTYPE)
    # cdef np.ndarray inds = np.ones((n, p), dtype=np.bool)
    # cdef np.ndarray Ft = np.empty((p, p, n), dtype=DTYPE)
    # cdef np.ndarray ZT = Z.T  # To avoid transposing it several times
    # cdef np.ndarray TT = T.T  # To avoid transposing it several times
    # cdef np.ndarray RT = R.T
    #
    # cdef np.ndarray dims = np.ones((n), dtype=np.int)
    # dims = dims * m

    # y should be (n x p)
    p = y.shape[1]
    n = y.shape[0]
    m = nStates

    yhat = np.empty((n,p))
    a = np.empty((m,n))
    P = np.empty((m,m,n))
    a[:,0] = a1
    P[:,:,0] = P1
    vt = np.empty((n,p))
    inds = np.ones((n,p), dtype=bool)
    Ft = np.empty((p,p,n))
    ZT = Z.T  # To avoid transposing it several times
    TT = T.T  # To avoid transposing it several times
    R = np.array(R)
    RT = R.T
    ind = np.zeros(y.shape[0])
    dims = np.ones((n),dtype=np.int)
    dims = dims * p

    ind[np.isnan(y).any(axis=1)] = 1  # Some NaNs
    ind[np.isnan(y).all(axis=1)] = 2  # All NaNs


    for t in range(0, n - 1):

        if ind[t] == 0:
        # if True:

            # (p) =  (p) - (p x m)(m x 1)

            vt[t,:] = y[t,:] - np.dot(Z, a[:,t])

            Ft[:,:,t] = (Z.dot(P[:,:,t]).dot(ZT) + H)

            Finv = inv(Ft[:,:,t])

            a[:,t] = a[:,t] + P[:,:,t].dot(ZT).dot(Finv).dot(vt[t,:])

            P[:,:,t] = P[:,:,t] - P[:,:,t].dot(ZT).dot(Finv).dot(Z).dot(P[:,:,t])

            a[:,t+1] = T.dot(a[:,t])

            P[:,:,t+1] = T.dot(P[:,:,t]).dot(TT) + R.dot(Q).dot(RT)

            yhat[t,:] = Z.dot(a[:,t])

        elif ind[t] == 2:  # In case the line is all nans

            vt[t, :] = np.zeros((p))

            Ft[:, :, t] = (Z.dot(P[:, :, t]).dot(ZT) + H)

            a[:, t + 1] = T.dot(a[:, t])

            P[:, :, t + 1] = T.dot(P[:, :, t]).dot(TT) + R.dot(Q).dot(RT)

            yhat[t, :] = Z.dot(a[:, t])

        else:
            # First use an index for nulls
            ind2 = ~np.isnan(y[t]).ravel()
            inds[t,:] = ind2
            yst = y[t,:][ind2]
            Zst = Z[ind2, :]
            ZstT = Zst.T
            dim = ind2.sum(dtype=np.int)
            dims[t] = dim
            select = np.diag(ind2)
            select = select[(select == True).any(axis=1)].astype(int)

            Hst = select.dot(H).dot(select.T)

            vt[t,ind2] = yst - np.dot(Zst, a[:,t])

            Ft[:dim,:dim,t] = Zst.dot(P[:, :,t]).dot(ZstT) + Hst

            Finv = inv(Ft[:dim, :dim, t])

            a[:,t] = a[:,t] + P[:, :,t].dot(ZstT).dot(Finv).dot(vt[t,ind2])

            P[:, :,t] = P[:, :,t] - P[:, :,t].dot(ZstT).dot(Finv).dot(Zst).dot(P[:, :,t])

            a[:, t + 1] = T.dot(a[:,t])

            P[:, :, t + 1] = T.dot(P[:,:,t]).dot(TT) + R.dot(Q).dot(RT)

            yhat[t, ind2] = Zst.dot(a[:, t])
            yhat[t ,~ind2] = Z.dot(a[:,t])[~ind2]


    # a = pd.DataFrame(np.concatenate(a, axis=1)).T
    # yhat = pd.DataFrame(np.concatenate(yhat, axis=1)).T
    # y = pd.DataFrame(y)


    ll = 0.0
    for t in range(0, n - 1):
        # print("ind: {ind!s}".format(ind=ind[t]))
        # print("dim: {ind!s}".format(ind=dims[t]))
        if ind[t] < 2:
            ll += np.log(det(Ft[:dims[t],:dims[t],t])) + vt[t,inds[t,:]].T.dot(inv(Ft[:dims[t],:dims[t],t])).dot(vt[t,inds[t,:]])
    ll = - n * p * 0.5 * np.log(2 * np.pi) - 0.5 * ll
    return {'ll' : np.exp(ll), 'yhat' : yhat, 'y' : y }

