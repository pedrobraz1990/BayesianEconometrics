import numpy as np
cimport numpy as np
cimport cython
# from numpy.linalg import inv
# from numpy.linalg import det
# import pandas as pd
# import datetime as dt
from libcpp cimport bool

DTYPE = np.float64
DTYPEI = np.int

ctypedef np.float64_t DTYPE_t
ctypedef np.int_t DTYPEI_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.nonecheck(False)
@cython.cdivision(True)

# cdef float KalmanFilter(np.ndarray[DTYPE_t, ndim = 2] y,
def KalmanFilter(np.ndarray[DTYPE_t, ndim = 2] y,
                DTYPEI_t nStates,
                np.ndarray[DTYPE_t, ndim = 2] Z,
                np.ndarray[DTYPE_t, ndim = 2] H,
                np.ndarray[DTYPE_t, ndim = 2] T,
                np.ndarray[DTYPE_t, ndim = 2] Q,
                np.ndarray[DTYPE_t, ndim = 1] a1,
                np.ndarray[DTYPE_t, ndim = 2] P1,
                np.ndarray[DTYPE_t, ndim = 2] R):
# def KalmanFilter(y, nStates, Z, H, T, Q, a1, P1, R):


    cdef DTYPEI_t t, dim
    cdef DTYPE_t ll = 0

    cdef DTYPEI_t p = y.shape[1]
    cdef DTYPEI_t n = y.shape[0]
    cdef DTYPEI_t m = nStates

    cdef np.ndarray[DTYPE_t, ndim = 2] yhat = np.empty((n, p), dtype=DTYPE)
    cdef np.ndarray[DTYPEI_t, ndim = 2] ind = np.zeros((n, p), dtype=DTYPEI)
    cdef np.ndarray[DTYPE_t, ndim = 2] a = np.empty((m, n), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 3] P = np.empty((m, m, n), dtype=DTYPE)
    a[:, 0] = a1
    P[:, :, 0] = P1
    cdef np.ndarray[DTYPE_t, ndim = 2] vt = np.zeros((n, p), dtype=DTYPE)
    # cdef np.ndarray[DTYPEI_t, ndim = 2] inds = np.ones((n, p), dtype=np.intc)
    cdef np.ndarray inds = np.ones((n, p), dtype=np.bool)
    cdef np.ndarray[DTYPE_t, ndim = 3] Ft = np.empty((p, p, n), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 2] ZT = Z.T  # To avoid transposing it several times
    cdef np.ndarray[DTYPE_t, ndim = 2] TT = T.T  # To avoid transposing it several times
    cdef np.ndarray[DTYPE_t, ndim = 2] RT = R.T

    cdef np.ndarray[DTYPEI_t, ndim = 1] dims = np.ones((n), dtype=DTYPEI)
    dims = dims * p

    # # y should be (n x p)
    # p = y.shape[1]
    # n = y.shape[0]
    # m = nStates

    # yhat = np.empty((n,p))
    # a = np.empty((m,n))
    # P = np.empty((m,m,n))
    # a[:,0] = a1
    # P[:,:,0] = P1
    # vt = np.empty((n,p))
    # inds = np.ones((n,p), dtype=bool)
    # Ft = np.empty((p,p,n))
    # ZT = Z.T  # To avoid transposing it several times
    # TT = T.T  # To avoid transposing it several times
    # R = np.array(R)
    # RT = R.T
    # ind = np.zeros(y.shape[0])
    # dims = np.ones((n),dtype=np.int)
    # dims = dims * p

    ind[np.isnan(y).any(axis=1)] = 1  # Some NaNs
    ind[np.isnan(y).all(axis=1)] = 2  # All NaNs


    for t in range(0, n - 1):

        if ind[t][0] == 0:
        # if True:

            # (p) =  (p) - (p x m)(m x 1)

            vt[t,:] = y[t,:] - np.dot(Z, a[:,t])

            Ft[:,:,t] = (Z.dot(P[:,:,t]).dot(ZT) + H)

            Finv = np.linalg.inv(Ft[:,:,t])

            a[:,t] = a[:,t] + P[:,:,t].dot(ZT).dot(Finv).dot(vt[t,:])

            P[:,:,t] = P[:,:,t] - P[:,:,t].dot(ZT).dot(Finv).dot(Z).dot(P[:,:,t])

            a[:,t+1] = T.dot(a[:,t])

            P[:,:,t+1] = T.dot(P[:,:,t]).dot(TT) + R.dot(Q).dot(RT)

            yhat[t,:] = Z.dot(a[:,t])

        elif ind[t][0] == 2:  # In case the line is all nans

            vt[t, :] = np.zeros((p))

            Ft[:, :, t] = (Z.dot(P[:, :, t]).dot(ZT) + H)

            a[:, t + 1] = T.dot(a[:, t])

            P[:, :, t + 1] = T.dot(P[:, :, t]).dot(TT) + R.dot(Q).dot(RT)

            yhat[t, :] = Z.dot(a[:, t])

        else:
            # First use an index for nulls
            ind2 = ~np.isnan(y[t]).ravel()
            inds[t,:] = ind2
            yst = y[t,ind2]
            Zst = Z[ind2, :]
            ZstT = Zst.T
            dim = ind2.sum(dtype=np.int)
            dims[t] = dim
            select = np.diag(ind2)
            select = select[(select == True).any(axis=1)].astype(int)

            Hst = select.dot(H).dot(select.T)

            vt[t,ind2] = yst - np.dot(Zst, a[:,t])

            Ft[:dim,:dim,t] = Zst.dot(P[:, :,t]).dot(ZstT) + Hst

            Finv = np.linalg.inv(Ft[:dim, :dim, t])

            a[:,t] = a[:,t] + P[:, :,t].dot(ZstT).dot(Finv).dot(vt[t,ind2])

            P[:, :,t] = P[:, :,t] - P[:, :,t].dot(ZstT).dot(Finv).dot(Zst).dot(P[:, :,t])

            a[:, t + 1] = T.dot(a[:,t])

            P[:, :, t + 1] = T.dot(P[:,:,t]).dot(TT) + R.dot(Q).dot(RT)

            yhat[t, ind2] = Zst.dot(a[:, t])
            yhat[t ,~ind2] = Z.dot(a[:,t])[~ind2]


    # a = pd.DataFrame(np.concatenate(a, axis=1)).T
    # yhat = pd.DataFrame(np.concatenate(yhat, axis=1)).T
    # y = pd.DataFrame(y)


    # ll = 0.0
    # inds = inds.astype(np.bool)
    for t in range(0, n - 1):
        if ind[t][0] < 2:
            # print(vt[t,inds[t,:]].T.dot(np.linalg.inv(Ft[:dims[t],:dims[t],t])).dot(vt[t,inds[t,:]]))
            # print(vt[t,inds[t,:]].T.shape)
            # print(Ft[:dims[t],:dims[t],t].shape)
            # print(inds[t,:])

            ll += np.log(np.linalg.det(Ft[:dims[t],:dims[t],t])) + vt[t,inds[t,:]].T.dot(np.linalg.inv(Ft[:dims[t],:dims[t],t])).dot(vt[t,inds[t,:]])
    ll = - n * p * 0.5 * np.log(2 * np.pi) - 0.5 * ll
    # return {'ll' : np.exp(ll), 'yhat' : yhat, 'y' : y }
    return ll
