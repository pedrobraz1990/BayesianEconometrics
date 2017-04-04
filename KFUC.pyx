import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
import pandas as pd
import datetime as dt
cimport numpy as np


def KFUC(
        np.ndarray[double, ndim = 2] y,
        np.ndarray[double, ndim = 2] Z,
        np.ndarray[double, ndim = 2] H,
        np.ndarray[double, ndim = 2] T,
        np.ndarray[double, ndim = 2] Q,
        np.ndarray[double, ndim = 2] R,
        np.ndarray[double, ndim = 2] a1,
        np.ndarray[double, ndim = 2] P1,
        int nStates,
        likelihood = True,
    ):

    cdef int i, t, m, p, n
    cdef np.ndarray[double, ndim = 2] ZT, TT, RT, yhat, v, F
    cdef np.ndarray[double, ndim = 3] a, K
    cdef np.ndarray[double, ndim = 4] P

    n = y.shape[0]
    p = y.shape[1]
    m = nStates

    yhat = np.empty((n, p))
    a = np.empty((n + 1, p + 1, m))
    a[0, 0, :] = np.array(a1.astype(float)).ravel()
    P = np.empty((n + 1, p + 1, m, m))
    P[0, 0, :, :] = np.array(P1.astype(float))
    v = np.empty((n, p))
    F = np.empty((n, p))
    K = np.empty((n, p, m))
    ZT = Z.T
    TT = T.T
    RT = R.T

    for t in range(0, n):
        ind = ~np.isnan(y[t,:])
        for i in range(0,p): # later on change to Pt
            if ind[i]:
                v[t, i] = y[t, i] - Z[i, :].reshape((1,m)).dot(a[t, i, :].T) #a should be mx1
                F[t, i] = Z[i, :].reshape((1,m)).dot(P[t, i, :, :]).dot(Z[i, :]) + H[i, i]
                K[t, i, :] = P[t, i, :, :].dot(Z[i, :]) * F[t, i] ** (-1)
                a[t, i+1, :] = a[t, i, :] + K[t, i, :] * v[t, i]
                P[t, i+1, :, :] = P[t, i, :, :] - (K[t, i, :] * F[t, i]).reshape((m,1)).dot(K[t, i].reshape((1,m)))
            else:
                # Setting all Z's to zeros
                v[t, i] = 0
                F[t, i] = H[
                    i, i]
                K[t, i, :] = np.zeros(K[t, i, :].shape)
                # a[t, i + 1, :] = a[t, i, :] + K[t, i, :] * v[t, i]
                a[t, i + 1, :] = a[t, i, :]
                # P[t, i + 1, :, :] = P[t, i, :, :] - (K[t, i, :] * F[t, i]).reshape(
                #     (m, 1)).dot(K[t, i].reshape((1, m)))
                P[t, i + 1, :, :] = P[t, i, :, :]

        a[t+1, 0, :] = T.dot(a[t, i+1, :])
        P[t+1, 0, :, :] = T.dot(P[t, i+1]).dot(TT) + R.dot(Q).dot(RT)

        yhat[t, :] = Z.dot(a[t, 0, :])

    # if likelihood:
    #     ll = 0
    #     for t in range(0, n):
    #         print(F[t, :].shape)
    #         print(np.log(det(F[t,:])).shape)
    #         ll += np.log(det(F[t,:])) + v[t,:].T.dot(inv(F[t,:])).dot(v[t,:])
    #     ll = - n * p * 0.5 * np.log(2 * np.pi) - 0.5 * ll



    return {'states' : a[:, 0, :],
            'yhat' : yhat,
            'y' : y,
            'F' : F,
            # 'likelihood' : -ll[0][0]
            }
