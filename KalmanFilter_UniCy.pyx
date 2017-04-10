import numpy as np
cimport numpy as np
import pandas as pd
import datetime as dt

DTYPE = np.float64
DTYPEI = np.int

ctypedef np.float64_t DTYPE_t
ctypedef np.int_t DTYPEI_t


def KalmanFilter(np.ndarray[DTYPE_t, ndim = 2] y,
                DTYPEI_t nStates,
                np.ndarray[DTYPE_t, ndim = 2] Z,
                np.ndarray[DTYPE_t, ndim = 2] H,
                np.ndarray[DTYPE_t, ndim = 2] T,
                np.ndarray[DTYPE_t, ndim = 2] Q,
                np.ndarray[DTYPE_t, ndim = 1] a1,
                np.ndarray[DTYPE_t, ndim = 2] P1,
                np.ndarray[DTYPE_t, ndim = 2] R,
                                 export=False):
# Only receives np arrays

    cdef DTYPEI_t p = y.shape[1]
    cdef DTYPEI_t n = y.shape[0]
    cdef DTYPEI_t m = nStates
    cdef DTYPEI_t t, i

    cdef np.ndarray[DTYPE_t, ndim = 2] yhat = np.empty((n, p), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 3] a = np.empty((n + 1, p + 1, m), dtype=DTYPE)  # each alpha t,i is mx1
    a[0, 0, :] = a1
    cdef np.ndarray[DTYPE_t, ndim = 4] P = np.empty((n + 1, p + 1, m, m), dtype=DTYPE)
    P[0, 0, :, :] = np.array(P1.astype(float), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 2] v = np.empty((n, p), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 2] F = np.empty((n, p), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 3] K = np.empty((n, p, m), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 2] ZT = Z.T  # To avoid transposing it several times
    cdef np.ndarray[DTYPE_t, ndim = 2] TT = T.T  # To avoid transposing it several times
    cdef np.ndarray[DTYPE_t, ndim = 2] RT = R.T
    cdef DTYPE_t ll = 0
    cdef DTYPE_t templl = 0
    cdef DTYPE_t pst = 0
    #cdef np.ndarray[DTYPE_t, ndim = 1] ind


    for t in range(0, n):
        # ind = ~np.isnan(y[t, :], dtype=DTYPEI)
        ind = ~np.isnan(y[t, :])
        templl = 0
        pst = 0
        for i in range(0, p):  # later on change to Pt
            if ind[i]:
                v[t, i] = y[t, i] - Z[i, :].reshape((1, m)).dot(a[t, i, :].T)  # a should be mx1
                F[t, i] = Z[i, :].reshape((1, m)).dot(P[t, i, :, :]).dot(Z[i, :]) + H[i, i]
                K[t, i, :] = P[t, i, :, :].dot(Z[i, :]) * F[t, i] ** (-1)
                a[t, i + 1, :] = a[t, i, :] + K[t, i, :] * v[t, i]
                P[t, i + 1, :, :] = P[t, i, :, :] - (K[t, i, :] * F[t, i]).reshape((m, 1)).dot(
                    K[t, i].reshape((1, m)))
            else:
                # Setting all Z's to zeros
                v[t, i] = 0
                F[t, i] = H[
                    i, i]
                K[t, i, :] = np.zeros(K[t, i, :].shape)
                a[t, i + 1, :] = a[t, i, :] + K[t, i, :] * v[t, i]
                P[t, i + 1, :, :] = P[t, i, :, :] - (K[t, i, :] * F[t, i]).reshape(
                    (m, 1)).dot(K[t, i].reshape((1, m)))
            if F[t,i] != 0:
                templl += np.log(F[t,i]) + (v[t,i] ** 2) / F[t,i]
                pst += 1

        ll+= pst * np.log(2*np.pi) + templl


        a[t + 1, 0, :] = T.dot(a[t, i + 1, :])
        P[t + 1, 0, :, :] = T.dot(P[t, i + 1]).dot(TT) + R.dot(Q).dot(RT)
        # yhat[t,:] = Z.dot(a[t,1,:]) # ERRADO

        if export:
            yhat[t, :] = Z.dot(a[t, 0, :])

    ll *= -0.5
    ll = np.exp(ll)

    if export:
        states = pd.DataFrame(a[:, 0, :])
        yhat = pd.DataFrame(yhat)
        y = pd.DataFrame(y)
        return {'states' : states,
                'yhat' : yhat,
                'y' : y,
                'll' : ll}
    else:
        return ll