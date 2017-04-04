import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
import pandas as pd
import datetime as dt


class KalmanFilter():
    #         """
    #             Based on Durbin Koopman


    #             For generic state-space systems like (Hamilton's notation):



    #             $$\underset{(p \times 1)}{y_t} = \underset{(p \times m)}{Z_t} \underset{(m \times 1)}{\alpha_t} + \underset{(p \times 1)}{\epsilon_t}, \qquad \epsilon_t \sim N(0,\underset{(p \times P)}{H_t}) $$

    #             $$\underset{(m \times 1)}{\alpha_{t+1}} = \underset{(m \times m)}{T_t}
    #             \underset{(m \times 1)}{\alpha_{t}} + \underset{(m \times r)}{R_t} \underset{(r \times r)}{\eta_t}, \qquad \eta_t \sim N(0,\underset{(r \times \ r)}{Q_t})$$


    #             $$\alpha_1 \sim N(a_1,\underset{(m \times m)}{P_1})$$

    #             Where

    #             * $p$ is the number of observed variables
    #             * $m$ is the number of latent states
    #             * $r$ is the number of disturbances

    #         """

    def __init__(self, y, Z, H, T, Q, a1, P1, R, nStates, performChecks = True):

        self.yindex = y.index
        self.ycols = y.columns
        self.p = y.shape[1]
        self.n = y.shape[0]
        self.y = np.array(y)
        self.m = nStates

        # Checks
        if performChecks:
            if Z.shape != (self.p,self.m):
                print('Wrong shape for Z')

            if a1.shape != (self.m, 1):
                print('Wrong shape for a1')

            if H.shape != (self.p, self.p):
                print('Wrong shape for H')

            if T.shape != (self.m, self.m):
                print('Wrong shape for T')

            if P1.shape != (self.m, self.m):
                print('Wrong shape for P1')

        ind = np.zeros(self.y.shape[0])
        ind[np.isnan(self.y).any(axis=1)] = 1  # Some NaNs
        ind[np.isnan(self.y).all(axis=1)] = 2  # All NaNs
        self.ind = ind

        self.yhat = []
        self.Z = np.array(Z.astype(float))
        self.H = np.array(H.astype(float))
        self.T = np.array(T.astype(float))
        self.Q = np.array(Q.astype(float))
        self.a = [np.array(a1.astype(float))]
        self.P = [np.array(P1.astype(float))]
        self.vt = []
        self.Ft = []
        self.Kt = []
        self.ZT = Z.T  # To avoid transposing it several times
        self.TT = self.T.T  # To avoid transposing it several times
        self.R = np.array(R)
        self.RT = self.R.T
        self.ranFilter = False

    def runFilter(self, ):
        # Implemented with non time varying coefficients

        for i in range(0, self.n - 1):

            if self.ind[i] == 0:

                self.vt.append(self.y[i].reshape((self.p, 1)) - np.dot(self.Z, self.a[i]))

                self.Ft.append(self.Z.dot(self.P[i]).dot(self.ZT) + self.H)

                Finv = inv(self.Ft[i])

                self.a[i] = self.a[i] + self.P[i].dot(self.ZT).dot(Finv).dot(self.vt[i])

                self.P[i] = self.P[i] - self.P[i].dot(self.ZT).dot(Finv).dot(self.Z).dot(self.P[i])

                self.a.append(self.T.dot(self.a[i]))

                self.P.append(self.T.dot(self.P[i]).dot(self.TT) + self.R.dot(self.Q).dot(self.RT))

                self.yhat.append(self.Z.dot(self.a[i]))

            elif self.ind[i] == 2:  # In case the line is all nans

                self.vt.append(np.zeros((self.p, 1)))
                #                 self.vt.append(self.y[i].reshape((self.p, 1)) - np.dot(self.Z, self.a[i]))

                self.Ft.append(self.Z.dot(self.P[i]).dot(self.ZT) + self.H)

                #                 Finv = inv(self.Ft[i])

                #                 self.a[i] = self.a[i] + self.P[i].dot(self.ZT).dot(Finv).dot(self.vt[i])

                #                 self.P[i] = self.P[i] - self.P[i].dot(self.ZT).dot(Finv).dot(self.Z).dot(self.P[i])

                self.a.append(self.T.dot(self.a[i]))

                self.P.append(self.T.dot(self.P[i]).dot(self.TT) + self.R.dot(self.Q).dot(self.RT))

                self.yhat.append(self.Z.dot(self.a[i]))

            else:
                # First use an index for nulls
                ind = ~np.isnan(self.y[i]).ravel()
                yst = self.y[i][ind]
                Zst = self.Z[ind, :]
                ZstT = Zst.T

                select = np.diag(ind)
                select = select[(select == True).any(axis=1)].astype(int)

                Hst = select.dot(self.H).dot(select.T)

                self.vt.append(yst.reshape((yst.shape[0], 1)) - np.dot(Zst, self.a[i]))

                self.Ft.append(Zst.dot(self.P[i]).dot(ZstT) + Hst)

                Finv = inv(self.Ft[i])

                self.a[i] = self.a[i] + self.P[i].dot(ZstT).dot(Finv).dot(self.vt[i])

                self.P[i] = self.P[i] - self.P[i].dot(ZstT).dot(Finv).dot(Zst).dot(self.P[i])

                self.a.append(self.T.dot(self.a[i]))

                self.P.append(self.T.dot(self.P[i]).dot(self.TT) + self.R.dot(self.Q).dot(self.RT))

                yhat = np.empty((self.p, 1))
                yhat[ind, :] = Zst.dot(self.a[i])
                yhat[~ind, :] = self.Z.dot(self.a[i])[~ind, :]

                self.yhat.append(yhat)

        self.a = pd.DataFrame(np.concatenate(self.a, axis=1)).T
        self.yhat = pd.DataFrame(np.concatenate(self.yhat, axis=1)).T
        self.y = pd.DataFrame(self.y)

        self.ranFilter = True



    def likelihood(self):
        if not self.ranFilter:
            self.runFilter()

        ll = 0
        for i in range(0, self.n-1):
            ll += np.log(det(self.Ft[i])) +  self.vt[i].T.dot(inv(self.Ft[i])).dot(self.vt[i])
        ll = - self.n* self.p*0.5 * np.log(2*np.pi) - 0.5 * ll
        self.ll = ll[0][0]
        return np.exp(ll[0][0])

#TODO Later check if any matrix multiplications always yields same result
#TODO change i for t