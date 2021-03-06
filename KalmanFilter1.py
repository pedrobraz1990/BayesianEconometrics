import numpy as np
from numpy.linalg import inv
import pandas as pd
import datetime as dt


class KalmanFilter():
        """
            Based on Durbin Koopman


            For generic state-space systems like (Hamilton's notation):



            $$\underset{(p \times 1)}{y_t} = \underset{(p \times m)}{Z_t} \underset{(m \times 1)}{\alpha_t} + \underset{(p \times 1)}{\epsilon_t}, \qquad \epsilon_t \sim N(0,\underset{(p \times P)}{H_t}) $$

            $$\underset{(m \times 1)}{\alpha_{t+1}} = \underset{(m \times m)}{T_t}
            \underset{(m \times 1)}{\alpha_{t}} + \underset{(m \times r)}{R_t} \underset{(r \times r)}{\eta_t}, \qquad \eta_t \sim N(0,\underset{(r \times \ r)}{Q_t})$$


            $$\alpha_1 \sim N(a_1,\underset{(m \times m)}{P_1})$$

            Where

            * $p$ is the number of observed variables
            * $m$ is the number of latent states
            * $r$ is the number of disturbances

        """

    def __init__(self, y, Z, H, T, Q, a1, P1, R):
        self.yindex = y.index
        self.ycols = y.columns
        self.p = y.shape[1]
        self.n = y.shape[0]
        self.y = np.array(y)
        self.yhat = []
        self.Z = np.array(Z)
        self.H = np.array(H)
        self.T = np.array(T)
        self.Q = np.array(Q)
        self.a = [np.array(a1)]
        self.P = [np.array(P1)]
        self.vt = []
        self.Ft = []
        self.Kt = []
        self.ZT = Z.T  # To avoid transposing it several times
        self.R = np.array(R)

    def runFilter(self, ):
        # Implemented with non time varying coefficients

        for i in range(0, self.n - 1):
            #         for i in range(0,1):
            #             print(self.y[i].shape)
            #             print(self.Z.shape)
            #             print(self.a[i].shape)

            self.vt.append(self.y[i].reshape((self.p, 1)) - np.dot(self.Z, self.a[i]))

            self.Ft.append(self.Z.dot(self.P[i]).dot(self.ZT) + self.H)

            Finv = inv(self.Ft[i])

            #             print(self.P[i].shape)
            #             print(self.ZT.shape)
            #             print(Finv.shape)
            #             print(self.vt[i].shape)
            self.a[i] = self.a[i] + self.P[i].dot(self.ZT).dot(Finv).dot(self.vt[i])

            self.P[i] = self.P[i] - self.P[i].dot(self.ZT).dot(Finv).dot(self.Z).dot(self.P[i])

            self.a.append(self.T.dot(self.a[i]))

            self.P.append(self.T.dot(self.P[i]).dot(self.T.T) + self.R.dot(self.Q).dot(self.R.T))

            self.yhat.append(self.Z.dot(self.a[i]))

        self.a = pd.DataFrame(np.concatenate(self.a, axis=1)).T
        self.yhat = pd.DataFrame(np.concatenate(self.yhat, axis=1)).T