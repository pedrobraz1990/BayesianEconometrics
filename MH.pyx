import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
import pandas as pd
import datetime as dt
import pyximport; pyximport.install()
from KalmanFilter import KalmanFilter
from scipy.stats import norm
from scipy.stats import chi2


class MH:

    priorMeans = np.array([0, 0, 0, 0])

    sample = 2000

    priorVariances = 1 * np.ones(4)

    thetaMH = []
    accept = np.zeros([sample])

    thetaMH.append(np.array([0.5, 0.1, 0.1, 0.1, ]))

    thetastemps = []
    disturbances = []
    sigma = 0.3
    sigmas = np.array([0.1, sigma, sigma, sigma])

    Z = pd.DataFrame([
        [1, 1, 0, 0],
    ])

    R = pd.DataFrame([
        [1, 0],
        [0, 1],
        [0, 0],
        [0, 0],
    ])

    a1 = pd.DataFrame([0, 0, 0, 0])

    P1 = pd.DataFrame(np.diag([1, 1, 1, 1]))

    def __init__(self,y):

        self.y = y

    def likelihood(self,theta):
        T = pd.DataFrame([
            [1, 0, 0, 0],
            [0, -1, -1, -theta[0]],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ])

        H = pd.DataFrame([theta[1]])

        Q = pd.DataFrame([
            [theta[2], 0],
            [0, theta[3]],
        ])

        kf = KalmanFilter(y=self.y,
                          Z=self.Z.astype(float),
                          H=H.astype(float),
                          T=T.astype(float),
                          Q=Q.astype(float),
                          a1=self.a1.astype(float),
                          P1=self.P1.astype(float),
                          R=self.R.astype(float),
                          nStates=4,
                          performChecks=False)

        return kf.likelihood()

    def posterior(self, theta, ):
        i = 0
        temp = norm.pdf(theta[i], self.priorMeans[i], self.priorVariances[i])
        for i in range(1, 3 + 1):
            temp *= chi2.pdf(theta[i], 1)

        return self.likelihood(theta) * temp

    def genSample(self):
        for i in range(1, self.sample):
            disturbance = np.multiply(np.random.randn(self.thetaMH[0].shape[0]), self.sigmas)
            #     disturbance[1:] = np.exp(disturbance[1:]) #problema está aqui
            #     print(disturbance)
            thetaTemp = self.thetaMH[i - 1].copy()
            thetaTemp[1:] = np.log(self.thetaMH[i - 1][1:]) + disturbance[1:]
            thetaTemp[1:] = np.exp(thetaTemp[1:])
            thetaTemp[0] = thetaTemp[0] + disturbance[0]
            self.disturbances.append(disturbance)
            self.thetastemps.append(thetaTemp.copy())
            #     print(thetaTemp)
            #     print(thetastemps[i-1])
            #     thetaTemp = (temp)

            lalpha = np.log(self.posterior(thetaTemp)) - np.log(
                self.posterior(self.thetaMH[i - 1]))
            r = np.min([1, np.exp(lalpha)])

            u = np.random.uniform()

            if u < r:
                self.accept[i] = 1
                self.thetaMH.append(thetaTemp)
            else:
                self.thetaMH.append(self.thetaMH[i - 1])