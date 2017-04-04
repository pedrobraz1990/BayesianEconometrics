import numpy as np
from numpy.linalg import inv
import pandas as pd
import datetime as dt

from KalmanFilter_Uni import *
from KFUC import KFUC

m = 2
p = 4


Z = [[0.3,0.7],[0.1,0],[0.5,0.5],[0,0.3]]


Z = pd.DataFrame(Z)

H = pd.DataFrame(np.diag([1.0,2.0,3.0,4.0]))


T = pd.DataFrame(np.identity(2))
R = pd.DataFrame(np.identity(2))

Q = pd.DataFrame(np.diag([0.2,0.4]))

n = 1000  # sample size
mut = [np.array([1.0, 10.0]).reshape(m, 1)]
yt = [np.array([0.0, 0.0, 0.0, 0.0]).reshape(p, 1)]

for i in range(0, 1000):
    temp = np.multiply(np.random.randn(m, 1), np.diag(Q).reshape((m, 1)))
    temp = R.dot(temp)
    temp = temp + mut[i]
    mut.append(temp)

    temp = np.multiply(np.random.randn(p, 1), np.diag(H).reshape((p, 1)))
    yt.append(temp + Z.dot(mut[i + 1]))

yt[0] = pd.DataFrame(yt[0])
y = pd.concat(yt, axis=1).T.reset_index(drop=True)
mut[0] = pd.DataFrame(mut[0])
mut = pd.concat(mut, axis=1).T.reset_index(drop=True)

nny = y
probNan = 0.00
for i in nny.index:
    ran = np.random.uniform(size=nny.iloc[i].shape)
    nny.iloc[i][ran < probNan] = np.nan

# kf = KalmanFilter(y,
#                   Z,
#                   H,
#                   T,
#                   Q,
#                   pd.DataFrame(np.array([0,0]).reshape(m,1)),
#                   pd.DataFrame(np.diag(np.array([1,1]))),
#                   R,
#                   nStates=2)
#
# kf.runFilter()


kf2 = KFUC(y=np.array(nny),
              Z=np.array(Z),
              H=np.array(H),
              T=np.array(T),
              Q=np.array(Q),
              a1=np.array(pd.DataFrame(np.array([0.0,0.0]).reshape(m,1))),
              P1=np.diag(np.array([1.0,1.0])),
              R=np.array(R),
             nStates=2)