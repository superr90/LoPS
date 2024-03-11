import numpy as np
from scipy.special import gammaln

from PGM.Utils import *


# from src.bayesianScore import BDscore

def BDscore(dataV, dataParents, nstatesV, nstatesPa, u):
    U = u * np.ones((int(np.prod(nstatesV)), int(np.prod(nstatesPa))))
    if len(dataParents) != 0:
        cvpa = count(np.vstack((dataV, dataParents)), np.hstack((nstatesV, nstatesPa)))
    else:
        cvpa = count(dataV, nstatesV)
    CvgPa = cvpa.reshape(np.prod(nstatesV), int(np.prod(nstatesPa)), order='F')
    Up = U + CvgPa
    score = np.sum(
        gammaln(np.sum(U, axis=0)) - gammaln(np.sum(Up, axis=0)) + np.sum(gammaln(Up), axis=0) - np.sum(gammaln(U),
                                                                                                        axis=0), axis=0)
    return score, Up


def logZdirichlet(u):
    return np.sum(gammaln(u), axis=0) - gammaln(np.sum(u, axis=0))


def condindepEmp(dataX, dataY, dataZ, X, Y, Z, thresh, opts, method="PC"):
    Uxgz = opts["Uxgz"]
    Uygz = opts["Uygz"]
    Uz = opts["Uz"]
    Uxyz = opts["Uxyz"]

    # model of p(x|z)
    if len(dataZ) != 0:
        cxz = count(np.vstack((dataX, dataZ)), np.hstack((X, Z)))
    else:
        cxz = count(dataX, X)
    logZux = logZdirichlet(Uxgz * np.ones(np.prod(X)))
    Cxgz = cxz.reshape(np.prod(X), int(np.prod(Z)), order='F')
    logpxgz = np.sum(logZdirichlet(Cxgz + Uxgz) - logZux)

    # model of p(y|z)
    if len(dataZ) != 0:
        cyz = count(np.vstack((dataY, dataZ)), np.hstack((Y, Z)))
    else:
        cyz = count(dataY, Y)
    logZuy = logZdirichlet(Uygz * np.ones(np.prod(Y)))
    Cygz = cyz.reshape(np.prod(Y), int(np.prod(Z)), order='F')
    logpygz = np.sum(logZdirichlet(Cygz + Uygz) - logZuy)

    # p(z)
    cz = count(dataZ, Z)
    logZuz = logZdirichlet(Uz * np.ones(int(np.prod(Z))))
    if len(cz) != 0:
        logpz = logZdirichlet(cz + Uz) - logZuz
    else:
        logpz = logZdirichlet(np.array([])) - logZuz

    if len(dataZ) != 0:
        logpindep = logpxgz + logpygz + logpz
    else:
        logpindep = logpxgz + logpygz

        # model of p(xyz)
    if len(dataZ) != 0:
        cxyz = count(np.vstack((dataX, dataY, dataZ)), np.hstack((X, Y, Z)))
        logZuxyz = logZdirichlet(Uxyz * np.ones(np.prod(np.hstack((X, Y, Z)))))
    else:
        cxyz = count(np.vstack((dataX, dataY)), np.hstack((X, Y)))
        logZuxyz = logZdirichlet(Uxyz * np.ones(np.prod(np.hstack((X, Y)))))

    logpdep = logZdirichlet(cxyz + Uxyz) - logZuxyz
    logBayesFactor = logpindep - logpdep
    ind = logBayesFactor > thresh
    if method == "BD":
        return logpxgz + logpygz
    else:
        return ind, logpindep, logpdep
