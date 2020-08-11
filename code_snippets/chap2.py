import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
from scipy.linalg import block_diag
from sklearn.covariance import LedoitWolf


def fix_shape(x):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    return x


# Snippet 2.1
def mpPDF(var, q, pts):
    """Marcenko--Pastur PDF"""
    # q = T/N

    eMin, eMax = var * (1 - (1.0 / q) ** 0.5) ** 2, var * (1 + (1.0 / q) ** 0.5) ** 2
    eVal = np.linspace(eMin, eMax, pts)
    pdf = q / (2 * np.pi * var * eVal) * ((eMax - eVal) * (eVal - eMin)) ** 0.5
    pdf = pd.Series(pdf, index=eVal)
    return pdf


# Snippet 2.2
def getPCA(matrix):
    """Get eVal, eVec from a Hermitian matrix"""
    eVal, eVec = np.linalg.eigh(matrix)
    indices = eVal.argsort()[::-1]  # arguments for sorting eVal desc
    eVal, eVec = eVal[indices], eVec[:, indices]
    eVal = np.diagflat(eVal)
    return eVal, eVec


def fitKDE(obs, bWidth=0.25, kernel="gaussian", x=None):
    """
    Fit kernel to a series of observations `obs` and derive the probability.

    `x` is the array of values on which the fit KDE will be evaluated
    """
    obs = fix_shape(obs)
    kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)
    if x is None:
        x = np.unique(obs).reshape(-1, 1)
    x = fix_shape(x)
    logProb = kde.score_samples(x)  # log(density)
    pdf = pd.Series(np.exp(logProb), index=x.flatten())
    return pdf


# Snippet 2.3
def getRndCov(nCols, nFacts):
    w = np.random.normal(size=(nCols, nFacts))
    cov = np.dot(w, w.T)
    cov += np.diag(np.random.uniform(size=nCols))
    return cov


def cov2corr(cov):
    """Derive the correlation matrix from covariance matrix"""
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1  # numerical error
    return corr


# Snippet 2.4
def errPDFs(var, eVal, q, bWidth, pts=1000):
    """Fit error"""
    pdf0 = mpPDF(var, q, pts)  # theoretical pdf
    pdf1 = fitKDE(
        eVal, bWidth, x=pdf0.index.values
    )  # empirical pdf with same x values as theoretical
    sse = np.sum((pdf1 - pdf0) ** 2)  # sum of square error
    return sse


def findMaxEval(eVal, q, bWidth):
    """Find max random eVal by fitting Marcenko's distribution"""
    out = minimize(
        lambda x, *args: errPDFs(x[0], *args),
        0.5,
        args=(eVal, q, bWidth),
        bounds=((1e-5, 1 - 1e-5),),
    )
    if out["success"]:
        var = out["x"][0]
    else:
        var = 1
    eMax = var * (1 + (1.0 / q) ** 0.5) ** 2
    return eMax, var


# Snippet 2.5
def denoisedCorr(eVal, eVec, nFacts):
    """Remove noise from corr by fixing random eigenvalues"""
    eVal_ = np.diag(eVal).copy()
    eVal_[nFacts:] = eVal_[nFacts:].sum() / float(eVal_.shape[0] - nFacts)
    eVal_ = np.diag(eVal_)
    corr1 = np.dot(eVec, eVal_).dot(eVec.T)
    corr1 = cov2corr(corr1)
    return corr1


# Snippet 2.6
def denoisedCorr2(eVal, eVec, nFacts, alpha=0):
    """Remove noise from corr through targeted shrinkage"""
    eValL, eVecL = eVal[:nFacts, :nFacts], eVec[:, :nFacts]
    eValR, eVecR = eVal[nFacts:, nFacts:], eVec[:, nFacts:]
    corr0 = np.dot(eVecL, eValL).dot(eVecL.T)
    corr1 = np.dot(eVecR, eValR).dot(eVecR.T)
    corr2 = corr0 + alpha * corr1 + (1 - alpha) * np.diag(np.diag(corr1))
    return corr2


# Snippet 2.7
def formBlockMatrix(nBlocks, bSize, bCorr):
    block = np.ones((bSize, bSize)) * bCorr
    block[range(bSize), range(bSize)] = 1
    corr = block_diag(*([block] * nBlocks))
    return corr


def formTrueMatrix(nBlocks, bSize, bCorr):
    corr0 = formBlockMatrix(nBlocks, bSize, bCorr)
    corr0 = pd.DataFrame(corr0)
    cols = corr0.columns.tolist()
    np.random.shuffle(cols)
    corr0 = corr0[cols].loc[cols].copy(deep=True)
    std0 = np.random.uniform(0.05, 0.2, corr0.shape[0])
    cov0 = corr2cov(corr0, std0)
    mu0 = np.random.normal(std0, std0, cov0.shape[0]).reshape(-1, 1)
    return mu0, cov0


# Snippet 2.8
def simCovMu(mu0, cov0, nObs, shrink=False):
    x = np.random.multivariate_normal(mu0.flatten(), cov0, size=nObs)
    mu1 = x.mean(axis=0).reshape(-1, 1)
    if shrink:
        cov1 = LedoitWolf().fit(x).covariance_
    else:
        cov1 = np.cov(x, rowvar=0)
    return mu1, cov1


# Snippet 2.9
def corr2cov(corr, std):
    cov = corr * np.outer(std, std)
    return cov


def deNoiseCov(cov0, q, bWidth):
    corr0 = cov2corr(cov0)
    eVal0, eVec0 = getPCA(corr0)
    eMax0, var0 = findMaxEval(np.diag(eVal0), q, bWidth)
    nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)
    corr1 = denoisedCorr(eVal0, eVec0, nFacts0)
    cov1 = corr2cov(corr1, np.diag(cov0) ** 0.5)
    return cov1


# Snippet 2.10
def optPort(cov, mu=None):
    inv = np.linalg.inv(cov)
    ones = np.ones(shape=(inv.shape[0], 1))
    if mu is None:
        mu = ones
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w)
    return w
