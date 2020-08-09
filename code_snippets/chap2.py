import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


def mpPDF(var, q, pts):
    """Marcenko--Pastur PDF"""
    # q = T/N
    eMin, eMax = var * (1 - np.sqrt(1.0 / q)) ** 2, var * (1 + np.sqrt(1.0 / q)) ** 2
    eVal = np.linspace(eMin, eMax, pts)
    pdf = q / (2 * np.pi * var * eVal) * np.sqrt((eMax - eVal) * (eVal - eMin))
    pdf = pd.Series(pdf, index=eVal)
    return pdf


def getPCA(matrix):
    """Get eVal, eVec from a Hermitian matrix"""
    eVal, eVec = np.linalg.eigh(matrix)
    indices = eVal.argsort()[::-1]  # artuments for sortinig eVal desc
    eVal, eVec = eVal[indices], eVec[:, indices]
    eVal = np.diagflat(eVal)
    return eVal, eVec


def fix_shape(x):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    return x


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
