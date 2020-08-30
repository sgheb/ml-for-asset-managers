import numpy as np
import scipy.stats as ss
from sklearn.metrics import mutual_info_score

# Snippet 3.2
def varInfo(x, y, bins, norm=False):
    """Variation of information"""
    cXY = np.histogram2d(x, y, bins)[0]
    iXY = mutual_info_score(None, None, contingency=cXY)

    hX = ss.entropy(np.histogram(x, bins)[0])  # marginal
    hY = ss.entropy(np.histogram(y, bins)[0])

    vXY = hX + hY - 2 * iXY  # variation of information
    if norm:
        hXY = hX + hY - iXY
        vXY /= hXY

    return vXY


# Snippet 3.3
def numBins(noObs, corr=None):
    """Optimal number of bins for discretization"""
    if corr is None:  # univariate case
        z = (8 + 324 * nObs + 12 * (36 * nObs + 729 * nObs ** 2) ** 0.5) ** 0.5
        b = round(z / 6.0 + 2.0 / (3 * z) + 1.0 / 3)
    else:  # bivariate case
        b = round(2 ** -0.5 * (1 + (1 + 24 * nObs / (1.0 - corr ** 2)) ** 0.5) ** 0.5)
    return int(b)


def varInfo2(x, y, norm=False):
    """Variation of information"""
    bXY = numBins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])
    cXY = np.histogram2d(x, y, bXY)[0]
    iXY = mutual_info_score(None, None, contingency=cXY)
    hX = ss.entropy(np.histogram(x, bXY)[0])
    hY = ss.entropy(np.histogram(y, bXY)[0])
    vXY = hX + hY - 2 * iXY
    if norm:
        hXY = hX + hY - iXY
        vXY /= hXY
    return vXY


# Snippet 3.4
def mutualInfo(x, y, norm=False):
    """Mutual information"""
    bXY = numBins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])
    cXY = np.histogram2d(x, y, bXY)[0]
    iXY = mutual_info_score(None, None, contigency=cXY)
    if norm:
        hX = ss.entropy(np.histogram(x, bXY)[0])
        hY = ss.entropy(np.histogram(y, bXY)[0])
        iXY /= min(hX, hY)
    return iXY
