#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@originalauthor: alessio ansuini (alessioansuini@gmail.com)

@repo: https://github.com/ansuini/IntrinsicDimDeep

@modified by: heiner spiess (h.spiess@tu-berlin.de)
"""

import numpy as np
from scipy.stats import pearsonr
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform


def estimate(X, sort=True, fraction=0.9, verbose=False):
    '''
        Estimates the intrinsic dimension of a system of points from
        the matrix of their distances X
        
        Args:
        X : 2-D Matrix X (n,n) where n is the number of points
        fraction : fraction of the data considered for the dimensionality
        estimation (default : fraction = 0.9)

        Returns:            
        x : log(mu)    (*)
        y : -(1-F(mu)) (*)
        reg : linear regression y ~ x structure obtained with scipy.stats.linregress
        (reg.slope is the intrinsic dimension estimate)
        r : determination coefficient of y ~ x
        pval : p-value of y ~ x
            
        (*) See cited paper for description
        
        Usage:
            
        _,_,reg,r,pval = estimate(X,fraction=0.85)
            
        The technique is described in : 
            
        "Estimating the intrinsic dimension of datasets by a 
        minimal neighborhood information"       
        Authors : Elena Facco, Maria d’Errico, Alex Rodriguez & Alessandro Laio        
        Scientific Reports 7, Article number: 12140 (2017)
        doi:10.1038/s41598-017-11873-y
    
    '''

    # sort distance matrix
    if sort:
        Y = np.sort(X, axis=1, kind='quicksort')

    # clean data
    k1 = Y[:, 1]
    k2 = Y[:, 2]

    zeros = np.where(k1 == 0)[0]
    if verbose:
        print('Found n. {} elements for which r1 = 0'.format(zeros.shape[0]))
        print(zeros)

    degeneracies = np.where(k1 == k2)[0]
    if verbose:
        print('Found n. {} elements for which r1 = r2'.format(
            degeneracies.shape[0]))
        print(degeneracies)

    good = np.setdiff1d(np.arange(Y.shape[0]), np.array(zeros))
    good = np.setdiff1d(good, np.array(degeneracies))

    if verbose:
        print('Fraction good points: {}'.format(good.shape[0] / Y.shape[0]))

    k1 = k1[good]
    k2 = k2[good]

    # n.of points to consider for the linear regression
    npoints = int(np.floor(good.shape[0] * fraction))

    # define mu and Femp
    N = good.shape[0]
    mu = np.sort(np.divide(k2, k1), axis=None, kind='quicksort')
    Femp = (np.arange(1, N + 1, dtype=np.float64)) / N

    # take logs (leave out the last element because 1-Femp is zero there)
    x = np.log(mu[:-2])
    y = -np.log(1 - Femp[:-2])

    # regression
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(x[0:npoints, np.newaxis], y[0:npoints, np.newaxis])
    r, pval = pearsonr(x[0:npoints], y[0:npoints])
    return x, y, regr.coef_[0][0], r, pval


def block_analysis(X, blocks=list(range(1, 21)), fraction=0.9):
    '''
        Perform a block-analysis of a system of points from
        the matrix of their distances X
        
        Args:
        X : 2-D Matrix X (n,n) where n is the number of points
        blocks : blocks specification, is a list of integers from
        1 to N_blocks where N_blocks is the number of blocks (default : N_blocks = 20)
        fraction : fraction of the data considered for the dimensionality
        estimation (default : fraction = 0.9)
    '''

    n = X.shape[0]
    dim = np.zeros(len(blocks))
    std = np.zeros(len(blocks))
    n_points = []

    for b in blocks:
        # split indexes array
        idx = np.random.permutation(n)
        npoints = int(np.floor((n / b)))
        idx = idx[0:npoints * b]
        split = np.split(idx, b)
        tdim = np.zeros(b)
        for i in range(b):
            I = np.meshgrid(split[i], split[i], indexing='ij')
            tX = X[tuple(I)]
            _, _, reg, _, _ = estimate(tX, fraction=fraction, verbose=False)
            tdim[i] = reg
        dim[blocks.index(b)] = np.mean(tdim)
        std[blocks.index(b)] = np.std(tdim)
        n_points.append(npoints)
    return dim, std, n_points


def computeID(features, nres=1, fraction=1.0, fraction_linear_regr=1.0, verbose=False, dist='euclidean'):
    '''
    nres: number of resamplings for error estimation
    fraction: fraction of data resampling for error estimation
    '''
    ids = computeID_unagg(features, nres, fraction, fraction_linear_regr, verbose, dist)
    mean = np.mean(ids)
    error = np.std(ids)
    return mean, error


def computeID_unagg(features, nres=1, fraction=1.0, fraction_linear_regr=1.0, verbose=False, dist='euclidean'):
    '''
    nres: number of resamplings for error estimation
    fraction: fraction of data resampling for error estimation
    '''
    ids = []
    n = int(np.round(features.shape[0] * fraction))
    dist = squareform(pdist(features, dist))
    for _ in range(nres):
        dist_s = dist
        perm = np.random.permutation(dist.shape[0])[0:n]
        dist_s = dist_s[perm, :]
        dist_s = dist_s[:, perm]
        ids.append(estimate(dist_s, fraction=fraction_linear_regr, verbose=verbose)[2])
    return ids


def computeIDfast(features, nres=1, fraction=1.0, fraction_linear_regr=1.0, verbose=False, dist='euclidean'):
    assert dist in ('euclidean', 'manhattan'), "Invalid distance. Change code to allow 'precomputed' metric."
    if dist == 'manhattan':
        p = 1
    else:
        p = 2

    ids = []
    n = int(np.round(features.shape[0] * fraction))
    for _ in range(nres):
        perm = np.random.permutation(features.shape[0])[0:n]
        feats = features[perm, :]

        nbrs = NearestNeighbors(3, p=p).fit(feats)
        dists, _ = nbrs.kneighbors(feats)

        ids.append(estimate(dists, sort=False, fraction=fraction_linear_regr, verbose=verbose)[2])
    mean = np.mean(ids)
    error = np.std(ids)
    return mean, error
