# basic
import os
import os.path as op
import sys

# common
import numpy as np
import pandas as pd
import xarray as xr


def Normalize(data, ix_scalar, ix_directional, minis=[], maxis=[]):
    '''
    Normalize data subset - norm = val - min) / (max - min)

    data - data to normalize, data variables at columns.
    ix_scalar - scalar columns indexes
    ix_directional - directional columns indexes
    '''

    data_norm = np.zeros(data.shape) * np.nan
    # calculate maxs and mins 
    if minis==[] or maxis==[]:

        # scalar data
        for ix in ix_scalar:
            v = data[:, ix]
            mi = np.amin(v)
            ma = np.amax(v)
            data_norm[:, ix] = (v - mi) / (ma - mi)
            minis.append(mi)
            maxis.append(ma)

        minis = np.array(minis)
        maxis = np.array(maxis)

    # max and mins given
    else:

        # scalar data
        for c, ix in enumerate(ix_scalar):
            v = data[:, ix]
            mi = minis[c]
            ma = maxis[c]
            data_norm[:,ix] = (v - mi) / (ma - mi)

    # directional data
    for ix in ix_directional:
        v = data[:,ix]
        data_norm[:,ix] = (v * np.pi / 180.0)/np.pi


    return data_norm, minis, maxis


def DeNormalize(data_norm, ix_scalar, ix_directional, minis, maxis):
    '''
    DeNormalize data subset for MaxDiss algorithm

    data - data to normalize, data variables at columns.
    ix_scalar - scalar columns indexes
    ix_directional - directional columns indexes
    '''

    data = np.zeros(data_norm.shape) * np.nan

    # scalar data
    for c, ix in enumerate(ix_scalar):
        v = data_norm[:,ix]
        mi = minis[c]
        ma = maxis[c]
        data[:, ix] = v * (ma - mi) + mi

    # directional data
    for ix in ix_directional:
        v = data_norm[:,ix]
        data[:, ix] = v * 180 #/ np.pi

    return data
