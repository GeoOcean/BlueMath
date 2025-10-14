#!/usr/bin/env python
# -*- coding: utf-8 -*-

# cartopy
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader

# bluemath resources
from ..config import p_shp_coast


def load_coast():
    '''
    Load resources coastfile and returns coast as ShapelyFeature
    '''

    return ShapelyFeature(
        Reader(p_shp_coast).geometries(),
        ccrs.PlateCarree(),
        edgecolor = 'black'
    )

