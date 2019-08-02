#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:17:28 2019

@author: b4042552
"""
#%%
import numpy as np
import sys
sys.path.insert(0,'/Users/b4042552/Dropbox/Python/HiPIMS')
from myclass import raster

objDEM = raster('eden_5m.gz')
#%%
Shapefile = 'NRFA/76014.shp'
#raster_ds = objDEM.to_osgeo_raster()
objDEM_clip = objDEM.Clip(Shapefile)
objDEM_clip.Write_asc('NRFA/76014_5m.asc',EPSG=27700)
#%%
#Shapefile = 'RoadsLine.shp'
#data = gdal.Open('DEM2m_NewcastleFrame2.tif')
#objDS = objDEM.To_osgeo_raster()
from myclass import inputHiPIMS
obj = inputHiPIMS(objDEM.array,objDEM.header)

# define boundary condition
bound1Points = np.array([[535, 206], [545, 206], [545, 210], [535, 210]])*1000
bound2Points = np.array([[520, 230], [530, 230], [530, 235], [520, 235]])*1000
dBound0 = {'type': 'rigid'} #outline rigid boundary
dBound1 = {'polyPoints': bound1Points,'type': 'IO','h': [[0,10],[60,10]]}
dBound2 = {'polyPoints': bound2Points,'type': 'IO','hU': [[0,50000],[60,30000]]}
boundList = [dBound0,dBound1,dBound2]
obj.GenerateBoundaryCondition(boundList)