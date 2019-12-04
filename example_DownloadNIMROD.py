#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 21:52:10 2019

@author: Xiaodong Ming
"""

# download file from CEDA ftp

import nimrodProcessing as NP 

# your username and password on CEDA website
NP.downloadNimrodTar(dateStr='20190131')
tarList = glob.glob('*.tar')
tarList.sort()

#%% read DEM data to generate rainfall mask and rainfall source
demFileName = 'eden_20m.asc'
demArray,demHead,demExtent = AP.arcgridread(demFileName)

# the start date and time of model simulation
initalDatetime = datetime.datetime(2019,1,31,0,0)

# rainMask is an array with the same size of DEM array
# rainSource is an rainfall rate array with time series (1st col in seconds) 
rainMask,rainSource=NF.getRainMaskSource(tarList,demHead,datetimeStart=initalDatetime)

# write rainfall mask and rainfall source to files
AP.arcgridwrite('rain_mask20m.asc',Z= rainMask,head=demHead)
np.savetxt('rain_Source.txt',rainSource,fmt='%g')