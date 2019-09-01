#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 10:01:43 2019

@author: Xiaodong Ming
"""
import os
import pandas as pd
import scipy.signal
#import copy
import numpy as np
from myclass import raster
#%% grid data for HiPIMS input format
class inputHiPIMS(object):
    """
    Properties:
        inputFolder: the absolute path of input folder, e.g. usr/case/0/,usr/case/1/
        cellInd: index of valid cells([0,1,...N],left to right, bottom to top) on DEM grid
        sharedID
        domain: object with
        bound: a boundary object
    methods(private): 
    """
    def __init__(self,demArray,header,numSection=1,inputFolder=None):
        if inputFolder is None:
            inputFolder=os.getcwd()
        self.inputFolder = inputFolder
        self.numSection = numSection
       # get row and col index of all cells on DEM grid 
        self.__GetCellSubs(demArray,header)

        if numSection>1:
            self.__DivideGrid()
            self.globalHeader = self.DEM.header
            self.DEM = None # delete global DEM to release system memory
            
    def __GenerateBoundaryCondition(self,boundList=None):
        """
        create a boundary object for boundary conditions
        
        boundList: a list of dict, each dict contain keys (polyPoints,type,h,hU)
            to define a boundary's position, type, and Input-Output (IO) sources 
            1.polyPoints is a numpy array giving X(1st col) and Y(2nd col) 
                coordinates of points to define the position of a boundary. 
                An empty polyPoints means outline boundary.
            2.type: 'open'|'rigid'
            3.h: a two-col numpy array. The 1st col is time(s). The 2nd col is 
                water depth(m)
            4.hU: a two-col numpy array. The 1st col is time(s). The 2nd col is 
                discharge(m3/s) or a three-col numpy array, the 2nd col and the
                3rd col are velocities(m/s) in x and y direction, respectively.
            Examples:
            
            bound1 = {'polyPoints': [],
                      'type': 'open',
                      'h': np.array([(0,0),(60,0)]),
                      'hU': np.array([(0,0,0),(60,0,0)])} #open outline boundary
            bound2 = {'polyPoints': np.array([(1,1),(1,10),(10,10),(10,1)]),
                      'type': 'open',
                      'h': np.array([(0,8),(60,10)]),
                      'hU': np.array([(0,0.1),(60,0.2)])}
            boundList = [bound1,bound2]
        
        """
        if boundList is None:
            boundList = []
        boundObj = boundary(boundList) # create a boundary object
        self.bound = boundObj
#%========================private functions==================================
    def __GetCellSubs(self,demArray,header):
        
        self.DEM = raster(array=demArray,header=header)
        #%Global and local operation: generate cell ID and outline boundary
        def __Get_ID_BNoutline_Mat(demMat):
            # cellID, bnMat_outline = Get_ID_BNoutline_Mat(demMat)
            Z = demMat*0+1
            Z_flip = np.flipud(Z)
            D1 = np.nancumsum(Z_flip)
            Z_flip1D = np.reshape(Z_flip,np.shape(D1))
            D1[np.isnan(Z_flip1D)]=np.nan
            D1 = np.reshape(D1,np.shape(Z_flip))
            # Series number of valid cells: 0 to number of cells-1
            # from bottom left corner towards right and top
            idMat = np.flipud(D1)-1
            del Z,Z_flip,D1,Z_flip1D
            D = idMat*0
            D[np.isnan(idMat)]=-1
            h_hv   = np.array([[0,1,0], [1,0,1], [0,1,0]])
            D = scipy.signal.convolve2d(D, h_hv,mode='same')
            D[D<0]=np.nan    
            D[0,:] = np.nan
            D[-1,:] = np.nan
            D[:,0] = np.nan
            D[:,-1] = np.nan
            # boundary cells with valid cell ID are extracted
            bnMat_outline = idMat*0-2 # non-outline cells:-2
            Outline_Cell_index = np.isnan(D)&~np.isnan(idMat)
                # outline boundary cell
            bnMat_outline[Outline_Cell_index] = 0 # outline cells:0
            return idMat,bnMat_outline
        
        cellIDArr,boundIDArr=__Get_ID_BNoutline_Mat(self.DEM.array)
        subs = np.where(~np.isnan(cellIDArr))
        cellIDVec = cellIDArr[subs]
        sortArr = np.c_[cellIDVec,subs[0],subs[1]]
        sortArr = sortArr[sortArr[:,0].argsort()]
        self.cellSubs = (sortArr[:,1].astype('int32'),sortArr[:,2].astype('int32'))
#        self.cellIDArr = cellIDArr.astype('int')
        subs = np.where(boundIDArr==0) #outline boundary cell
        boundIDVec = boundIDArr[subs]
        sortArr = np.c_[boundIDVec,subs[0],subs[1]]
        self.outlineSubs = (sortArr[:,1].astype('int32'),sortArr[:,2].astype('int32'))
        self.z = self.DEM.array[self.cellSubs]

    
#%% Global operation: divide Grid into sections and return rows on border
    # return sectionRowInd
    def __DivideGrid(self):
        numSection = self.numSection
        demHeader = self.DEM.header
        Z_valid = ~np.isnan(self.DEM.array)
        Z_valid_rowsum = np.sum(Z_valid,axis=1)
        Z_valid_rowCumsum = np.cumsum(Z_valid_rowsum)
        sectionRowInd = np.zeros((numSection,2))
        # divide Z matrix by rows
        rowStart = 0
        exchangeRows = 2
        self.sections = []
        for i in range(numSection):
            numSectionCells=Z_valid_rowCumsum[-1]*(i+1)/numSection
            rowEnd = -1+np.sum(Z_valid_rowCumsum-numSectionCells<=0)
            sectionRowInd[i,0] = rowStart
            if i==numSection: #to cover the last row
                rowEnd = demHeader['nrows']-1
            sectionRowInd[i,1] = rowEnd
            globalSubs = (int(rowStart),int(rowEnd))
            subDemHead = demHeader.copy()
            subDemHead['yllcorner']=demHeader['yllcorner']+(
                                    demHeader['nrows']-1-rowEnd
                                    )*demHeader['cellsize']
            subDemHead['nrows'] = rowEnd-rowStart+1
            subDEM = self.DEM.array[rowStart:rowEnd,:]
            inputFolder = self.inputFolder+'/'+str(i)+'/input'
            sectionObj = inputHiPIMS_MG(globalSubs, subDEM, subDemHead, inputFolder)
            self.sections.append(sectionObj)
            rowStart = rowEnd-(exchangeRows-1)
        return None
                
#%% sub-class definition
class inputHiPIMS_MG(inputHiPIMS):
    """
    object for each section, child class of inputHiPIMS
    Properties:        
        globalSubs: (tuple,int) two numpy array indicating rows and cols of valid
                    cells on global grid
    """
    def __init__(self,globalSubs,demArray,header,inputFolder):
        self.globalSubs = globalSubs
        inputHiPIMS.__init__(self,demArray,header,inputFolder=inputFolder)
#        self.__GetCellSubs(demArray,header)
#%% boundary class definition
class boundary(object):
    """
    object for boundary conditions
    default outline boundary: IO, h and Q are given as constant 0 values 
    Public Properties:
        number: number of boundaries
        type: a list of string 'open','rigid',
                input-output boundary is open boundary with given water depth 
                and/or velocities
        extent: (2-col numpy array) poly points to define the extent of a 
                IO boundary. If extent is not given, then the boundary is the 
                domain outline
        hSources: a two-col numpy array. The 1st col is time(s). The 2nd col is 
                water depth(m)
        hUSources: a two-col numpy array. The 1st col is time(s). The 2nd col is 
                discharge(m3/s) or a three-col numpy array, the 2nd col and the
                3rd col are velocities(m/s) in x and y direction, respectively.   
    Private Properties:
        code: 3-element row vector for each boundary cell
        
    Methods    
        Classify
        Gen3Code: Generate 3-element boundary codes
        CellLocate: fine boundary cells with given extent
    """
    def __init__(self,boundList):
        # convert boundList to a dataframe
        boundDF = pd.DataFrame(boundList)
        # set default outline boundary [0]
        self.types[0] = 'open'
        self.extents[0] = 'outline'
        self.hSources[0] = np.array([[0,0],[1,0]])
        self.hUSources[0] = np.array([[0,0,0],[1,0,0]])
        N = 1
        deleteDefaultBound = False
        for onebound in boundList:
            self.types[N] = onebound['type']
            if 'h' in onebound.keys():
                self.hSources[N] = onebound['h']
            else:
                self.hSources[N] = None
            if 'hU' in onebound.keys():
                self.hUSources[N] = onebound['hU']
            else:
                self.hUSources[N] = None
            if 'polyPoints' in onebound.keys():
                self.extents[N] = onebound['polyPoints']
            else: 
                # No polyPoints means outline boundary is user-defined 
                # The default outline boundary need to be deleted
                self.extents[N] = 'outline'
                deleteDefaultBound = True
            N=N+1
        if deleteDefaultBound is True:
            boundDF.drop()
        self.PrintSummary()
    def PrintSummary(self):
        print('Number of boundaries: '+str(self.number))
        print('Outline boundary: '+self.types[0])
#    def BoundaryClassify(bnMat_outline,demHead,boundList=[]):
    #bnMat = BoundaryClassify(bnMat_outline,demHead,boundList)
    #%get rows and columns of outline bound cells
    # bnMat: nan: invalida cell; -2: non-bound cell; 0: outline cell;
    #               1,~: user-defined IO bound cell on the outline
#        bnMat = bnMat_outline
#        R = demHead
#        Extent = demHead2Extent(R)
#        BoundSubs = np.where(bnMat_outline==0)
#        Bound_Cell_X = R['xllcorner']+(BoundSubs[1]+0.5)*R['cellsize']
#        Bound_Cell_Y = R['yllcorner']+(R['nrows']-BoundSubs[0]-0.5)*R['cellsize']    
#        BoundSubs = np.array([BoundSubs[0],BoundSubs[1]])
#        BoundSubs = np.transpose(BoundSubs)
#        n=1 # sequence number of boundaries
#        for dBound in boundList:        
#            if len(dBound['polyPoints'])==0: #outline boundary
#                polyPoints = makeDiagonalShape(Extent)
#            elif len(dBound['polyPoints'])==2:
#                xyv = dBound['polyPoints']
#                polyPoints = makeDiagonalShape([np.min(xyv[:,0]),
#                                                np.max(xyv[:,0]),
#                                                np.min(xyv[:,1]),
#                                                np.max(xyv[:,1])])
#            else:
#                polyPoints = dBound['polyPoints']
#            
#            poly = mplP.Polygon(polyPoints, closed=True)
#            Bound_Cell_XY = np.array([Bound_Cell_X,Bound_Cell_Y])
#            Bound_Cell_XY = np.transpose(Bound_Cell_XY)
#            ind1 = poly.contains_points(Bound_Cell_XY)
#            row = BoundSubs[ind1,0]
#            col = BoundSubs[ind1,1]
#            bnMat[row,col]=n
#            #BoundNum[ind1,0] = n+1
#            n=n+1
#        return bnMat
        


