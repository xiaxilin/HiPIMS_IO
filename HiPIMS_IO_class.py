#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 10:01:43 2019

@author: Xiaodong Ming
Progress:
    Set boundary conditions done
    Set grid parameters done

"""
import os
import pandas as pd
import scipy.signal
#import copy
import numpy as np
import myclass
import matplotlib.patches as mplP
import warnings                

#%% grid data for HiPIMS input format
class inputHiPIMS(object):
    """
    Properties:
        demFile: string, filename of DEM data
        caseFolder: the absolute path of input folder, e.g. usr/case/0/,usr/case/1/
        cellInd: index of valid cells([0,1,...N],left to right, bottom to top) on DEM grid
        sharedID
        domain: object with
        bound: a boundary object
        outlineSubs: (tuple,int) two numpy array indicating rows and cols of valid
                    cells on a local grid
        cellSubs: (tuple,int) two numpy array indicating rows and cols of valid
                    cells on a local grid from left to right, bottom to top
        cellSubsOnGlobal: (tuple,int) two numpy array indicating rows and cols of valid
                    cells on the global grid
    methods(private):
        
    """
    # set default parameter values
    h0 = 0
    hU0x = 0
    hU0y = 0
    rain_mask = 0
    manning = 0.035
    sewer_sink = 0
    cumulative_depth = 0
    hydraulic_conductivity = 0
    capillary_head = 0
    water_content_diff = 0
    def __init__(self,demArray=None,header=None,numSection=1,demFile=None,caseFolder=None):
        if caseFolder is None:
            caseFolder=os.getcwd()
        self.caseFolder = caseFolder
        self.numSection = numSection
        self.demFile = demFile
        # get row and col index of all cells on DEM grid
        # define attributes:
#        cellSubs,outlineSubs,z(dem vector)
        if demFile is not None:
            demObj = myclass.raster(demFile)
            self.__GetCellSubs(demObj)
        else:
            self.__GetCellSubs(demArray=demArray,header=header)
        # divide model domain to several sections
        self.globalHeader = self.DEM.header
        if numSection>1:
            # each section contains a "HiPIMS_IO_class.inputHiPIMS_MG" object 
            self.__DivideGrid() # create a sections attribute to self           
#            self.DEM = None # delete global DEM to release system memory
        
            
    def SetBoundaryCondition(self,boundList=None,outlinebound='open'):
        """
        create a boundary object for boundary conditions, containing outlineBound,
        a dataframe of boundary type, extent, source data, code, ect...,
        and a boundary subscrpits tuple (boundSubs)
        If the number of section is larger than 1, then a boundary subscrpits 
        tuple (boundSubs_l) based on sub-grids will be created for each section
        outlinebound: 'open' or 'rigid'
        boundList: a list of dict, each dict contain keys (polyPoints,type,h,hU)
            to define a IO boundary's position, type, and Input-Output (IO) sources 
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
        boundObj = boundary(boundList,outlinebound=outlinebound)
        boundObj.LocateBoundaryCells(self.outlineSubs,self.globalHeader)
        self.bound = boundObj
        self.bound.PrintSummary()
        if self.numSection>1:
            head_g = self.globalHeader
            outlineSubs = self.outlineSubs
            for i in range(self.numSection):
                obj_section = self.sections[i]
                head_l = obj_section.DEM.header
                
                outlineSubs_l = self.__CellSubs_Global2Local(outlineSubs,head_g,head_l)
                boundObj = boundary(boundList,outlinebound=outlinebound)
                boundObj.LocateBoundaryCells(outlineSubs_l,head_l)
                obj_section.bound = boundObj
        return None
    
    def SetParameter(self,paraName,paraValue):
        """
        Set the value of one parameter including h0, hU0x, hU0y, manning, 
            rain_mask,rain_source,sewer_sink, 
            cumulative_depth, hydraulic_conductivity, 
            capillary_head, water_content_diff
        All parameter values are given to global grid not local grids. The grid
            parameter values will be divided to local grids in writing 
            process if multiple sections are defined
        """
        if paraName not in ['h0','hU0x','hU0y','rain_mask','manning','sewer_sink',
                        'cumulative_depth', 'hydraulic_conductivity',
                        'capillary_head', 'water_content_diff']:
            raise ValueError('Parameter is not recognized: '+paraName)

        if type(paraValue) is np.ndarray:
            if paraValue.shape!=self.DEM.array.shape:
                raise ValueError('The array of the parameter value should have the same shape with DEM array')
        elif np.isscalar(paraValue) is False:
            raise ValueError('The parameter value must be either a scalar or an numpy array')
        vars(self)[paraName] = paraValue
        return None
                
    def WriteInputFiles(self,fileName):
        """
        Write input files
        fileToCreate: 'all'|'z','h','hU','manning','sewer_sink',
                        'cumulative_depth','hydraulic_conductivity',
                        'capillary_head','water_content_diff'
                        'precipitation_mask','precipitation_source',
                        'boundary_condition','gauges_pos'        
        """
        fileNames = ['DEM','h','hU','boundary_conditions'
                     'precipitation_mask','precipitation_source',
                     'manning','sewer_sink',
                     'cumulative_depth', 'hydraulic_conductivity',
                        'capillary_head', 'water_content_diff']
        if fileName=='h':
            fileTags = 'h0'
        elif fileName=='hU':
            fileTags = 'hU0'
        elif fileName=='DEM':
            fileTags = fileName
        else:
            fileTags = fileNames
        outputArray=self.__Generate_ID_Value_Array(fileTags)
        return outputArray

            
                   
            
#%========================private method==================================
    #%create IO Folders for single GPU
    def __CreateIOFolders(self):
        #dirInput,dirOutput,dirMesh,dirField = CreateIOFolders(folderName)
        folderName = self.caseFolder
        if folderName[-1]!='/':
            folderName = folderName+'/'        
        dirInput = folderName+'input/'
        dirOutput = folderName+'output/'
        if not os.path.exists(dirOutput):
            os.makedirs(dirOutput)
        if not os.path.exists(dirInput):
            os.makedirs(dirInput)
        dirMesh = dirInput+'mesh/'
        if not os.path.exists(dirMesh):
            os.makedirs(dirMesh)
        dirField = dirInput+'field/'
        if not os.path.exists(dirField):
            os.makedirs(dirField)
        return dirInput,dirOutput,dirMesh,dirField
    def __GetCellSubs(self,demObj=None,demArray=None,header=None):
        if demObj is not None:
            self.DEM = demObj
        else:
            self.DEM = myclass.raster(array=demArray,header=header)
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
        subs = np.where(boundIDArr==0) # outline boundary cell
        boundIDVec = boundIDArr[subs]
        sortArr = np.c_[boundIDVec,subs[0],subs[1]]
        self.outlineSubs = (sortArr[:,1].astype('int32'),sortArr[:,2].astype('int32'))
#        self.z = self.DEM.array[self.cellSubs]

    
#% Global operation: divide Grid into sections and return rows on border
    # return sectionRowInd
    def __DivideGrid(self):
        """
        Divide DEM grid to sub grids
        Create objects based on sub-class inputHiPIMS_MG
        """
        
        numSection = self.numSection
        demHeader = self.DEM.header
        Z_valid = ~np.isnan(self.DEM.array)
        Z_valid_rowsum = np.sum(Z_valid,axis=1)
        Z_valid_rowCumsum = np.cumsum(Z_valid_rowsum)
        # divide Z matrix by rows
        rowEnd = demHeader['nrows']-1
        exchangeRows = 2
        splitRows = [] # sub of the split row [0,1,...]  

        sections = []
        sectionNumbers = np.arange(numSection)
#        sectionNumbers = np.flipud(sectionNumbers)
        for i in np.arange(numSection): # from bottom to top
            numSectionCells = Z_valid_rowCumsum[-1]*(i+1)/numSection
            splitRow = np.sum(Z_valid_rowCumsum-numSectionCells>0)
            splitRows.append(splitRow)

        for i in sectionNumbers:
            # from bottom to top
            sectionNO = i # sectionNO start from bottom            
            if sectionNO == sectionNumbers.max(): # top section
                rowStart = 0
            else:
                rowStart = splitRows[i]-exchangeRows
                
            if sectionNO == 0: # bottom section
                rowEnd = demHeader['nrows']-1
            else:
                rowEnd = splitRows[i-1]+exchangeRows-1
            
            subDemArray = self.DEM.array[rowStart:rowEnd+1,:]
            subDemHead = demHeader.copy()
            subDemHead['nrows'] = subDemArray.shape[0]
            yllcorner_new = demHeader['yllcorner']+(demHeader['nrows']-1-rowEnd)*demHeader['cellsize']
            subDemHead['yllcorner'] = yllcorner_new
            ind = np.logical_and(self.cellSubs[0]>=rowStart,self.cellSubs[0]<=rowEnd)
            
            cellSubsOnGlobal = (self.cellSubs[0][ind],self.cellSubs[1][ind])
            rowIndOnGlobal = (rowStart,rowEnd)
            
            caseFolder = self.caseFolder+'/'+str(i)+'/input'
            sectionObj = inputHiPIMS_MG(cellSubsOnGlobal, subDemArray, subDemHead, caseFolder)
            sectionObj.rowIndOnGlobal = rowIndOnGlobal
            #get index of top two rows and bottom two rows
            topH = np.where(sectionObj.cellSubs[0]==0)
            topL = np.where(sectionObj.cellSubs[0]==1)
            bottomH = np.where(sectionObj.cellSubs[0]==sectionObj.cellSubs[0].max()-1)
            bottomL = np.where(sectionObj.cellSubs[0]==sectionObj.cellSubs[0].max())
            sharedCellsID = {'topH':topH,
                                          'topL':topL,
                                          'bottomH':bottomH,
                                          'bottomL':bottomL}
            sectionObj.sharedCellsID=sharedCellsID
            sectionObj.sectionNO = i
            sectionObj.numSection = numSection
            sections.append(sectionObj)
        self.sections = sections
        return None
    
#% Global operation: prepare arrays based on grid values
    def __Generate_ID_Value_Array(self,paraName):
        """
        To generate a array of ID-Value for outputing Z type files
        two col: 1. cell ID, 2. cell value.
        If there are multiple sections, ID_Value_Array will be generated based
        on the local grid.
        Return outputArray 2-col array or a list of 2-col arrays for multi-section
        
        """
        # get the parameter value on the global grid
        paraValue = eval('self.'+paraName)
        globalID = np.arange(self.cellSubs[0].size)
        demShape = (self.globalHeader['nrows'],self.globalHeader['ncols'])
        if np.isscalar(paraValue):
            grid_values = np.zeros(demShape)*0+paraValue
        else: # paraValue is a grid rather than a scalar
            grid_values = paraValue
        # whether for multiple sections
        if self.numSection>1:
            outputArray = []
            for i in np.arange(self.numSection):
                subs = self.sections[i].cellSubsOnGlobal
                vect_values = grid_values[subs]
                ID = np.arange(vect_values.size)
                outputArray.append(np.c_[ID,vect_values])            
        else:
            vect_values = paraValue[self.cellSubs]
            outputArray = np.c_[globalID,vect_values]
        return outputArray

#===================================Static method==============================

    @staticmethod
    def __CellSubs_Global2Local(cellSubs_g,head_g,head_l):
        """
        Convert global cell subs to divided local cell subs and return cellSubs_l
        only rows need to be changed
        cellSubs_g: rows and cols of the global grid
        head_g: head information of the global grid
        head_l: head information of the local grid
        """
        # X and Y coordinates of the centre of the first cell
        y00c_g = head_g['yllcorner']+(head_g['nrows']+0.5)*head_g['cellsize']
        y00c_l = head_l['yllcorner']+(head_l['nrows']+0.5)*head_l['cellsize']
        row_gap = (y00c_g-y00c_l)/head_l['cellsize']
        row_gap = round(row_gap)
        rows = cellSubs_g[0]
        cols = cellSubs_g[1]
        rows = rows-row_gap
        rows = rows.astype(cols.dtype)
        # remove subs out of range of the local DEM
        ind = np.logical_and(rows>=0,rows<head_l['nrows'])
        rows = rows[ind]
        cols = cols[ind]
        cellSubs_l = (rows,cols)
        return cellSubs_l
    
    @staticmethod
    def __Write_ZtypeFile(fileName,id_zV,id_BoundCode):
        # write z type files
        # add a 0.00001 to boundary with IO file
        fmt = ['%d %g']
        if fileName[-4:]!='.dat':
            fileName = fileName+'.dat'
        if fileName[-5:]=='h.dat':        
            ind=id_BoundCode[id_BoundCode[:,1]==3,0]
            ind = ind.astype('uint32')
            if len(ind)>0:
                if np.max(id_zV[ind,1])==0:
                    id_zV[ind,1] = id_zV[ind,1]+0.0001
        if fileName[-6:]=='hU.dat':
            fmt = ['%d %.8e %.8e']
            ind = id_BoundCode[id_BoundCode[:,1]==3,0]   
            ind = ind.astype('uint32')
            if len(ind)>0:
                if np.max(id_zV[ind,1])==0:
                    id_zV[ind,1] = id_zV[ind,1]+0.0001
    
        fmt = '\n'.join(fmt*id_zV.shape[0])
        id_zV_Str = fmt % tuple(id_zV.ravel())
        fmt=['%-12d %2d %2d %2d']
        fmt = '\n'.join(fmt*id_BoundCode.shape[0])
        id_BoundCode_Str = fmt % tuple(id_BoundCode.ravel()) 
        with open(fileName, 'w') as file2write:
            file2write.write("$Element Number\n")
            file2write.write("%d\n" % id_zV.shape[0])
            file2write.write("$Element_id  Value\n")
            file2write.write(id_zV_Str)
            file2write.write("\n$Boundary Numbers\n")
            file2write.write("%d\n" % id_BoundCode.shape[0])
            file2write.write("$Element_id  Value\n") 
            file2write.write(id_BoundCode_Str)
        return None                      
#%% sub-class definition
class inputHiPIMS_MG(inputHiPIMS):
    """
    object for each section, child class of inputHiPIMS
    Properties:
        sectionNO: the serial number of each section
        cellSubs: (tuple,int) two numpy array indicating rows and cols of valid
                    cells on the local grid        
        cellSubsOnGlobal: (tuple,int) two numpy array indicating rows and cols of valid
                    cells on the global grid
        sharedCellsID: 2-row shared Cells ID on a local grid
        caseFolder: input folder of each section
        outlineSubs: (tuple,int) two numpy array indicating rows and cols of valid
                    cells on a local grid
    """
    def __init__(self,cellSubsOnGlobal,demArray,header,caseFolder):
        self.cellSubsOnGlobal = cellSubsOnGlobal
        inputHiPIMS.__init__(self,demArray,header,caseFolder=caseFolder)
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
    def __init__(self,boundList=None,outlinebound = 'open'):
        # convert boundList to a dataframe
#        boundDF = pd.DataFrame(boundList)
        DF = pd.DataFrame(columns=['type','extent','hSources','hUSources','hCode','hUCode'])
        # set default outline boundary [0]
        if outlinebound == 'open':
            DF=DF.append({'type':'open','extent':None,
                   'hSources':np.array([[0,0],[1,0]]),
                   'hUSources':np.array([[0,0,0],[1,0,0]])},ignore_index=True)
        elif outlinebound == 'rigid':
            DF=DF.append({'type':'rigid','extent':None,
                   'hSources':None,
                   'hUSources':None},ignore_index=True)
        else:
            raise ValueError("outlinebound must be either open or rigid!")
                    
        N = 1
        if boundList is None:
            boundList = []
        for onebound in boundList:
            # Only a bound with polyPoints can be regarded as a boundary 
            if 'polyPoints' in onebound.keys() and type(onebound['polyPoints']) is np.ndarray:                
                DF = DF.append({'extent':onebound['polyPoints'],},ignore_index=True)
                DF.type[N] = onebound['type']
                if 'h' in onebound.keys():
                    DF.hSources[N] = onebound['h']
                else:
                    DF.hSources[N] = None
                if 'hU' in onebound.keys():
                    DF.hUSources[N] = onebound['hU']
                else:
                    DF.hUSources[N] = None
                N=N+1
            else:
                warningStr = 'The boundary without polyPoints is ignored: '+str(N-1)
                warnings.warn(warningStr)                

#        Get the three column boundary code
        #default outline boundary
        numBound = DF.shape[0]
        boundInfor = []
        n=0 #sequence of boundary
        m_h=0 #sequence of boundary with IO source files  
        m_hU=0
        for n in range(numBound):
            boundInfor1 = DF.type[n]
            DF.hCode[n] = np.array([2,0,0])
            
            if DF.type[n]=='rigid':
                DF.hUCode[n] = np.array([2,2,0])
            else: #self.type[n]=='open':
                DF.hUCode[n] = np.array([2,1,0])
                if DF.hSources[n] is not None:
                    DF.hCode[n] = np.array([3,0,m_h]) #[3 0 m]
                    boundInfor1 = boundInfor1+', h given'
                    m_h=m_h+1
                if DF.hUSources[n] is not None:
                    DF.hUCode[n] = np.array([3,0,m_hU]) #[3 0 m]
                    boundInfor1 = boundInfor1+', hU given'                    
                    m_hU=m_hU+1
            boundInfor.append(boundInfor1)
        DF['boundInfor'] = boundInfor
        self.df = DF
        self.outlineBound = self.df.type[0]
        if type(self.df.hSources[0]) is np.ndarray:
            self.outlineBound=self.outlineBound+', '+'h Given as zero'
        if type(self.df.hUSources[0]) is np.ndarray:
            self.outlineBound=self.outlineBound+', '+'hU Given as zero'                        
#        self.PrintSummary()
        
    def PrintSummary(self):
        print('Outline boundary: '+self.outlineBound)
        print('Number of boundaries: '+str(self.df.shape[0]))
        for n in range(self.df.shape[0]):
            if self.boundSubs is not None:
                numCells = self.boundSubs[n][0].size
                boundInfor=self.df.boundInfor[n] + ', number of cells: '+str(numCells)
            print(str(n)+'. '+boundInfor)

    def LocateBoundaryCells(self,outlineSubs,demHead):
        """
        To get the subs of boundary cells after
        # BoundaryClassify(bnMat_outline,demHead,boundList)
        # get rows and columns of outline bound cells
        # bnMat: nan: invalida cell; -2: non-bound cell; 0: outline cell;
        #               1,~: user-defined IO bound cell on the outline
        """
        R = demHead
        Extent = myclass.demHead2Extent(demHead)
        Bound_Cell_X = R['xllcorner']+(outlineSubs[1]+0.5)*R['cellsize']
        Bound_Cell_Y = R['yllcorner']+(R['nrows']-outlineSubs[0]-0.5)*R['cellsize']    
        outlineSubs = np.array([outlineSubs[0],outlineSubs[1]]) # convert to numpy array
        outlineSubs = np.transpose(outlineSubs)
        n=1 # sequence number of boundaries
        DF = self.df
        boundSubs = []
        for n in range(DF.shape[0]):        
            if DF.extent[n] is None: #outline boundary
                polyPoints = myclass.makeDiagonalShape(Extent)
            elif len(DF.extent[n])==2:
                xyv = DF.extent[n]
                polyPoints = myclass.makeDiagonalShape([np.min(xyv[:,0]),
                                                np.max(xyv[:,0]),
                                                np.min(xyv[:,1]),
                                                np.max(xyv[:,1])])
            else:
                polyPoints = DF.extent[n]
            
            poly = mplP.Polygon(polyPoints, closed=True)
            Bound_Cell_XY = np.array([Bound_Cell_X,Bound_Cell_Y])
            Bound_Cell_XY = np.transpose(Bound_Cell_XY)
            ind1 = poly.contains_points(Bound_Cell_XY)
            row = outlineSubs[ind1,0]
            col = outlineSubs[ind1,1]
            boundSubs.append([row,col])
        self.boundSubs=boundSubs
        return None