#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 13:03:25 2019

@author: b4042552
"""
import os
import numpy as np
import pandas as pd
import gzip
import matplotlib.pyplot as plt
import copy
import scipy.signal
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Test(object):
    def __init__(self,name,score):
        self.name = name
        self.score = score
        self.__add_attributes()
        
    def __add_attributes(self):
        self.name_upper = self.name.upper()
        self.score_rate = self.score/100
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
        self.__GetCellSubs(demArray,header)

        if numSection>1:
            self.__DivideGrid()
            self.globalHeader = self.DEM.header
            self.DEM = None # delete global DEM to release system memory
            
    def GenerateBoundaryCondition(self,boundList=None):
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
#%% sub-class definition
class boundary(object):
    """
    object for boundary conditions
    default outline boundary: IO, h and Q are given as constant 0 values 
    Properties:
        number: number of boundaries
        type: a list of string 'open','rigid',
                (input-output boundary is open boundary with water depth or velocity)
        extent: (2-col numpy array) poly points to define the extent of boundary
                if extent is not given, then the boundary is the domain outline
        code: 3-element row vector for each boundary cell
    Methods    
        Classify
        Gen3Code: Generate 3-element boundary codes
        CellLocate: fine boundary cells with given extent
    """
    def __init__(self,boundList):
        # boundary is a dataframe
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
        

#%% To deal with raster data
class raster(object):
    """
    Created on Tue Apr 7 2019
    @author: Xiaodong Ming 
    
    To deal with raster data with a ESRI ASCII or GTiff format  
    
    Properties:
        sourceFile: file name to read grid data
        outputFile: file name to write a raster object
        array: a numpy array storing grid cell values
        header: a dict storing reference information of the grid
        extent: a tuple storing outline limits of the raster (left,right,bottom,top)
        extent_dict: a dictionary storing outline limits of the raster
        projection: (string) the Well-Known_Text (wkt) projection information
    
    Methods(public):
        Write_asc: write grid data into an asc file compressed(.gz) or decompressed 
        To_osgeo_raster: convert this object to an osgeo raster object
        RectClip: clip raster according to a rectangle extent
        Clip: clip raster according to a polygon
        Rasterize: rasterize the shapefile to the raster object and return a bool array
            with Ture value in and on the polygon/polyline
        Mapshow:    
            
    
    Methods(private):
        __header2extent: convert header to extent
        __read_asc
        __map2sub
        __sub2map
        __read_asc
        __read_tif
        
        
        
    """
#    __slots__ = ('sourceFile', 'outputFile',
#                 'array', 'header', 'extent', 'extent_dict', 
#                 'projection') # attributes
#%%======================== initialization function ===========================   
    def __init__(self,sourceFile=None,array=None,header=None,epsg=None,projection=None):
        """
        sourceFile: name of a asc/tif file if a file read is needed
        
        """
        self.sourceFile = sourceFile
        self.projection = projection
        if epsg is not None:
            self.projection = self.__SetWktProjection(epsg)
        else:
            self.projection = None
        if sourceFile is None:
            self.array = array
            self.header = header
            self.sourceFile = 'sourceFile.asc'
        else:
            if sourceFile.endswith('.tif'):
                self.__read_tif() # only read the first band
            else:
                self.__read_asc()

        if isinstance(self.header,dict)==0:
            raise ValueError('header is not a dictionary')
        else:
            # create self.extent and self.extent_dict 
            self.__header2extent()
            
#%%============================= Spatial analyst ==============================
   
    def RectClip(self,clipExtent):
        """
        clipExtent: left,right,bottom,top
        clip raster according to a rectangle extent
        return:
           a new raster object
        """
        new_obj = copy.deepcopy(self)
        X = clipExtent[0:2]
        Y = clipExtent[2:4]
        rows,cols = self.__map2sub(X,Y)
        Xcentre,Ycentre = self.__sub2map(rows,cols)
        xllcorner = min(Xcentre)-0.5*self.header['cellsize']
        yllcorner = min(Ycentre)-0.5*self.header['cellsize']
        # new array
        new_obj.array = self.array[min(rows):max(rows),min(cols):max(cols)]
        # new header
        new_obj.header['nrows'] = new_obj.array.shape[0]
        new_obj.header['ncols'] = new_obj.array.shape[1]
        new_obj.header['xllcorner'] = xllcorner
        new_obj.header['yllcorner'] = yllcorner
        # new extent
        new_obj.__header2extent()
        new_obj.sourceFile = None       
        return new_obj
    
    def Clip(self,mask=None):
        """
        clip raster according to a mask
        mask: 
            1. string name of a shapefile
            2. numpy vector giving X and Y coords of the mask points
        
        return:
            a new raster object
        """
        from osgeo import ogr
        if isinstance(mask, str):
            shpName =  mask
        # Open shapefile datasets        
        shpDriver = ogr.GetDriverByName('ESRI Shapefile')
        shpDataset = shpDriver.Open(shpName, 0) # 0=Read-only, 1=Read-Write
        layer = shpDataset.GetLayer()
        shpExtent = np.array(layer.GetExtent()) #(minX,maxY,maxX,minY)           
        # 1. rectangle clip raster
        new_obj = self.RectClip(shpExtent)
        new_raster = copy.deepcopy(new_obj)                
        indexArray = new_raster.Rasterize(shpDataset)
        arrayClip = new_raster.array
        arrayClip[indexArray==0]=new_raster.header['NODATA_value']
        new_raster.array = arrayClip        
        shpDataset.Destroy()
        
        return new_raster
    
    def Rasterize(self,shpDSName,rasterDS=None):
        """
        rasterize the shapefile to the raster object and return a bool array
            with Ture value in and on the polygon/polyline
        shpDSName: string for shapefilename, dataset for ogr('ESRI Shapefile') object
        
        return numpy array
        """
        from osgeo import gdal,ogr
        if isinstance(shpDSName,str):
            shpDataset = ogr.Open(shpDSName)
        else:
            shpDataset = shpDSName
        layer = shpDataset.GetLayer()
        if rasterDS is None:
            target_ds = self.To_osgeo_raster()
        else:
            target_ds = rasterDS
        gdal.RasterizeLayer(target_ds, [1], layer,burn_values=[-3333])
        indexArray = target_ds.ReadAsArray()
        indexArray[indexArray!=-3333]=0
        indexArray[indexArray==-3333]=1
        target_ds=None
        return indexArray
    
    def Resample(self,newCellsize,method='bilinear'):
        """
        resample the raster to a new cellsize
        newCellsize: cellsize of the new raster
        method: Resampling method to use. Available methods are:
            near: nearest neighbour resampling (default, fastest algorithm, worst interpolation quality).        
            bilinear: bilinear resampling.        
            cubic: cubic resampling.        
            cubicspline: cubic spline resampling.        
            lanczos: Lanczos windowed sinc resampling.        
            average: average resampling, computes the average of all non-NODATA contributing pixels.        
            mode: mode resampling, selects the value which appears most often of all the sampled points.        
            max: maximum resampling, selects the maximum value from all non-NODATA contributing pixels.        
            min: minimum resampling, selects the minimum value from all non-NODATA contributing pixels.        
            med: median resampling, selects the median value of all non-NODATA contributing pixels.        
            q1: first quartile resampling, selects the first quartile value of all non-NODATA contributing pixels.        
            q3: third quartile resampling, selects the third quartile value of all non-NODATA contributing pixels
        """
        cellSize = self.header['cellsize']
        rasterXSize = self.header['ncols']
        newRasterXSize = int(rasterXSize*cellSize/newCellsize)
        rasterYSize = self.header['nrows']
        newRasterYSize = int(rasterYSize*cellSize/newCellsize)
        
        from osgeo import gdal
        g = self.To_osgeo_raster() # get original gdal dataset
        total_obs = g.RasterCount
        drv = gdal.GetDriverByName( "MEM" )
        dst_ds = drv.Create('', g.RasterXSize, g.RasterYSize,1,eType=gdal.GDT_Float32)
        dst_ds.SetGeoTransform( g.GetGeoTransform())
        dst_ds.SetProjection ( g.GetProjectionRef() )
        hires_data = self.array
        dst_ds.GetRasterBand(1).WriteArray ( hires_data )
        
        geoT = g.GetGeoTransform()
        drv = gdal.GetDriverByName( "MEM" )
        resampled_ds = drv.Create('',newRasterXSize, newRasterYSize, 1, eType=gdal.GDT_Float32)

        newGeoT = (geoT[0], newCellsize, geoT[2],
                   geoT[3], geoT[3], -newCellsize)
        resampled_ds.SetGeoTransform(newGeoT )
        resampled_ds.SetProjection (g.GetProjectionRef() )
        resampled_ds.SetMetadata ({"TotalNObs":"%d" % total_obs})

        gdal.RegenerateOverviews(dst_ds.GetRasterBand(1),[resampled_ds.GetRasterBand(1)], method)
    
        resampled_ds.GetRasterBand(1).SetNoDataValue(self.header['NODATA_value'])
        
        new_obj = self.__osgeoDS2raster(resampled_ds)
        resampled_ds = None

        return new_obj
         
#%%=============================Visualization==================================
    #%% draw inundation map with domain outline
    def Mapshow(self,figureName=None,figsize=None,dpi=300,vmin=None,vmax=None,
                cax=True):
        """
        Display raster data without projection
        figureName: the file name to export map,if figureName is empty, then
            the figure will not be saved
        figsize: the size of map
        dpi: The resolution in dots per inch
        vmin and vmax define the data range that the colormap covers
        """
        np.warnings.filterwarnings('ignore')    
        fig, ax = plt.subplots(1, figsize=figsize)
        # draw inundation
        zMat = self.array
#        zMat[zMat==self.header['NODATA_value']]=np.nan
        img=plt.imshow(zMat,extent=self.extent,vmin=vmin,vmax=vmax)
        # colorbar
    	# create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        if cax==True:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(img, cax=cax)
        ax.axes.grid(linestyle='-.',linewidth=0.2)
        # save figure
        if figureName is not None:
            fig.savefig(figureName, dpi=dpi)
            
        return fig,ax
    
    #%% draw velocity (vector) map
    def VelocityShow(self,other,figureName=None,figsize=None,dpi=300,**kw):
        """
        plot velocity map of U and V, whose values stored in two raster
        objects seperately
        """
        X,Y = self.GetXYcoordinate()        
        U = self.array
        V = other.array
        if U.shape!=V.shape:
            raise TypeError('bad argument: header')
        if 'figsize' in kw:
            figsize = kw['figsize']
        else:
            figsize = None
        fig, ax = plt.subplots(1,figsize=figsize)
        plt.quiver(X,Y, U, V)
        ax.set_aspect('equal', 'box')
        ax.tick_params(axis='y',labelrotation=90)
        if figureName is not None:
            fig.savefig(figureName, dpi=dpi)
        return fig,ax
#%%===============================output=======================================
    
    def GetXYcoordinate(self):
        """
        get X and Y coordinates of raster cells
        return xv,yv numpy array with the same size of the raster object
        """
        ny, nx = self.array.shape
        cellsize = self.header['cellsize']
        # coordinate of the centre on the top-left pixel
        x00centre = self.extent_dict['left'] + cellsize/2
        y00centre = self.extent_dict['top'] - cellsize/2
        x = np.arange(x00centre, x00centre+cellsize*nx, cellsize)
        y = np.arange(y00centre, y00centre-cellsize*ny, -cellsize)
        xv, yv = np.meshgrid(x, y)
        return xv,yv
    
    def GetGeoTransform(self):
        """
        get GeoTransform tuple for osgeo raster dataset
        """
        GeoTransform = (self.extent_dict['left'], self.header['cellsize'], 0.0,
                        self.extent_dict['top'], 0.0, -self.header['cellsize'])
        return GeoTransform
    
    def Write_asc(self,outputFile,EPSG=None,compression=False):
        
        """
        write raster as asc format file 
        outputFile: output file name
        EPSG: epsg code, if it is given, a .prj file will be written
        compression: logic, whether compress write the asc file as gz
        """
        if compression:
            if not outputFile.endswith('.gz'):
                outputFile=outputFile+'.gz'        
        self.outputFile = outputFile
        Z = self.array
        header = self.header
        Z[np.isnan(Z)]= header['NODATA_value']
        if not isinstance(header,dict):
            raise TypeError('bad argument: header')
                     
        if outputFile.endswith('.gz'):
            f = gzip.open(outputFile, 'wb') # write compressed file
        else:
            f = open(outputFile, 'wb')
        f.write(b"ncols    %d\n" % header['ncols'])
        f.write(b"nrows    %d\n" % header['nrows'])
        f.write(b"xllcorner    %g\n" % header['xllcorner'])
        f.write(b"yllcorner    %g\n" % header['yllcorner'])
        f.write(b"cellsize    %g\n" % header['cellsize'])
        f.write(b"NODATA_value    %g\n" % header['NODATA_value'])
        np.savetxt(f,Z,fmt='%g', delimiter=' ')
        f.close()
        if EPSG is not None:
            self.__SetWktProjection(EPSG)
        # if projection is defined, write .prj file    
        if self.projection is not None:    
            prjFileName = outputFile
            wkt = self.projection
            if outputFile.endswith('.asc'):
                prjFileName=prjFileName[0:-4]+'.prj'
            elif outputFile.endswith('.asc.gz'):
                prjFileName=prjFileName[0:-7]+'.prj'
                
            prj = open(prjFileName, "w")            
            prj.write(wkt)
            prj.close()
            
        return None
    
    # convert this object to an osgeo raster object
    def To_osgeo_raster(self,filename=None,fileformat = 'GTiff',destEPSG=27700):        
        """
        convert this object to an osgeo raster object, write a tif file if 
            necessary
        filename: the output file name, if it is given, a tif file will be produced
        fileformat: GTiff or AAIGrid
        destEPSG: the EPSG projection code default: British National Grid
        
        return:
            an osgeo raster dataset
            or a tif filename if it is written
        """
        from osgeo import gdal,osr
        if filename is None:
            dst_filename = ''
            driverName = 'MEM'
        else:
            dst_filename = filename
            driverName = fileformat
        if not dst_filename.endswith('.tif'):
            dst_filename = dst_filename+'.tif'
    
        # You need to get those values like you did.
        PIXEL_SIZE = self.header['cellsize']  # size of the pixel...        
        x_min = self.extent[0] # left  
        y_max = self.extent[3] # top
        dest_crs = osr.SpatialReference()
        dest_crs.ImportFromEPSG(destEPSG)
        # create dataset with driver
        driver = gdal.GetDriverByName(driverName)    
        dataset = driver.Create(dst_filename,
            xsize=self.header['ncols'],
            ysize=self.header['nrows'],
            bands=1,
            eType=gdal.GDT_Float32)
    
        dataset.SetGeoTransform((
            x_min,    # 0
            PIXEL_SIZE,  # 1
            0,                      # 2
            y_max,    # 3
            0,                      # 4
            -PIXEL_SIZE))  
    
        dataset.SetProjection(dest_crs.ExportToWkt())
        array = self.array
#        array[array==self.header['NODATA_value']]=np.nan
        dataset.GetRasterBand(1).WriteArray(array)
        dataset.GetRasterBand(1).SetNoDataValue(self.header['NODATA_value'])
        if filename is not None:
            dataset.FlushCache()  # Write to disk.
            dataset = None
            return dst_filename
        else:
            return dataset

#%%=========================== private functions ==============================
    def __osgeoDS2raster(self,ds):
        """
        convert an osgeo dataset to a raster object
        """
        array = ds.ReadAsArray()
        geoT = ds.GetGeoTransform()
        projection = ds.GetProjection()
        left = geoT[0]
        top = geoT[3]
        cellsize = geoT[1]
        nrows = ds.RasterYSize
        ncols = ds.RasterXSize
        xllcorner = left
        yllcorner = top - cellsize*nrows
        NODATA_value = ds.GetRasterBand(1).GetNoDataValue()
        if NODATA_value is None:
            NODATA_value = -9999
        header = {'xllcorner':xllcorner,'yllcorner':yllcorner,
                  'nrows':nrows, 'ncols':ncols,
                  'cellsize':cellsize, 'NODATA_value':NODATA_value}
        newObj = raster(array=array,header=header,projection=projection)
        return newObj
        
    def __header2extent(self):
        """
        To convert header (dict) to a spatial extent of the DEM
        extent: (left,right,bottom,top)
        """
        R = self.header
        left = R['xllcorner']
        right = R['xllcorner']+R['ncols']*R['cellsize']
        bottom = R['yllcorner']
        top = R['yllcorner']+R['nrows']*R['cellsize']
        self.extent = (left,right,bottom,top)
        self.extent_dict = {'left':left, 'right':right, 'bottom':bottom, 'top':top}

    def __map2sub(self,X,Y):
        """
        convert map points to subscripts of a matrix with geo reference header
        X,Y: coordinates in map units
        return
            rows,cols: (int) subscripts of the data matrix
        """
        #x and y coordinate of the centre of the first cell in the matrix
        X = np.array(X)
        Y = np.array(Y)
        header = self.header
        
        x0 = header['xllcorner']+0.5*header['cellsize']
        y0 = header['yllcorner']+(header['nrows']-0.5)*header['cellsize']
        rows = (y0-Y)/header['cellsize'] # row and col number starts from 0
        cols = (X-x0)/header['cellsize']
        if isinstance(rows,np.ndarray):
            rows = rows.astype('int64')
            cols = cols.astype('int64') #.astype('int64')
        else:
            rows = int(rows)
            cols = int(cols)
        return rows,cols

    def __sub2map(self,rows,cols):
        """
        convert subscripts of a matrix to map coordinates 
        rows,cols: subscripts of the data matrix, starting from 0
        return
            X,Y: coordinates in map units
        """
        #x and y coordinate of the centre of the first cell in the matrix
        if not isinstance(rows,np.ndarray):
            rows = np.array(rows)
            cols = np.array(cols)        
        
        header = self.header
        left = self.extent[0] #(left,right,bottom,top)
        top = self.extent[3]
        X = left + (cols+0.5)*header['cellsize']
        Y = top  - (rows+0.5)*header['cellsize']
         
        return X,Y

# read ascii file        
    def __read_asc(self):
        """
        read asc file and return array,header
        if self.sourceFile ends with '.gz', then read the compressed file
        """
        fileName = self.sourceFile
        try:
            fh = open(fileName, 'r')
            fh.close()
        # Store configuration file values
        except FileNotFoundError:
            # Keep preset values
            print('Error: '+fileName+' does not appear to exist')
            return
        # read header
        header = {} # store header information including ncols, nrows,...
        numheaderrows = 6
        n=1
        if fileName.endswith('.gz'):
            # read header
            with gzip.open(fileName, 'rt') as f:                
                for line in f:
                    if n<=numheaderrows:
                        line = line.split(" ",1)
                        header[line[0]] = float(line[1])
                    else:
                        break
                    n = n+1
        else:
            # read header
            with open(fileName, 'rt') as f:            
                for line in f:
                    if n<=numheaderrows:
                        line = line.split(" ",1)
                        header[line[0]] = float(line[1])
                    else:
                        break
                    n = n+1
    # read value array
        array  = np.loadtxt(fileName, skiprows=numheaderrows,dtype='float64')
        array[array == header['NODATA_value']] = float('nan')
        header['ncols']=int(header['ncols'])
        header['nrows']=int(header['nrows'])
        self.array = array
        self.header = header
        prjFile = self.sourceFile[:-4]+'.prj'
        if os.path.isfile(prjFile):
            with open(prjFile, 'r') as file:
                projection = file.read()
            self.projection = projection
        return None

# read GTiff file
    def __read_tif(self):
        """
        read tif file and return array,header
        only read the first band
        """
        from osgeo import gdal
        tifName = self.sourceFile
        ds = gdal.Open(tifName)
        
        ncols = ds.RasterXSize
        nrows = ds.RasterYSize
        geoTransform = ds.GetGeoTransform()
        x_min = geoTransform[0]
        cellsize = geoTransform[1]
        y_max = geoTransform[3]
        xllcorner = x_min
        yllcorner = y_max - nrows*cellsize
        rasterBand = ds.GetRasterBand(1)
        NODATA_value = rasterBand.GetNoDataValue()
        array = rasterBand.ReadAsArray()
        header = {'ncols':ncols, 'nrows':nrows, 
                  'xllcorner':xllcorner, 'yllcorner':yllcorner,
                  'cellsize':cellsize,'NODATA_value':NODATA_value}        
        self.header = header
        self.array = array
        self.projection = ds.GetProjection()
        rasterBand = None
        ds = None
        return None
    def __SetWktProjection(self,epsg_code):
        """
        get coordinate reference system (crs) as Well Known Text (WKT) 
            from https://epsg.io
        epsg_code: the epsg code of a crs, e.g. BNG:27700, WGS84:4326
        return wkt text
        """
        import requests
        # access projection information
        wkt = requests.get('https://epsg.io/{0}.prettywkt/'.format(epsg_code))
        # remove spaces between charachters
        remove_spaces = wkt.text.replace(" ","")
        # place all the text on one line
        output = remove_spaces.replace("\n", "")
        self.projection = output
        return output         

        
