# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 12:30:12 2019

@author: Xiaodong Ming
"""

#%% read downloaded NIMROD tar file
import os
import tarfile
import gzip
import nimrod
import numpy as np
import getpass
from datetime import datetime
from myclass import Raster
import matplotlib.pyplot as plt
import ArcGridDataProcessing as AP
#%% download tar data from internet
def downloadNimrodTar(dateStr,localDir=None,cedaUsername=None,cedaPassword=None):
    """
    dateStr: string YYYYMMDD or 
    cedaUsername: ceda username
    cedaPassword: ceda password
    """
    if localDir is None:
        localDir = os.getcwd()+'/'
    # # your username and password on CEDA website    
    if cedaUsername is None:
        
        print('please type your username in CEDA:')
        cedaUsername = input()
    if cedaPassword is None:    
        print('please type your password:')
        cedaPassword = getpass.getpass()

    from ftplib import FTP
    if type(dateStr)!=list:
        dateStr = [dateStr]
    for onedate in dateStr:
        yearStr = onedate[0:4]
        ftp = FTP('ftp.ceda.ac.uk')
        ftp.login(cedaUsername,cedaPassword)
        remoteDir = 'badc/ukmo-nimrod/data/composite/uk-1km/'+yearStr+'/'
        ftp.cwd(remoteDir)
        #files = ftp.nlst()# Get All Files
        fileString = '*'+onedate+'*'
        files = ftp.nlst(fileString)
        # Print out the files
        for file in files:
            print("Downloading..." + file)
            ftp.retrbinary("RETR " + file ,open(localDir + file, 'wb').write)
        ftp.close()

def get_datetime_str_from_tar(tarfileName):
    """Get a list of datetime stored in a nimrod tar file
    """
    datetime_list = []
    tar = tarfile.open(tarfileName)
    members = tar.getmembers()
    for member in members:
        name = member.name.split('_')
        datetime_list.append(name[2])
    tar.close()
    return datetime_list

def read_gridded_data_from_tar(tarfileName, datetimeStr=None):
    """
    input:
        tarfileName: NIMROD tar file name
        datetimeStr: yyyyMMddHHmm('201012310830') the date and time string
            if datetimeStr is not given, output data of all records
    output:
        a Raster object
    """

    if datetimeStr is None:
        datetimeStr = get_datetime_str_from_tar(tarfileName)
    else:
        if type(datetimeStr) is not list:
           datetimeStr = [datetimeStr]
    tar = tarfile.open(tarfileName)
    members = tar.getmembers()
    output_grids = []
    output_datetimes = []
    for member in members:
        name = member.name.split('_')
        name = name[2]
        if name in datetimeStr:
            output_datetimes.append(name)
            fgz = tar.extractfile(member)
            f=gzip.open(fgz,'rb')
            gridObj = nimrod.Nimrod(f)
            array, header, _ = gridObj.extract_asc()
            obj_raster = Raster(array=array, header=header)
            output_grids.append(obj_raster)
            f.close()            
    tar.close()   
    return output_grids, output_datetimes
    
#%% read one grid from NIMROD tar file
def extractOneGridFromNimrodTar(tarfileName,datetimeStr):
    """
    input:
        tarfileName: NIMROD tar file name
        datetimeStr: yyyyMMddHHmm('201012310830') the date and time string
    output:
        a nimrod object
    """
    gridObj = []
    tar = tarfile.open(tarfileName)
    for member in tar.getmembers():
        fgz = tar.extractfile(member)
        #print(member.name)
        if datetimeStr in member.name:
            print(member.name)
            f=gzip.open(fgz,'rb')
        # using nimrod package
            gridObj=nimrod.Nimrod(f)
            f.close()
    tar.close()   
    return gridObj
    
#%% extract data from NIMROD tar file
def getzMatFromNimrodTar(fileName,clipExtent=[]):
    #  clipExtent = (left,right,bottom,top)
    zMatList=[]
    dtStrList = []
    tar = tarfile.open(fileName)
    tar.getmembers()
    for member in tar.getmembers():
        fgz = tar.extractfile(member)
        #print(member.name)
        datetimeStr = member.name[31:31+12]
        f=gzip.open(fgz,'rb')
        # using nimrod package
        gridObj=nimrod.Nimrod(f)
        f.close()
        #del f,fgz,member
        if len(clipExtent)==4:
            gridObj = gridObj.apply_bbox(clipExtent[0],clipExtent[1],clipExtent[2],clipExtent[3])
        zMat,head,zExtent = gridObj.extract_asc()
        zMat[zMat==0]=np.nan #unit: mm/h
        zMatList.append(zMat)
        #del zMat
        dtStrList.append(datetimeStr)       
    tar.close()
    #del tar
    #import gc
    #gc.collect()
    zMatList = [x for _,x in sorted(zip(dtStrList,zMatList))]
    dtStrList.sort()
    return dtStrList,zMatList,head,zExtent
#%% create rainfall mask and rainfall source array for HiPIMS from 
def getRainMaskSource(tarList, demHead, datetimeStart=None, datetimeEnd=None):
    
    """
    INPUTS
    tarList: a list of NIMROD tart files downloaded from badc
    demHead: a dictionary of dem file information
    datetimeStart: string[yyyyMMddHHmm] or a datetime object, the start 
        datetime for rainfall source array. The default is the earliest 
        datetime in the source tar data
    datetimeEnd: string or a datetime object, the end datetime for 
        rainfall source array. The default is the latest datetime in the
        source tar data
    """
    tarList.sort()

    #%define clip extent    
    R = demHead
    left = R['xllcorner']
    right = R['xllcorner']+R['ncols']*R['cellsize']
    bottom = R['yllcorner']
    top = R['yllcorner']+R['nrows']*R['cellsize']
    clipExtent = (left,right,bottom,top)
    #%create rainfall mask grid according to the size of dem
    zMatList = []
    dtStrList = []
    for tarName in tarList:
        dtStr1,zMat1,head,zExtent = getzMatFromNimrodTar(tarName,clipExtent)
        zMatList = zMatList+zMat1
        dtStrList = dtStrList+dtStr1
#    zMatClip,headClip,extentClip = AP.ArraySquareClip(zMatList[0],head,zExtent)
     #mask value start from 0 and increase colum-wise
    maskValue = np.arange(np.size(zMat1[0])).reshape(
                    (head['nrows'],head['ncols']),order='F')
    rainMask= AP.MaskExtraction(maskValue, head, demHead)
    #%create rainfall source array
    zMatList = [x for _,x in sorted(zip(dtStrList,zMatList))]
    dtStrList.sort()
    if type(datetimeStart) is str:
        datetimeStart = datetime.strptime(datetimeStart,'%Y%m%d%H%M')
    elif datetimeStart is None:
        datetimeStart = datetime.strptime(dtStrList[0],'%Y%m%d%H%M')
    if type(datetimeEnd) is str:
        datetimeEnd = datetime.strptime(datetimeEnd,'%Y%m%d%H%M')
    elif datetimeEnd is None:
        datetimeEnd = datetime.strptime(dtStrList[-1],'%Y%m%d%H%M')
    dates_list = [datetime.strptime(oneDate,'%Y%m%d%H%M')-datetimeStart
                  for oneDate in dtStrList]
    timeSeries = [oneInterval.total_seconds() for oneInterval in dates_list]
    zMatSelected = [a.flatten(order='F') for a, b in 
                    zip(zMatList, timeSeries) if b>=0]
    timeSeries = np.array(timeSeries)
    timeSeries = timeSeries[timeSeries>=0]
    rainArray = np.array(zMatSelected)
    rainArray[np.isnan(rainArray)]=0
    rainSource = np.c_[timeSeries,rainArray/3600/1000]
    if datetimeEnd is None: 
        endTime = datetimeEnd-datetimeStart
        endTime = endTime.total_seconds()
        rainSource = rainSource[timeSeries<=endTime,:]
        
    return rainMask, rainSource
    
