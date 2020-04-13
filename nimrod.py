#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jan 1 2019 based on nimrod.py (for python 2) by 
Richard Thomas 2015 @contributer: Xiaodong Ming 

Extract data from UK Met Office Rain Radar NIMROD image files.

Parse NIMROD format image files, display header data and allow extraction of
raster image to an ESRI ASCII (.asc) format file. A bounding box may be
specified to clip the image to the area of interest. Can be imported as a
Python module or run directly as a command line script.

Author: Richard Thomas
Version: 1.0 (13 April 2015)
Public Repository: https://github.com/richard-thomas/MetOffice_NIMROD

Command line usage:
  python nimrod.py [-h] [-q] [-x] [-bbox XMIN XMAX YMIN YMAX] [infile] [outfile]

positional arguments:
  infile                (Uncompressed) NIMROD input filename
  outfile               Output raster filename (*.asc)

optional arguments:
  -h, --help            show this help message and exit
  -q, --query           Display metadata
  -x, --extract         Extract raster file in ASC format
  -bbox XMIN XMAX YMIN YMAX
                        Bounding box to clip raster data to

Note that any bounding box must be specified in the same units and projection
as the input file. The bounding box does not need to be contained by the input
raster but must intersect it.

Example command line usage:
  python nimrod.py -bbox 279906 285444 283130 290440
    -xq 200802252000_nimrod_ng_radar_rainrate_composite_1km_merged_UK_zip
    plynlimon_catchments_rainfall.asc

Example Python module usage:
    import nimrod
    a = nimrod.Nimrod(open(
        '200802252000_nimrod_ng_radar_rainrate_composite_1km_merged_UK_zip'))
    a.query()
    a.extract_asc(open('full_raster.asc', 'w'))
    a.apply_bbox(279906, 285444, 283130, 290440)
    a.query()
    a.extract_asc(open('clipped_raster.asc', 'w'))

Notes:
  1. Valid for v1.7 and v2.6-4 of NIMROD file specification
  2. Assumes image origin is top left (i.e. that header[24] = 0)
  3. Tested on UK composite 1km and 5km data, under Linux and Windows XP
  4. Further details of NIMROD data and software at the NERC BADC website:
      http://badc.nerc.ac.uk/browse/badc/ukmo-nimrod/   

Copyright (c) 2015 Richard Thomas
(Nimrod.__init__() method based on read_nimrod.py by Charles Kilburn Aug 2008)

This program is free software: you can redistribute it and/or modify
it under the terms of the Artistic License 2.0 as published by the
Open Source Initiative (http://opensource.org/licenses/Artistic-2.0)

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
"""
#%% read downloaded NIMROD tar file
import sys
import struct
import array
import numpy as np
import os
import tarfile
import gzip
import nimrod
import getpass
from datetime import datetime
from Raster import Raster
from spatial_analysis import map2sub, sub2map
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
     #mask value start from 0 and increase colum-wise
    maskValue = np.arange(np.size(zMat1[0])).reshape(
                    (head['nrows'],head['ncols']),order='F')
    rainMask= MaskExtraction(maskValue, head, demHead)
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

#%% zMask = MaskExtraction(maskMat,maskHead,zHead)
def MaskExtraction(maskMat,maskHead,zHead,maskValue=False):
    """
    extract rainfall mask to model domian with its size and resolution
    # zMask = MaskExtraction(maskMat,maskHead,zHead)
    # maskValue=False:mask value is not given in maskMat, so a mask value mask is to be created
    """
    if ~maskValue:
        maskMat = np.arange(np.size(maskMat)).reshape((maskHead['nrows'],maskHead['ncols']),order='F')
    zMask = np.zeros((zHead['nrows'],zHead['ncols']))
    rows_Z,cols_Z = np.where(~np.isnan(zMask))
    X,Y = sub2map(rows_Z,cols_Z,zHead)
    rowsInMask,colsInMask = map2sub(X,Y,maskHead)
    
    # make sure rows and cols of domain scells are inside mask 
    rowsInMask[rowsInMask<0]=0
    colsInMask[colsInMask<0]=0
    rowsInMask[rowsInMask>maskHead['nrows']-1]=maskHead['nrows']-1
    colsInMask[colsInMask>maskHead['ncols']-1]=maskHead['ncols']-1
    
    values = maskMat[rowsInMask,colsInMask] # mask values
    zMask[rows_Z,cols_Z]=values
    return zMask

class Nimrod:
    """Reading, querying and processing of NIMROD format rainfall data files."""

    class RecordLenError(Exception):
        """
        Exception Type: NIMROD record length read from file not as expected.
        """
    
        def __init__(self, actual, expected, location):
            self.message = (
                "Incorrect record length %d bytes (expected %d) at %s."
                % (actual, expected, location))

    class HeaderReadError(Exception):
        """Exception Type: Read error whilst parsing NIMROD header elements."""
        pass
 
    class PayloadReadError(Exception):
        """Exception Type: Read error whilst parsing NIMROD raster data."""
        pass
 
    class BboxRangeError(Exception):
        """
        Exception Type: Bounding box specified out of range of raster image.
        """
        pass
 
 
    def __init__(self, infile):
        """
        Parse all header and data info from a NIMROD data file into this object.
        (This method based on read_nimrod.py by Charles Kilburn Aug 2008)
                
        Args:
            infile: NIMROD file object opened for binary reading
        Raises:
            RecordLenError: NIMROD record length read from file not as expected
            HeaderReadError: Read error whilst parsing NIMROD header elements
            PayloadReadError: Read error whilst parsing NIMROD raster data
        """
        
        def check_record_len(infile, expected, location):
            """
            Check record length in C struct is as expected.
            
            Args:
                infile: file to read from
                expected: expected value of record length read
                location: description of position in file (for reporting)
            Raises:
                HeaderReadError: Read error whilst reading record length
                RecordLenError: Unexpected NIMROD record length read from file
            """
            
            # Unpack length from C struct (Big Endian, 4-byte long)
            try:
                record_length, = struct.unpack(">l", infile.read(4))
            except Exception:
                raise Nimrod.HeaderReadError
            if record_length != expected:
                raise Nimrod.RecordLenError(record_length, expected, location)
        
        
        # Header should always be a fixed length record
        check_record_len(infile, 512, "header start")
        
        try:
            # Read first 31 2-byte integers (header fields 1-31)
            gen_ints = array.array("h")
            gen_ints.fromfile(infile, 31)
            gen_ints.byteswap()
            
            # Read next 28 4-byte floats (header fields 32-59)
            gen_reals = array.array("f")
            gen_reals.fromfile(infile, 28)
            gen_reals.byteswap()
            
            # Read next 45 4-byte floats (header fields 60-104)
            spec_reals = array.array("f")
            spec_reals.fromfile(infile, 45)
            spec_reals.byteswap()
            
            # Read next 56 characters (header fields 105-107)
            characters = array.array("b")
            characters.fromfile(infile, 56)
            
            # Read next 51 2-byte integers (header fields 108-)
            spec_ints = array.array("h")
            spec_ints.fromfile(infile, 51)
            spec_ints.byteswap()
        except Exception:
            infile.close()
            raise Nimrod.HeaderReadError
    
        check_record_len(infile, 512, "header end")

        # Extract strings and make duplicate entries to give meaningful names
        chars = characters.tobytes().decode()
        self.units = chars[0:8].rstrip('\x00')
        self.data_source = chars[8:32].rstrip('\x00')
        self.title = chars[32:55].rstrip('\x00')

        # Store header values in a list so they can be indexed by "element
        # number" shown in NIMROD specification (starts at 1)
        self.hdr_element = [None]           # Dummy value at element 0
        self.hdr_element.extend(gen_ints)
        self.hdr_element.extend(gen_reals)
        self.hdr_element.extend(spec_reals)
        self.hdr_element.extend([self.units])
        self.hdr_element.extend([self.data_source])
        self.hdr_element.extend([self.title])
        self.hdr_element.extend(spec_ints)
        
        # Duplicate some of values to give more meaningful names
        self.nrows = self.hdr_element[16]
        self.ncols = self.hdr_element[17]
        self.n_data_specific_reals = self.hdr_element[22]
        self.n_data_specific_ints = self.hdr_element[23] + 1
            # Note "+ 1" because header value is count from element 109
        self.y_top = self.hdr_element[34]
        self.y_pixel_size = self.hdr_element[35]
        self.x_left = self.hdr_element[36]
        self.x_pixel_size = self.hdr_element[37]
        self.nodata_value = self.hdr_element[38]
        
        # calculate other attributes
        self.__update_atrributes()

        # Read payload (actual raster data)
        array_size = self.ncols * self.nrows
        check_record_len(infile, array_size * 2, "data start")
             
        self.data = array.array("h")
        try:
            self.data.fromfile(infile, array_size)
            self.data.byteswap()
        except Exception:
            infile.close()
            raise Nimrod.PayloadReadError

        check_record_len(infile, array_size * 2, "data end")
        infile.close()
        
    def __update_atrributes(self):
        """
         calculate and update some attributes according to some other attributes
        """
                # Calculate other image bounds (note these are pixel centres)
        self.x_right = (self.x_left + self.x_pixel_size * (self.ncols - 1))
        self.y_bottom = (self.y_top - self.y_pixel_size * (self.nrows - 1))
        self.xllcorner = self.x_left   - self.x_pixel_size/2
        self.yllcorner = self.y_bottom - self.y_pixel_size/2
        
        # Calculate image extent (left, right, bottom, top) defined by Matplotlib
        # extent represents coordinates on the outside edge of bound pixels        
        self.extent = (self.x_left   - self.x_pixel_size/2,
                       self.x_right  + self.x_pixel_size/2,
                       self.y_bottom - self.y_pixel_size/2,
                       self.y_top    + self.y_pixel_size/2)
        
        # create a header for Arcgrid asc file
        self.asc_head = {'ncols':self.ncols,
                         'nrows':self.nrows,
                         'xllcorner':self.xllcorner,
                         'yllcorner':self.yllcorner,
                         'cellsize':self.x_pixel_size,
                         'NODATA_value':self.nodata_value}

    
    def query(self):
        """Print complete NIMROD file header information."""
        
        print("NIMROD file raw header fields listed by element number:")
        print("General (Integer) header entries:")
        for i in range(1, 32):
            print(" ", i, "\t", self.hdr_element[i])
        print("General (Real) header entries:")
        for i in range(32, 60):
            print(" ", i, "\t", self.hdr_element[i])
        print("Data Specific (Real) header entries (%d):"
               % self.n_data_specific_reals)
        for i in range(60, 60 + self.n_data_specific_reals):
            print(" ", i, "\t", self.hdr_element[i])
        print("Data Specific (Integer) header entries (%d):"
               % self.n_data_specific_ints)
        for i in range(108, 108 + self.n_data_specific_ints):
            print(" ", i, "\t", self.hdr_element[i])
        print("Character header entries:")
        print("  105 Units:           ", self.units)
        print("  106 Data source:     ", self.data_source)
        print("  107 Title of field:  ", self.title)    
            
        # Print out info & header fields
        # Note that ranges are given to the edge of each pixel
        print("\nValidity Time:  %2.2d:%2.2d on %2.2d/%2.2d/%4.4d" % (
            self.hdr_element[4], self.hdr_element[5],
            self.hdr_element[3], self.hdr_element[2], self.hdr_element[1]))
        print("Easting range:  %.1f - %.1f (at pixel steps of %.1f)"
               % (self.x_left - self.x_pixel_size / 2,
                  self.x_right + self.x_pixel_size / 2, self.x_pixel_size))
        print("Northing range: %.1f - %.1f (at pixel steps of %.1f)"
               % (self.y_bottom - self.y_pixel_size / 2,
                  self.y_top + self.y_pixel_size / 2, self.y_pixel_size))
        print("Image size: %d rows x %d cols" % (self.nrows, self.ncols))

       
    def apply_bbox(self, xmin, xmax, ymin, ymax):
        """
        Clip raster data to all pixels that intersect specified bounding box.

        Note that existing object data is modified and all header values
        affected are appropriately adjusted. Because pixels are specified by
        their centre points, a bounding box that comes within half a pixel
        width of the raster edge will intersect with the pixel.
        
        Args:
            xmin: Most negative easting or longitude of bounding box
            xmax: Most positive easting or longitude of bounding box
            ymin: Most negative northing or latitude of bounding box
            ymax: Most positive northing or latitude of bounding box
        Raises:
            BboxRangeError: Bounding box specified out of range of raster image
        """
        
        # Check if there is no overlap of bounding box with raster
        if (
                xmin > self.x_right  + self.x_pixel_size / 2 or
                xmax < self.x_left   - self.x_pixel_size / 2 or
                ymin > self.y_top    + self.y_pixel_size / 2 or
                ymax < self.y_bottom - self.x_pixel_size / 2):
            raise Nimrod.BboxRangeError

        # Limit bounds to within raster image
        xmin = max(xmin, self.x_left)
        xmax = min(xmax, self.x_right)
        ymin = max(ymin, self.y_bottom)
        ymax = min(ymax, self.y_top)

        # Calculate min and max pixel index in each row and column to use
        # Note addition of 0.5 as x_left location is centre of pixel
        # ('int' truncates floats towards zero)
        xMinPixelId = int((xmin - self.x_left) / self.x_pixel_size + 0.5)
        xMaxPixelId = int((xmax - self.x_left) / self.x_pixel_size + 0.5)
        
        # For y (northings), note the first data row stored is most north 
        yMinPixelId = int((self.y_top - ymax) / self.y_pixel_size + 0.5)
        yMaxPixelId = int((self.y_top - ymin) / self.y_pixel_size + 0.5)
          
        bbox_data = []
        for i in range(yMinPixelId, yMaxPixelId + 1):
            bbox_data.extend(self.data[i * self.ncols + xMinPixelId:
                                       i * self.ncols + xMaxPixelId + 1])
            
        # Update object where necessary
        self.data = bbox_data
        self.x_right = self.x_left + xMaxPixelId * self.x_pixel_size
        self.x_left += xMinPixelId * self.x_pixel_size
        self.ncols = xMaxPixelId - xMinPixelId + 1
        self.y_bottom = self.y_top - yMaxPixelId * self.y_pixel_size
        self.y_top -= yMinPixelId * self.y_pixel_size
        self.nrows = yMaxPixelId - yMinPixelId + 1
        self.hdr_element[16] = self.nrows
        self.hdr_element[17] = self.ncols
        self.hdr_element[34] = self.y_top
        self.hdr_element[36] = self.x_left
        self.__update_atrributes()
        return self


    def extract_asc(self, fileName=[]):
        """
        Write raster data to an ESRI ASCII (.asc) format file.
        Or output value matrix, head and extent of the raster data
        
        Args:
            fileName: file to be written
        """
        
        # As ESRI ASCII format only supports square pixels, warn if not so
        if self.x_pixel_size != self.y_pixel_size:
            print ("Warning: x_pixel_size(%d) != y_pixel_size(%d)"
                   % (self.x_pixel_size, self.y_pixel_size))
        zData = np.array(self.data)
        zData = zData.reshape((self.nrows,self.ncols))
        head = self.asc_head        
        # if outfile is given, then write a asc file        
        if len(fileName)!=0:
            with open(fileName, 'wb') as f:       
                f.write(b"ncols    %d\n" % head['ncols'])
                f.write(b"nrows    %d\n" % head['nrows'])
                f.write(b"xllcorner    %g\n" % head['xllcorner'])
                f.write(b"yllcorner    %g\n" % head['yllcorner'])
                f.write(b"cellsize    %g\n" % head['cellsize'])
                f.write(b"NODATA_value    %g\n" % head['NODATA_value'])    
                np.savetxt(f,zData,fmt='%g', delimiter=' ')
        # return zData, head, and extent
        zData = zData.astype('float64')
        ind = zData==self.nodata_value
        head['NODATA_value']=-9999
        zData = zData/32 # from 'mm/h*32' to 'mm/h'
        zData[ind]=-9999
        return zData,head,self.extent

#-------------------------------------------------------------------------------
# Handle if called as a command line script
# (And as an example of how to invoke class methods from an importing module)
#-------------------------------------------------------------------------------
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract information and data from a NIMROD format file",
        epilog="""Note that any bounding box must be specified in the same
                  units and projection as the input file. The bounding box
                  does not need to be contained by the input raster but
                  must intersect it.""")
    parser.add_argument("-q", "--query", action="store_true",
                        help="Display metadata")
    parser.add_argument("-x", "--extract", action="store_true",
                        help="Extract raster file in ASC format")
    parser.add_argument('infile', nargs='?', type=argparse.FileType('rb'),
                        default=sys.stdin,
                        help="(Uncompressed) NIMROD input filename")
    parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'),
                        default=sys.stdout,
                        help="Output raster filename (*.asc)")
    parser.add_argument("-bbox", type=float, nargs=4,
                        metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
                        help="Bounding box to clip raster data to")
    args = parser.parse_args()

    if not args.query and not args.extract:
        parser.print_help()
        sys.exit(1)
        
    # Initialise data object by reading NIMROD file
    # (Only trap record length exception as others self-explanatory)
    try:
        rainfall_data = Nimrod(args.infile)
    except Nimrod.RecordLenError as error:
        sys.stderr.write("ERROR: %s\n" % error.message)
        sys.exit(1)
          
    if args.bbox:
        sys.stderr.write(
            "Trimming NIMROD raster to bounding box...\n")
        try:
            rainfall_data.apply_bbox(*args.bbox)
        except Nimrod.BboxRangeError:
            sys.stderr.write("ERROR: bounding box not within raster image.\n")
            sys.exit(1)

    # Perform query after any bounding box trimming to allow sanity checking of
    # size of resulting image
    if args.query:
        rainfall_data.query()
        
    if args.extract:
        sys.stderr.write(
            "Extracting NIMROD raster to ASC file...\n")
        sys.stderr.write(
            "  Outputting data array (%d rows x %d cols = %d pixels)\n"
            % (rainfall_data.nrows, rainfall_data.ncols,
               rainfall_data.nrows * rainfall_data.ncols))
        rainfall_data.extract_asc(args.outfile)
    