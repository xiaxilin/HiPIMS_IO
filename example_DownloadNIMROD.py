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
