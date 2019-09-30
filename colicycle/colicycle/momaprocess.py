"""
This module allows to parse MoMA output into a Pandas dataframe
 """
# Author: Guillaume Witz, Biozentrum Basel, 2019
# License: MIT License


import os
import re
import glob
import numpy as np
import pandas as pd

from skimage.feature import hessian_matrix, hessian_matrix_eigvals, match_template
import scipy

from . import tools_GW as tgw


def addNameToDictionary(d, name, emptystruct):
    if name not in d:
        d[name] = emptystruct

#time of frames is 0 based. Cells on the first frame are born at t = -1
def parse_exported(moma_path):
    """Parse a MoMA output file into a Pandas dataframe
    
    Parameters
    ----------
    moma_path : str
        path to a MoMA output file
    
    
    Returns
    -------
    time_mat : Pandas dataframe
        Each line of the dataframe corresponds to one cell cycle.
        Columns contain information such as pixel limits, birth time etc.
    """
    
    if not os.path.exists(moma_path):
        print('no such file')
        return None
        
    file = open(moma_path, 'r')
    tline = file.readline()
    while not re.search('trackRegionInterval',tline):
        tline = file.readline()
    pixlim = re.findall('(\d+)',tline)
    tracklim = int(pixlim[0])
    
    time_mat={}
    while tline:
        if re.search('id=',tline):
            index = int(re.search('(\d+)',tline).group(0))
            addNameToDictionary(time_mat, index,{})
            addNameToDictionary(time_mat[index], 'tracklim',[])
            addNameToDictionary(time_mat[index], 'born',[])
            addNameToDictionary(time_mat[index], 'genealogy',[])
            addNameToDictionary(time_mat[index], 'pixlim',[])
            addNameToDictionary(time_mat[index], 'pos_GL',[])
            addNameToDictionary(time_mat[index], 'exit_type',[])
            
            time_mat[index]['tracklim'] = tracklim
            time_mat[index]['born'] = int(re.search('(?<=birth_frame=)-*(\d+)',tline).group(0))
            tline = file.readline()
            while re.search('(frame=)|(output=)',tline):
                if re.search('frame=',tline):
                    frame = int(re.search('(?<=frame=)(\d+)',tline).group(0))+1
                    time_mat[index]['genealogy'] =re.search('(?<=genealogy=)([0-9TB]*)',tline).group(0)
                    pix_low = int(re.findall('pixel_limits=\[(\d*),',tline)[0])
                    pix_high = int(re.findall('pixel_limits=\[\d*,(\d*)\]',tline)[0])
                    time_mat[index]['pixlim'].append([pix_low,pix_high])

                    pos_GL = int(re.findall('pos_in_GL=\[(\d*),',tline)[0]);#position from top in GL
                    num_GL = int(re.findall('pos_in_GL=\[\d*,(\d*)\]',tline)[0]);#total cells in GL
                    time_mat[index]['pos_GL'].append([pos_GL,num_GL])
                tline = file.readline()
            if re.search('DIVISION',tline):
                time_mat[index]['exit_type'] = 'DIVISION'
            elif re.search('EXIT',tline):
                time_mat[index]['exit_type'] = 'EXIT'
            elif re.search('USER_PRUNING',tline):
                time_mat[index]['exit_type'] = 'USER_PRUNING'
            elif re.search('ENDOFDATA',tline):
                time_mat[index]['exit_type'] = 'ENDOFDATA'
            time_mat[index]['pixlim'] = np.array(time_mat[index]['pixlim'])
            time_mat[index]['pos_GL'] = np.array(time_mat[index]['pos_GL'])
        else:
            tline = file.readline()
    time_mat = pd.DataFrame(time_mat).T
    return time_mat

def moma_cleanup(time_mat):
    """Clean-up of cell cycle dataframe
    
    Parameters
    ----------
    time_mat : Pandas dataframe
        datframe with cell cycle information
    
    
    Returns
    -------
    time_mat : Pandas dataframe
        cleaned-up datframe with cell cycle information
    """
    
    #remove cell cycles where the cell is at one point at the top of the channel
    time_mat = time_mat[time_mat.pos_GL.apply(lambda x: 1 not in x[:,0])]
    #remove cells born before start of acquisition
    time_mat = time_mat[time_mat.born>0]
    
    return time_mat




                        