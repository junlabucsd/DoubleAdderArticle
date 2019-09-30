"""
This module allows to handle the output of a microscopy experiment 
analyzed with the software MoMA. It can parse existing files, help
to load data etc.
 """
# Author: Guillaume Witz, Biozentrum Basel, 2019
# License: MIT License


import os
import glob
import re
import struct
import numpy as np
import pandas as pd
import tifffile as tff

def find_firstfile(directory, file_regexp):
    """Returns the name and path of the first file found within a directory that
    conforms to a given regular expression
    
    Parameters
    ----------
    directory : string
        path of directory
    regepx: string 
        regular expression

    Returns
    -------
    file : string
        first file found to conform within the tree
    filepath: string
        path of file
    """
    file = None
    filepath = None
    found =False
    for root, dirs, files in os.walk(directory,topdown=False):
        for f in files:
            if re.search(file_regexp,f):
                filepath = root
                file = f
                found = True
                break
        if found:
            break
    return file, filepath

class Momaobj:
    """Parsing of MicroManager metadata"""
    def __init__(self, version = None, data_folder = None, rawdata_folder = None,
                 moma_generic =None, tif_generic = None, gl = None, pos = None,
                 gl_list = None, pos_list = None, all_pos=None, all_gl = None, color = 1, time = 0,
                 col_nb = None, height = None, width = None, no_raw = False):
        """Standard __init__ method.

        Parameters
        ----------
        version : int
            moma version (way to save data)
        data_folder : str
            location of cut-out tiff data
        rawdata_folder : str
            location of raw MM images
        moma_generic : str
            generic name of a momoa file
        tif_generic : str
            generic name of a tif file
        gl : str
            current growth lane
        pos : str
            current position
        gl_list : list of str
            list of analysed growth lanes
        pos_list : list of str
            list of analysed positions
        all_gl : list of str
            list of growth lanes
        all_pos : list of str
            list of positions
        color : int
            current color
        time : int
            time point
        col_nb : int
            number of different colors
        height : int
            height of image
        width : int
            width of image
        no_raw : bool
            the original tiffs present in the folder 
        """
                
        self.data_folder = data_folder
        self.rawdata_folder = rawdata_folder
        
        self.get_generics()
        self.get_version()
        
        if no_raw:
            self.get_pos_gl_noraw()
            
        else:
            self.get_pos_gl()
            self.get_pos_gl_analyzed()
        
        
        if self.moma_generic is None:
            self.gl = self.all_gl[0]
            self.pos = self.all_pos[0]
        else:
            self.gl = self.gl_list[0]
            self.pos = self.pos_list[0]
                
            

        self.col = color
        self.time = time
        self.col_nb = col_nb
        
        self.get_dims()
    
    #based on file name find if data are saved as stack or not
    def get_version(self):
        stack_or_not = len(re.findall('.*t\d*_c\d*.*',self.tif_generic))
        self.version = stack_or_not
        
    #going through all MoMA cropped data, find all available positions and growth lanes
    def get_pos_gl(self):
        all_pos_gl = []
        for root, dirs, files in os.walk(self.data_folder,topdown=False):
            if len(dirs)>0:
                for d in dirs:
                    pos_gl = re.findall('.*[pP]os(\d+).*_GL(\d+).*',d)
                    if len(pos_gl)>0:
                        all_pos_gl.append(pos_gl[0])
        self.all_pos = [x[0] for x in all_pos_gl]
        self.all_gl = [x[1] for x in all_pos_gl]
        return all_pos_gl
    
    #if image data are not available, position and growth lanes are taked from MoMA csv files
    def get_pos_gl_noraw(self):
        csvdir = os.path.dirname(self.moma_generic)
        pos_gl = [re.findall('.*[pP]os(\d+).*_GL(\d+).*',d)[0] for d in glob.glob(csvdir+'/*.csv')]
        self.all_pos = [x[0] for x in pos_gl]
        self.all_gl = [x[1] for x in pos_gl]
        
        self.pos_list = self.all_pos.copy()
        self.gl_list = self.all_gl.copy()
    
    #get generic path to MoMA csv and tif files
    def get_generics(self):
        csv_file, csv_path = find_firstfile(self.data_folder,'Exported.*csv')
        tif_file, tif_path = find_firstfile(self.data_folder,'.*tif')
        if csv_path is None:
            self.moma_generic = None
        else:
            self.moma_generic = csv_path+'/'+csv_file
        self.tif_generic = tif_path+'/'+tif_file
    
    #given a position and growth lane get the corresponding MoMA file name
    def get_momapath(self, pos = None, gl = None):
        if not pos:
            pos = self.pos
        if not gl:
            gl = self.gl
        
        new_csv = re.sub('(?<=[pP]os)\d*?(?=[/_])', pos,self.moma_generic)
        new_csv = re.sub('(?<=GL)\d*(?!=\d)', gl,new_csv)
        return new_csv
    
    #given a position and growth lane get the corresponding tif file name
    def get_tifpath(self, pos = None, gl = None):
        if not pos:
            pos = self.pos
        if not gl:
            gl = self.gl
        new_tif = re.sub('(?<=[pP]os)\d*?(?=[/_])', pos,self.tif_generic)
        new_tif = re.sub('(?<=GL)\d*(?!=\d)', gl,new_tif)
        if self.version == 1:
            new_tif = re.sub('(?<=t)\d*(?=_c)', (4-len(str(self.time+1)))*'0'+str(self.time+1),new_tif)
            new_tif = re.sub('(?<=t\d{4}_c)\d*(?=\.tif)', (4-len(str(self.col)))*'0'+str(self.col),new_tif)
        return new_tif
    
    #get a position and growth lane list of analyzed data (MoMA data)
    def get_pos_gl_analyzed(self):
        if self.moma_generic is None:
            self.pos_list = None
            self.gl_list = None
        else:
            all_pos = self.all_pos
            all_gl = self.all_gl
            pos_analyzed = [all_pos[x] for x in range(len(all_pos)) if os.path.exists(self.get_momapath(all_pos[x],all_gl[x]))]
            gl_analyzed = [all_gl[x] for x in range(len(all_gl)) if os.path.exists(self.get_momapath(all_pos[x],all_gl[x]))]
            #gl_analyzed = [x[1] for x in all_pos_gl if os.path.exists(self.get_momapath(x[0],x[1]))]
            self.pos_list = pos_analyzed
            self.gl_list = gl_analyzed
   
    #get maximum available time point in experiment
    def get_max_time(self):
        if self.version == 1:
            maxtime = np.max([int(re.findall('.*\_t(\d*)\_.*',os.path.basename(x))[0]) for x in glob.glob(os.path.dirname(self.tif_generic)+'/*.tif')])-1
        else:
            maxtime = int(len(tff.TiffFile(self.tif_generic).pages)/self.col_nb)-1
        return maxtime
    
    #setter for position and growth lane
    def set_pos_gl(self, pos, gl):
        self.pos = pos
        self.gl = gl
        self.get_dims()
    
    #get image dimensions
    def get_dims(self):
        image = self.load_moma_im()
        self.height = image.shape[0]
        self.width = image.shape[1]
    
    #load a tif file based on current status of object (color, time etc.)
    def load_moma_im(self):
        """Return image at current position, growth lane, color and time point"""
        
        tif_path = self.get_tifpath()
        if self.version == 1:   
            #image = io.imread(tif_path)
            image = tff.imread(tif_path)
        else:
            '''img = Image.open(tif_path)
            img.seek((self.time)*self.col_nb+self.col-1)
            image = img.getdata()
            image = np.reshape(np.array(image),newshape=[img.height,img.width])'''
            image = tff.imread(tif_path,key=(self.time)*self.col_nb+self.col-1)
        return image
    
    #load a time series based on the times numpy array and the current status of object (color, time etc.) 
    def load_moma_timeseries(self, times):
        """Return image at current position, growth lane, color and time point"""
        
        image=np.zeros((self.height, self.width, np.array(times).shape[0]))
        for idx, t in enumerate(times):
            self.time = t
            tif_path = self.get_tifpath()
            if self.version == 1:   
                #image = io.imread(tif_path)
                image[:,:,idx] = tff.imread(tif_path)
            else:
                image[:,:,idx] = tff.imread(tif_path,key=(self.time)*self.col_nb+self.col-1)
        return image