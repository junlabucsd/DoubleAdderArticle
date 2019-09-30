"""
This module allows detect spots in fluorescence microscopy images. The algorithms
are translations of those published in Aguet et al. Dev. Cell 2013. Refer to that publication
and corresponding code for detailed information.
 """
# Author: Guillaume Witz, Biozentrum Basel, 2019
# License: MIT License


import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy
import pandas as pd
from skimage.feature import peak_local_max
from scipy.ndimage.filters import convolve
from scipy.signal import fftconvolve
from scipy.optimize import least_squares

from . import tools_GW as tools

def spot_filter_convfft(image, templ, logfilt, alpha = 0.05, loc_max_dist = 4):
    """Simulate double-adder model
    
    Parameters
    ----------
    image : 2 or 3D numpy array
        image to analyse
    templ: 2 or 3D numpy array
        template image for filtering
    logfilt: 2 or 3D numpy array
        LoG version of the filter
    alpha: float
        significance threshold
    loc_max_dist: float
        minimal distance between spots
    
    Returns
    -------
    spot_result : dict
        dictionary with spot detection information (amplitude, background, filtered image etc.)
    """
    
    if len(image.shape) == 2:
        dim = 2
    else:
        dim = 3
        
    border = ((np.array(logfilt.shape)-1)/2).astype(int)
    if dim==3:
        image = np.pad(image,((border[0],border[0]),(border[1],border[1]),(border[2],border[2])), mode = 'reflect')
    else:
        image = np.pad(image,((border[0],border[0]),(border[1],border[1])), mode = 'reflect')
    
    convmode = 'same'
    image = image.astype(float)
    
    T_s = np.sum(templ)
    T2_s = np.sum(templ**2)
    n = np.size(templ)
    
    ones_mat = np.ones(templ.shape)
    if dim ==3:
        I_s = fftconvolve(image,ones_mat, mode=convmode)[border[0]:-border[0],border[1]:-border[1],border[2]:-border[2]]
        I2_s = fftconvolve(image**2,ones_mat, mode=convmode)[border[0]:-border[0],border[1]:-border[1],border[2]:-border[2]]
        ITconv = fftconvolve(image,templ, mode=convmode)[border[0]:-border[0],border[1]:-border[1],border[2]:-border[2]]
    else:
        I_s = fftconvolve(image,ones_mat, mode=convmode)[border[0]:-border[0],border[1]:-border[1]]
        I2_s = fftconvolve(image**2,ones_mat, mode=convmode)[border[0]:-border[0],border[1]:-border[1]]
        ITconv = fftconvolve(image,templ, mode=convmode)[border[0]:-border[0],border[1]:-border[1]]
        
    
    A = (ITconv-I_s*T_s/n)/(T2_s-T_s**2/n)
    amplitude = A
    
    c=(I_s-A*T_s)/n
    background=c
    
    #statistical analysis
    J = np.column_stack((templ.flatten(), np.ones(np.size(templ))))
    C = np.linalg.inv(J.T@(J))
    f_c = I2_s - 2*c*I_s + n*c**2
    RSS = A**2*T2_s-2*A*(ITconv-c*T_s)+f_c
    RSS[RSS<0]=0
    sigma_e2 = RSS/(n-3)
    sigma_A = np.sqrt(sigma_e2*C[0,0])
    sigma_res = np.sqrt((RSS-(A*T_s+n*c-I_s)/n)/(n-1))
    kLevel = scipy.stats.norm.ppf(1-alpha/2, loc=0, scale=1)
    SE_sigma_c = sigma_res/np.sqrt(2*(n-1))*kLevel
    df2 = (n-1)*(sigma_A**2+SE_sigma_c**2)**2/(sigma_A**4+SE_sigma_c**4)
    scomb = np.sqrt((sigma_A**2+SE_sigma_c**2)/n)
    T = (A-sigma_res*kLevel)/scomb
    mask= scipy.stats.t.cdf(-T,df2)<alpha
    
    #find peaks
    if dim==3:
        imgLoG = fftconvolve(image,logfilt,mode = 'same')[border[0]:-border[0],border[1]:-border[1],border[2]:-border[2]]
    else:
        imgLoG = fftconvolve(image,logfilt,mode = 'same')[border[0]:-border[0],border[1]:-border[1]]
        
    locmax_base = peak_local_max(-imgLoG,min_distance = loc_max_dist, indices = False)
    imgLM = locmax_base*mask
    
    spot_result={'amplitude':A, 'background':c, 'prob_mask':mask, 'logmask':locmax_base, 'mask':imgLM, 'imgLoG': imgLoG}
    
    return spot_result



def get_candidates(filtered, subimage):
    """Gather candidate spots. Remove spots close to border
    or with low amplitude
    
    Parameters
    ----------
    filtered : dict
        output of spot_filter_convfft()
    subimage: 2 or 3D numpy array
        subimage
    
    Returns
    -------
    spot_prop : Pandas dataframe
        dataframe with spot information x, y, (z), amplitude, background
    """
    
    im_size = subimage.shape
    im_middle = int((im_size[1]-1)/2)
    
    medval = np.median(subimage[:,im_middle-5:im_middle+6])
    medvalmask = filtered['amplitude']>0.1*medval
    
    spot_mask = filtered['mask']*medvalmask
    amp_0 = filtered['amplitude'][spot_mask]
    b_0 = filtered['background'][spot_mask]

    spot_coord = np.where(spot_mask)
    spot_coord = np.stack(spot_coord).T
    
    if len(subimage.shape)==3:
        spot_prop = pd.DataFrame(np.c_[spot_coord,amp_0, b_0],columns=['z','x','y','A','b'])
        spot_prop = spot_prop[(spot_prop.z-4>=0)&(spot_prop.z+5<subimage.shape[0])]
        spot_prop = spot_prop[(spot_prop.x-5>=0)&(spot_prop.x+6<subimage.shape[1])]
        spot_prop = spot_prop[(spot_prop.y-5>=0)&(spot_prop.y+6<subimage.shape[2])]
    else:
        spot_prop = pd.DataFrame(np.c_[spot_coord,amp_0, b_0],columns=['x','y','A','b'])
        spot_prop = spot_prop[(spot_prop.x-5>=0)&(spot_prop.x+6<subimage.shape[0])]
        spot_prop = spot_prop[(spot_prop.y-5>=0)&(spot_prop.y+6<subimage.shape[1])]

    return spot_prop


def fit_candidates(subimage, spot_prop, sigmaXY, show_fit = False, fit_type = 'None'):
    """Gather candidate spots. Remove spots close to border
    or with low amplitude
    
    Parameters
    ----------
    subimage : 2 or 3D numpy array
        subimage
    spot_prop : Pandas dataframe
        spot properties, output of get_candidates()
    sigmaXY : float
        approximate spot width
    show_fit : bool
        show plot of the fit
    fit_type : str
        how spots should be fitted (fixed sigma, fixed background etc.)
    
    Returns
    -------
    fit_res : 2d numpy array
        fit results. Each row is one spot. The first columns up
        to and including the antepenultimate are fit output of 
        scipy.optimize.least_squares. The two last columns are 
        square sum error and normalized cross-correlation.
    """
    im_size = subimage.shape
    im_middle = int((im_size[1]-1)/2)
    
    fitbox = [int(np.ceil(4*sigmaXY)),int(np.ceil(4*sigmaXY))]
    xgrid, ygrid = np.meshgrid(range(2*fitbox[0]+1),range(2*fitbox[1]+1),
                                          indexing='ij')
    if fit_type == 'None':
        fit_res = np.zeros((len(spot_prop),5+2))
    elif fit_type == 'B':
        fit_res = np.zeros((len(spot_prop),4+2))
        cstB = np.median(subimage[:,im_middle-5:im_middle+6])
    elif fit_type == 'sigma':
        fit_res = np.zeros((len(spot_prop),4+2))
    elif fit_type == 'sigmaB':
        fit_res = np.zeros((len(spot_prop),3+2))
        cstB = np.median(subimage[:,im_middle-5:im_middle+6])
    
    spot_prop = spot_prop[(spot_prop.x>fitbox[0])&(spot_prop.y>fitbox[1])&(spot_prop.y<im_size[1]-fitbox[1])]
    for x in range(len(spot_prop)):

        spot_image = subimage[int(spot_prop.iloc[x].x)-fitbox[0]:int(spot_prop.iloc[x].x)+fitbox[0]+1,
                              int(spot_prop.iloc[x].y)-fitbox[1]:int(spot_prop.iloc[x].y)+fitbox[1]+1]

        if fit_type == 'sigma':
            param_init = [spot_prop.iloc[x].A, fitbox[0],fitbox[1], spot_prop.iloc[x].b]
            res = least_squares(tools.LSE_gauss2D_cstsigma, param_init, args=(xgrid, ygrid, sigmaXY, spot_image))
            res_abs = np.abs(res.x)
            fitim = tools.fun_gauss2D_cstsigma(xgrid, ygrid,sigmaXY, *res_abs)
        elif fit_type == 'None':
            param_init = [spot_prop.iloc[x].A, fitbox[0],fitbox[1],sigmaXY, spot_prop.iloc[x].b]
            res = least_squares(tools.LSE_gauss2D, param_init, args=(xgrid, ygrid, spot_image))
            res_abs = np.abs(res.x)            
            fitim = tools.fun_gauss2D(xgrid, ygrid,zgrid, *res_abs)
        elif fit_type == 'B':
            param_init = [spot_prop.iloc[x].A, fitbox[0],fitbox[1],sigmaXY]
            res = least_squares(tools.LSE_gauss2D_cstB, param_init, args=(xgrid, ygrid, cstB, spot_image))
            res_abs = np.abs(res.x)
            fitim = tools.fun_gauss2D_cstB(xgrid, ygrid,cstB, *res_abs)
        elif fit_type == 'sigmaB':
            param_init = [spot_prop.iloc[x].A, fitbox[0],fitbox[1]]
            res = least_squares(tools.LSE_gauss2D_cstBsigma, param_init, args=(xgrid, ygrid,cstB,
                                                                               sigmaXY, spot_image))
            res_abs = np.abs(res.x)
            fitim = tools.fun_gauss2D_cstBsigma(xgrid, ygrid,cstB,sigmaXY, *res_abs)
        
        #print(res.x)
        #print(res.cost)
        
      
        vec1 = np.ravel(spot_image)
        vec2 = np.ravel(fitim)
        vec1 = vec1-np.mean(vec1)
        vec2 = vec2-np.mean(vec2)
        vec1 = vec1/np.sqrt(np.sum(vec1**2))
        vec2 = vec2/np.sqrt(np.sum(vec2**2))
        ncc = np.sum(vec1*vec2)
        
        if show_fit:
            res_abs = np.abs(res.x)
            #fitim = tools.fun_gauss3D_cstsigma(xgrid, ygrid,zgrid,sigmaXY, sigmaZ, *res_abs)
            fitim = tools.fun_gauss2D(xgrid, ygrid, *res_abs)
            plt.subplot(1,2,1)
            plt.imshow(np.sum(spot_image,axis = 0))
            plt.subplot(1,2,2)
            plt.imshow(np.sum(fitim,axis = 0))
            plt.show()
           
        fit_res[x,0:-2]= res.x
        fit_res[x,-2]=res.cost
        fit_res[x,-1]=ncc
        
    return fit_res

    
def make_g_filter(modelsigma, modelsigmaZ = None):
    """Create a Gaussian spot model for filtering
    
    Parameters
    ----------
    modelsigma : float
        expected xy standard dev.
    modelsigmaZ: float
        expected z standard dev.
    
    Returns
    -------
    g : 2 or 3D numpy array
        Gaussian filter
    """
    if modelsigmaZ is None:
        ws=round(4*modelsigma)
        x = np.arange(-ws, ws+1, 1)
        y = np.arange(-ws, ws+1, 1)
        xx, yy = np.meshgrid(x, y)
        g = np.exp(-(xx**2+yy**2)/(2*modelsigma**2))
    else:
        ws=round(4*modelsigma)
        wsZ = round(4*modelsigmaZ)
        x = np.arange(-ws, ws+1, 1)
        y = np.arange(-ws, ws+1, 1)
        z = np.arange(-wsZ, wsZ+1, 1)
        xx, yy, zz = np.meshgrid(x, y, z, indexing = 'ij')
        g = np.exp(-(xx**2+yy**2)/(2*modelsigma**2))*np.exp(-(zz**2)/(2*modelsigmaZ**2))
    return g


def make_laplacelog(modelsigma, modelsigmaZ = None):
    """Create a LoG spot model for filtering
    
    Parameters
    ----------
    modelsigma : float
        expected xy standard dev.
    modelsigmaZ: float
        expected z standard dev.
    
    Returns
    -------
    g : 2 or 3D numpy array
        LoG filter
    """
    if modelsigmaZ is None:
        ws=round(4*modelsigma)
        x = np.arange(-ws, ws+1, 1)
        y = np.arange(-ws, ws+1, 1)
        xx, yy = np.meshgrid(x, y)
        g = np.exp(-(xx**2+yy**2)/(2*modelsigma**2))
    else:
        ws=round(4*modelsigma)
        wsZ = round(4*modelsigmaZ)
        x = np.arange(-ws, ws+1, 1)
        y = np.arange(-ws, ws+1, 1)
        z = np.arange(-wsZ, wsZ+1, 1)
        xx, yy, zz = np.meshgrid(x, y, z, indexing = 'ij')
        g = (xx**2/modelsigma**4-1/modelsigma**2+yy**2/modelsigma**4-1/modelsigma**2+zz**2/modelsigmaZ**4-1/modelsigmaZ**2)*np.exp(-(xx**2+yy**2)/(2*modelsigma**2)-(zz**2)/(2*modelsigmaZ**2))
    return g