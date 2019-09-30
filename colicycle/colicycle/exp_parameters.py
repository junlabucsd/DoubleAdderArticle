"""
This module allows to collect experimental variables from fits 
to data that can then be used as input to simulations
 """
# Author: Guillaume Witz, Biozentrum Basel, 2019
# License: MIT License



import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats

import colicycle.time_mat_operations as tmo
import colicycle.tools_GW as tgw

def load_data(file_to_load, size_scale = 0.065, period=None):
    """Loads a cell cycle dataframe and completes some information
    
    Parameters
    ----------
    file_to_load : str
        path to file to load
    size_scale : float
        pixel to nm scaling
    period : int
        period of dataframe to keep
         
    Returns
    -------
    colidata : Pandas dataframe
        cell cycle dataframe
    """
    
    colidata = pd.read_pickle(file_to_load)

    #scale lengths in microns
    colidata[['DLi','Lb_fit','Ld_fit','Ld','Lb','Ld','Li','Li_fit','Li_old']] \
    =colidata[['DLi','Lb_fit','Ld_fit','Ld','Lb','Ld','Li','Li_fit','Li_old']].applymap(lambda x: x*size_scale)

    #keep only good data
    colidata = colidata[colidata.pearson_log>0.95]
    colidata = colidata[colidata.tau_fit>0]

    #recover mother information
    colidata['mLi'] = colidata.apply(lambda row: tmo.mother_var(row, colidata, 'Li_fit'), axis = 1)
    colidata['mLi_fit'] = colidata.apply(lambda row: tmo.mother_var(row, colidata, 'Li_fit'), axis = 1)
    colidata['mLd_fit'] = colidata.apply(lambda row: tmo.mother_var(row, colidata, 'Ld_fit'), axis = 1)
    colidata['mLb_fit'] = colidata.apply(lambda row: tmo.mother_var(row, colidata, 'Lb_fit'), axis = 1)
    colidata['mtau_fit'] = colidata.apply(lambda row: tmo.mother_var(row, colidata, 'tau_fit'), axis = 1)
    
    colidata['numori_born'] = colidata.Ti.apply(lambda x: 1 if x>=0 else 2)
    
    if period is not None:
        colidata = colidata[colidata.period == period]
    
    return colidata

def calculate_div_ratio(colidata):
    """Calculate the division ratio for an experiment (ratio of daughter lengths)
    
    Parameters
    ----------
    colidata : Pandas dataframe
        cell cycle dataframe
         
    Returns
    -------
    divR : float
        division ratio
    """
    #calculate the std of the division ratio
    ratio = []
    for id in colidata.mother_id.unique():
        if len(colidata[colidata.mother_id==id])==2:
        
            minL = np.min([colidata[colidata.mother_id==id].iloc[0].Lb_fit,colidata[colidata.mother_id==id].iloc[1].Lb_fit])
            maxL = np.max([colidata[colidata.mother_id==id].iloc[0].Lb_fit,colidata[colidata.mother_id==id].iloc[1].Lb_fit])
            ratio.append(maxL/minL)
            ratio.append(minL/maxL)
    divR = np.std(ratio)
    return divR

def calculate_tau_correlation(colidata, field):
    """Calculate mother-daugther correlation for a given variable
    
    Parameters
    ----------
    colidata : Pandas dataframe
        cell cycle dataframe
    field : str
        key of dataframe to consider 
         
    Returns
    -------
    tau_corr : float
        correlation
    """
    temp1 = colidata[(colidata.mother_id>0)][field].rename('tau1')
    temp2 = colidata.loc[colidata[(colidata.mother_id>0)].mother_id][field].rename('tau2')
    temp1.index = np.arange(len(temp1))
    temp2.index = np.arange(len(temp2))

    mother_daughter_tau = pd.concat([temp1,temp2],axis=1,ignore_index=False)
    mother_daughter_tau = mother_daughter_tau[(mother_daughter_tau.tau1<300)&(mother_daughter_tau.tau2<300)]
    tau_corr = scipy.stats.pearsonr(mother_daughter_tau.dropna().tau1,mother_daughter_tau.dropna().tau2)
    return tau_corr


def fit_logn(colidata, field, fit_range):
    """Fit histogram of a variable with a log-normal
    
    Parameters
    ----------
    colidata : Pandas dataframe
        cell cycle dataframe
    field : str
        key of dataframe to consider 
    fit_range : numpy array
        bins to use for histogram
         
    Returns
    -------
    bin_pos : numpy array
        histogram bins positions
    val_bins : numpy array
        histogram values
    res_fit : numpy array
        fit output from scipy.optimize.minimize
    """
    
    valbins, binmean = np.histogram(np.log(colidata[field]).dropna(), bins=fit_range)
    valbins = valbins/np.sum(valbins)*(binmean[1]-binmean[0])
    bin_pos= np.array([0.5*(binmean[x]+binmean[x+1]) for x in range(len(binmean)-1)])

    additional = (bin_pos, valbins)
    res_fit = scipy.optimize.minimize(fun=gauss_single_fit, args=additional,
                                      x0=np.array([np.max(valbins),np.mean(np.log(colidata[field].dropna())),
                                                   np.var(np.log(colidata[field].dropna()))]),method='BFGS')
    
    return bin_pos, valbins, res_fit

def fit_normal(colidata, field, fit_range):
    """Fit histogram of a variable with a gaussian
    
    Parameters
    ----------
    colidata : Pandas dataframe
        cell cycle dataframe
    field : str
        key of dataframe to consider 
    fit_range : numpy array
        bins to use for histogram
         
    Returns
    -------
    bin_pos : numpy array
        histogram bins positions
    val_bins : numpy array
        histogram values
    res_fit : numpy array
        fit output from scipy.optimize.minimize
    """
    valbins, binmean = np.histogram(colidata[field].dropna(), bins=fit_range)
    valbins = valbins/np.sum(valbins)*(binmean[1]-binmean[0])
    bin_pos= np.array([0.5*(binmean[x]+binmean[x+1]) for x in range(len(binmean)-1)])

    additional = (bin_pos, valbins)
    res_fit = scipy.optimize.minimize(fun=gauss_single_fit, args=additional, 
                                                  x0=np.array([np.max(valbins),np.mean(colidata[field].dropna()),
                                                               np.var(colidata[field].dropna())]),method='BFGS')
    
    return bin_pos, valbins, res_fit



def fun_single_gauss(x, A0, x0, sigma):
    """Gaussian distribution
    
    Parameters
    ----------
    x : numpy array
        x values
    A0 : float
        amplitude
    x0 : float
        mean
    sigma : float
        standard dev.
         
    Returns
    -------
     : numpy array
        function values
    """
    return A0*np.exp(-((x-x0)**2)/(2*sigma**2))

def gauss_single_fit(p, *args):
    """Sum squared error
    
    Parameters
    ----------
    p : numpy array
        triplet of values for gaussian (amplitude, mu, sigma)
    args : numpy arrays
        two arrays should be passed here: x and y values of function
         
    Returns
    -------
    nll : float
        sum of squared error
    """
    x,data = args[0], args[1]
    nll = np.sum((fun_single_gauss(x,p[0],p[1],p[2])-data)**2)
    return nll

def correlated_normal(old_val, mu, sigma, rho):
    """Generated correlated gaussian distributions
    
    Parameters
    ----------
    old_val : float
        previous drawn value
    mu: float
        normal mean
    sigma: float
        normal standard dev.
    rho: float
        correlation (0-1)
    
    Returns
    -------
    correlated : float
        new correlated value from gaussian
    """
    x1 = (old_val-mu)/sigma
    x2 = np.random.normal(0,1)
    x3 = rho*x1+np.sqrt(1-rho**2)*x2
    
    correlated = x3*sigma+mu
    return correlated