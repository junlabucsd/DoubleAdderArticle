"""
This module contains a set of helper functions mainly used
for fits.
 """
# Author: Guillaume Witz, Biozentrum Basel, 2019
# License: MIT License


import numpy as np

def locmax(profile, window):
    length = len(profile)
    #window = 5
    locmaxlogic = [((profile[x]>profile[max([0,x-window]):x]).all()) and ((profile[x]>profile[x+1:min([length,x+window+1])]).all()) for x in range(length)]
    return locmaxlogic

def fun_expgrowht2(x, L0, T):
    return L0 * 2**(x/T)
    #return L0 + x*T
    
def fun_double_gauss(x, A0, x0, A1, x1, B):
    return A0*np.exp(-((x-x0)**2)/(2*1.5)**2) + A1*np.exp(-((x-x1)**2)/(2*1.5)**2) +B

def fun_double_gauss_same_amp(x, A0, x0, x1, B ,sigma = 1.0):
    return A0*np.exp(-((x-x0)**2)/(2*sigma)**2) + A0*np.exp(-((x-x1)**2)/(2*sigma)**2) +B

def fun_double_gauss_constr(x, A0, x0, x1, B,r):
    return A0*np.exp(-((x-x0)**2)/(2*1.0)**2) + r*A0*np.exp(-((x-x1)**2)/(2*1.0)**2) +B

def fun_gauss(x, A0, x0, sigma0):
    return A0*np.exp(-((x-x0)**2)/(2*sigma0)**2)

def step_fun(x,x0): 
    return 0.5*(np.sign(x-x0))+1.5

def signGW(x):
    s = x.copy()
    s[x<0] = -1
    s[x>=0] = 1 
    
    return s

def square_wave(x,x0,x1,A): 
    return 0.8*A*0.5*(-signGW(x-x0)*signGW(x-x1)+1)+A

def fun_gauss2D(x,y, A, x0, y0, sigma, B):
    return A*np.exp(-((x-x0)**2)/(2*sigma)**2)*np.exp(-((y-y0)**2)/(2*sigma)**2) +B

def fun_gauss2D_cstB(x,y,B, A, x0, y0, sigma):
    return A*np.exp(-((x-x0)**2)/(2*sigma)**2)*np.exp(-((y-y0)**2)/(2*sigma)**2) +B



def fun_gauss2D_double(x,y, A, x0, y0, sigma, B, A2, x02, y02):
    return A*np.exp(-((x-x0)**2)/(2*sigma)**2)*np.exp(-((y-y0)**2)/(2*sigma)**2) +A2*np.exp(-((x-x02)**2)/(2*sigma)**2)*np.exp(-((y-y02)**2)/(2*sigma)**2) +B

def fun_gauss3D(x,y,z, A, x0, y0,z0, sigma, sigmaZ, B):
    return A*np.exp(-((x-x0)**2)/(2*sigma)**2)*np.exp(-((y-y0)**2)/(2*sigma)**2)*np.exp(-((z-z0)**2)/(2*sigmaZ)**2) +B

def fun_gauss3D_cstB(x,y,z,B, A, x0, y0,z0, sigma,sigmaZ):
    return A*np.exp(-((x-x0)**2)/(2*sigma)**2)*np.exp(-((y-y0)**2)/(2*sigma)**2)*np.exp(-((z-z0)**2)/(2*sigmaZ)**2) +B

def fun_gauss3D_cstBsigma(x,y,z,B,sigma,sigmaZ, A, x0, y0,z0):
    return A*np.exp(-((x-x0)**2)/(2*sigma)**2)*np.exp(-((y-y0)**2)/(2*sigma)**2)*np.exp(-((z-z0)**2)/(2*sigmaZ)**2) +B

def fun_gauss3D_cstsigma(x,y,z,sigma,sigmaZ, A, x0, y0,z0, B):
    return A*np.exp(-((x-x0)**2)/(2*sigma)**2)*np.exp(-((y-y0)**2)/(2*sigma)**2)*np.exp(-((z-z0)**2)/(2*sigmaZ)**2) +B

def LSE_gauss2D(p,x,y,val):
    #p[-1]=80
    residual = fun_gauss2D_cstB(x,y,*p)-val
    return np.ravel(residual)

def LSE_gauss2D_cstB(p,x,y,B,val):
    #p[-1]=80
    residual = fun_gauss2D_cstB(x,y,B,*p)-val
    return np.ravel(residual)

def LSE_gauss3D_cstB(p,x,y,z,B,val):
    #p[-1]=80
    residual = fun_gauss3D_cstB(x,y,z,B,*p)-val
    return np.ravel(residual)

def LSE_gauss3D_cstBsigma(p,x,y,z,B,sigma,sigmaZ,val):
    #p[-1]=80
    residual = fun_gauss3D_cstBsigma(x,y,z,B,sigma,sigmaZ,*p)-val
    return np.ravel(residual)

def LSE_gauss3D_cstsigma(p,x,y,z,sigma,sigmaZ,val):
    #p[-1]=80
    residual = fun_gauss3D_cstsigma(x,y,z,sigma,sigmaZ,*p)-val
    return np.ravel(residual)


def MLE_gauss2D_fit(p, *args):
    x,y, data = args[0], args[1],args[2]
    nll = np.sum(fun_gauss2D(x,y,p[0],p[1],p[2],p[3],p[4])) - np.sum(data*np.log(fun_gauss2D(x,y,p[0],p[1],p[2],p[3],p[4])))
    #nll = np.sum(fun_gauss2D(x,y,p[0],p[1],p[2],2.5,p[4])) - np.sum(data*np.log(fun_gauss2D(x,y,p[0],p[1],p[2],2.5,p[4])));
    return nll  

def MLE_gauss2D_double_fit(p, *args):
    x,y, data = args[0], args[1],args[2]
    nll = np.sum(fun_gauss2D_double(x,y,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7])) - np.sum(data*np.log(fun_gauss2D_double(x,y,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7])))
    #nll = np.sum(fun_gauss2D_double(x,y,p[0],p[1],p[2],2.5,p[4],p[5],p[6],p[7])) - np.sum(data*np.log(fun_gauss2D_double(x,y,p[0],p[1],p[2],2.5,p[4],p[5],p[6],p[7])))
    return nll  

def gauss2D_double_fit(p, *args):
    x,y, data = args[0], args[1],args[2]
    nll = np.sum((fun_gauss2D_double(x,y,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7])-data)**2)
    return nll 


def rad_time_fit(x,rad,tau_c,tau_d):
    x0 = x[x<=tau_c]
    x1 = x[x>tau_c]
    y0 = rad*np.ones(x0.shape[0])
    y1 = rad*(1-((x1-tau_c)/(tau_d-tau_c))**2)**(1/2)
    y1[np.isnan(y1)]=1000#np.inf
    y = np.concatenate((y0,y1))
    return y

def lnlike(p, x, y):
    rad, tau_c,tau_d,eps = p
    model = rad_time_fit(x,rad, tau_c,tau_d)
    # the likelihood is sum of the lot of normal distributions
    denom = eps**2
    lp = -0.5*np.sum(((y - model)**2)/denom+ np.log(denom)+ np.log(2*np.pi))
    return lp

def rad_time_alpha_fit(x,rad,tau_c,tau_d,alpha):
    x0 = x[x<=tau_c]
    x1 = x[x>tau_c]
    y0 = rad*np.ones(x0.shape[0])
    y1 = rad*(1-((x1-tau_c)/(tau_d-tau_c))**alpha)**(1/alpha)
    y1[np.isnan(y1)]=1000#np.inf
    y = np.concatenate((y0,y1))
    return y

def lnlike_alpha(p, x, y):
    rad, tau_c,tau_d,eps, alpha = p
    model = rad_time_alpha_fit(x,rad, tau_c,tau_d, alpha)
    # the likelihood is sum of the lot of normal distributions
    denom = eps**2
    lp = 0.5*np.sum(((y - model)**2)/denom+ np.log(denom)+ np.log(2*np.pi))
    return lp    