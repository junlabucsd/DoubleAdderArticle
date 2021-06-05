import numpy as np
import pandas as pd
import scipy.stats as sst

import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.colors as mco
import matplotlib.ticker as ticker

from functions_response import make_binning_edges, get_binned

def run_IDA(ngen, params):
    """
    Run simulations of the IDA model.
    """

    # load parameters
    ## initiation size
    mu_i = params['LAi']['mu']
    cv_i = params['LAi']['cv']

    ## division size
    mu_d = params['Sd']['mu']
    cv_d = params['Sd']['cv']


    # compute derived parameters
    s_i = cv_i*mu_i
    s_d = cv_d*mu_d
    mu_ii = 0.5*mu_i
    s_ii = np.sqrt(3.)/2. * s_i
    cv_ii = s_ii/mu_ii
    mu_dd = 0.5*mu_d
    s_dd = np.sqrt(3.)/2. * s_d
    cv_dd = s_dd/mu_dd

    # initial state
    LAi_0 = mu_i
    Sd_m1 = mu_d

    # loop
    LAis = []
    dLAis = []
    Sds = []
    dSds = []

    LAi = LAi_0
    Sd = Sd_m1

    for n in range(ngen):
        LAis.append(LAi)    # appended here to have the forward convention

        dSd = mu_dd * (1+cv_dd*np.random.normal())
        Sd = 0.5*Sd + dSd
        Sds.append(Sd)
        dSds.append(dSd)

        dLAi = mu_ii*(1.+cv_ii*np.random.normal())
        dLAis.append(dLAi)
        LAi = 0.5*LAi + dLAi

    return pd.DataFrame(data={'LAi': LAis, 'dLAi': dLAis, 'Sd': Sds, 'dSd': dSds})


def plot_2varcorr_overlay(dataframes, field_x, field_y, \
        normalize=False, npts_bin=10, binw_dict=None, \
        x0=None, x1 = None, func_slope=None, \
        lw=0.5, ms=2, figsize=None):
    """
    binw:   bin width for Sb (note if normalized is True, then this value must be adjusted accordingly).
    """

    # checks
    ndata = len(dataframes)
    if binw_dict is None:
        binw_dict = [None]*ndata

    if type(binw_dict) == float:
        binw_dict = [binw_dict for n in range(ndata)]

    norm = mco.Normalize(vmin=0, vmax=ndata-1)
    cmap = cm.rainbow

    # make figure
    fig = plt.figure(num=None, facecolor='w', figsize=figsize)
    ax = fig.gca()

    data = [] # place holder for all data
    for n in range(ndata):
        color = cmap(norm(n))
        XY = dataframes[n].loc[:, [field_x, field_y]].dropna().to_numpy().astype('float64')

        if normalize:
            mu = np.nanmean(XY, axis=0)
            XY = XY/mu

            data.append(XY)

        X,Y = XY.T

        # make X binning
        binw = binw_dict[n]
        edges = make_binning_edges(X, x0=x0, x1=x1, binw=binw)
        binw = np.diff(edges)[0]
        print("binw = {:.4f}".format(binw))
        x0 = edges[0]
        x1 = edges[-1]
        hist, edges = np.histogram(X, bins=edges, density=True)
        X_binned = 0.5*(edges[:-1]+ edges[1:])
        nbins = len(X_binned)

        # make Y binning
        Y_binned_sets = get_binned(X, Y, edges)
        Y_binned = np.zeros(nbins)
        Y_counts = np.zeros(nbins)
        Y_vars = np.zeros(nbins)

        for i in range(nbins):
            Yi = Y_binned_sets[i]
            Zi = np.nansum(np.isfinite(Yi),axis=0)
            if (Zi == 0):
                continue
            m = np.nansum(Yi,axis=0) / Zi
            v = np.nansum((Yi-m)**2,axis=0) / Zi
            Y_counts[i] = Zi
            Y_binned[i] = m
            Y_vars[i] = v

        idx = np.isfinite(Y_counts) & (Y_counts > npts_bin)
        X_binned = X_binned[idx]
        Y_binned = Y_binned[idx]
        Y_binned_err = np.sqrt(Y_vars[idx]/Y_counts[idx])

        r = sst.pearsonr(X, Y)[0]
        ax.errorbar(X_binned, Y_binned, yerr=Y_binned_err, color=color, ls='-', marker='o', ms=ms, mfc='w', mec=color, ecolor=color, elinewidth=2*lw, lw=lw, label="r = {:.2f}".format(r))

        if not func_slope is None:
            Xfit = np.array([np.min(X_binned), np.max(X_binned)])
            A = func_slope(dataframes[n])
            Yfit = A*(Xfit-1) + 1
            ax.plot(Xfit,Yfit,'-', lw=2*lw, color=color)

    ax.set_xlabel(field_x, fontsize='large')
    ax.set_ylabel(field_y, fontsize='large')
    ax.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)
    ax.tick_params(length=4)
    if normalize:
        ax.set_xlim(0.5, 1.5)
        ax.set_ylim(0.5, 1.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    else:
        ax.set_xlim(0., None)
        ax.set_ylim(0., None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize='medium', bbox_to_anchor=(1.0, 0.98), frameon=False)

    # figure title
    fig.tight_layout(rect=[0.,0.,1.,0.98])
    return fig
