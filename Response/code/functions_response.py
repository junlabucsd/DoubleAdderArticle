# Guillaume Le Treut, UC San Diego
# gletreut@ucsd.edu
# April 2020.

############################################################################
# imports
############################################################################
import os
import copy
import pickle as pkl
import numpy as np
import scipy.stats as sst
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.colors as mco
import matplotlib.ticker

############################################################################
# functions
############################################################################

### helper methods ###

def histogram(X,density=True):
    valmax = np.max(X)
    valmin = np.min(X)
    iqrval = sst.iqr(X)
    nbins_fd = (valmax-valmin)*np.float_(len(X))**(1./3)/(2.*iqrval)
    if (nbins_fd < 1.0e4):
        return np.histogram(X,bins='auto',density=density)
    else:
        return np.histogram(X,bins='sturges',density=density)

def make_binning_edges(X, x0=None, x1=None, binw=None):
    if x0 is None:
        x0 = np.min(X)
    if x1 is None:
        x1 = np.max(X)

    nx = len(X)
    if binw is None:
        nbins = np.ceil(np.sqrt(nx))
        binw = (x1-x0)/nbins

    nbins = float(x1-x0)/binw
    nbins = int(np.ceil(nbins))
    x1 = x0 + nbins*binw
    edges = np.arange(nbins+1)*binw + x0

    return edges

def get_binned(X, Y, edges):
    nbins = len(edges)-1
    digitized = np.digitize(X,edges)
    Y_subs = [None for n in range(nbins)]
    for i in range(1, nbins+1):
        Y_subs[i-1] = np.array(Y[digitized == i])

    return Y_subs

def autolabel_vertical(ax, rects, fontsize='small', fmt_str='{:.2f}'):
    """
    Attach a text label above each bar in *rects*, displaying its height.
    From: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """

    for rect in rects:
        height = rect.get_height()
        ax.annotate(fmt_str.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=fontsize)

def autolabel_horizontal(ax, rects, fontsize='small', fmt_str='{:.2f}'):
    """
    Attach a text label to the right each bar in *rects*, displaying its width.
    """

    for rect in rects:
        width = rect.get_width()
        ax.annotate(fmt_str.format(width),
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(3, 0),  # 3 points horizontal offset
                    textcoords="offset points",
                    ha='left', va='center', fontsize=fontsize)

def get_mean_std_logn(mu, std):
    """
    Return the mean and standard deviation of a log-normal distribution.
    INPUT:
      mu: mean of the Gaussian distribution of the log-variable.
      std: standard deviation of the Gaussian distribution of the log-variable.
    """
    m = np.exp(mu+0.5*std**2)
    s = m*np.sqrt(np.exp(std**2)-1.)
    return m,s

### Model check ###

def plot_model_check(data_dict, binw_dict=None, \
                    fig=None, lw=0.5, ms=2, alpha=0.2 ,fig_title=None):

    # parameters and input
    df = data_dict['df']
    color = data_dict['color']
    nori_initial = data_dict['nori_init']

    # conditionning
    data_format = data_dict['format']
    if data_format == 'SIM':
        df = process_df_SIM(df)
    elif data_format == 'EXP2':
        df = process_df_EXP2(df)
    else:
        raise ValueError("Data format must be \'SIM\' or \'EXP2\'!")

    # extraction
    columns = ['Lb', 'Ld', 'LAi', 'dLi', 'dLdLi', 'mother_id', 'rfact']
    df = copy.deepcopy(df.loc[:, columns])
    if len(df.dropna()) == 0:
        raise ValueError("Problem with one of the columns")

    for key in ['mLd', 'mLAi']:
        columns.append(key)
        df[key] = np.nan
    for cid in df.index:
        mid = df.at[cid,'mother_id']
        if mid in df.index:
            df.at[cid, 'mLd'] = df.at[mid, 'Ld']
            df.at[cid, 'mLAi'] = df.at[mid, 'LAi']

    # prepare data
    keys = ['delta_ii', 'Lambda_i', 'delta_id', 'Ld', 'rfact']
    if binw_dict is None:
        binw_dict = {key: None for key in keys}
    else:
        for key in keys:
            if not key in binw_dict.keys():
                binw_dict[key] = None
    # name data
    data = df.loc[:, columns].dropna().to_numpy().astype('float64')
    Lb, Ld, LAi, delta_ii, delta_id, mid, rfact, mLd, mLAi = data.T


    #################
    # INITIATION SIZE
    #################
    ## prediction
    LAi_pred = 0.5*mLAi + delta_ii
    x0 = np.min(LAi)
    x1 = np.max(LAi)
    LAi_fit = np.linspace(x0,x1,1000)
    ## binned
    binw = binw_dict['Lambda_i']
    edges = make_binning_edges(mLAi, x0=None, x1=None, binw=binw)
    X_binned = 0.5*(edges[:-1] + edges[1:])
    nbins = len(edges)-1

    Y_binned_sets = get_binned(mLAi,LAi,edges)
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
    idx = np.isfinite(Y_counts) & (Y_counts > 2)
    mLAi_binned = X_binned[idx]
    LAi_binned = Y_binned[idx]
    LAi_binned_err = np.sqrt(Y_vars[idx]/Y_counts[idx])

    #################
    # INITIATION SIZE
    #################
    ## prediction
    Ld_pred = nori_initial *(LAi + 2*delta_id)
    x0 = np.min(Ld)
    x1 = np.max(Ld)
    Ld_fit = np.linspace(x0,x1,1000)
    ## binned
    binw = binw_dict['Ld']
    edges = make_binning_edges(mLd, x0=None, x1=None, binw=binw)
    X_binned = 0.5*(edges[:-1] + edges[1:])
    nbins = len(edges)-1

    Y_binned_sets = get_binned(mLd,Ld,edges)
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
    idx = np.isfinite(Y_counts) & (Y_counts > 2)
    mLd_binned = X_binned[idx]
    Ld_binned = Y_binned[idx]
    Ld_binned_err = np.sqrt(Y_vars[idx]/Y_counts[idx])

    # distributions
    dists = {}
    for key,X,label in [ ['delta_ii', delta_ii,"$\delta_{ii}$"], ['Lambda_i', LAi, "$\Lambda_{i}$"], ['delta_id', delta_id, "$\delta_{id}$"], ['Ld', Ld, "$L_d$"] ]:
        mydict = {}
        binw = binw_dict[key]
        #edges = make_binning_edges(X, x0=None, x1=None, binw=binw)
        #hist, edges = histogram(X, bins=edges, density=False)
        hist, edges = histogram(X, density=False)
        mydict['color'] = 'k'
        mydict['xlabel'] = label
        mydict['hist'] = hist
        mydict['edges'] = edges
        mydict['mean'] = np.nanmean(X)
        mydict['std'] = np.nanstd(X)
        mydict['CV'] = mydict['std'] / mydict['mean']
        dists[key] = mydict

    # specific things
    ## colors
    dists['delta_ii']['color'] = 'darkblue'
    dists['delta_id']['color'] = 'darkblue'
    ## fits
    mu_ii = dists['delta_ii']['mean']
    s_ii = dists['delta_ii']['std']
    mu_id = dists['delta_id']['mean']
    s_id = dists['delta_id']['std']
    acf_i = 0.5
    #print("s_id = ", s_id, "s_ii = ", s_ii)
    acf_d = 0.5 / (1. + 3.*s_id**2/s_ii**2)
#    gaussfit = {'mean': mu_ii, 'std': s_ii}
#    dists['delta_ii']['fit'] = gaussfit
    gaussfit = {'mean': 2*mu_ii, 'std': 2./np.sqrt(3) * s_ii, 'acf': acf_i}
    dists['Lambda_i']['fit'] = gaussfit
    gaussfit = {'mean': nori_initial*2*(mu_ii+mu_id), 'std': nori_initial*2.*np.sqrt(s_ii**2 / 3. + s_id**2), 'acf': acf_d}
    dists['Ld']['fit'] = gaussfit

    # make the plot
    nrows = 5
    ncols = 4
    if fig is None:
        figsize=(2*ncols,2*nrows)
        fig = plt.figure(num='none', facecolor='w',figsize=figsize)
    gs = mgs.GridSpec(nrows, ncols, hspace=0.1)

    axes = []

    # Li
    ax = fig.add_subplot(gs[0:2,0:2])
    axes.append(ax)
    ax.plot(LAi,LAi_pred,'ko', ms=ms)
    ax.plot(LAi_fit, LAi_fit, 'r--', lw=lw)
    ax.set_aspect('equal')
    xlabel='$\Lambda_i$'
    ylabel='$\\frac{1}{2}\Lambda_i^{(n-1)} + \delta_{ii}$'
    ax.set_xlabel(xlabel, fontsize='large')
    ax.set_ylabel(ylabel, fontsize='large')
    ax.set_title("Initiation size per ori", fontsize='large')
    ax.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Ld
    ax = fig.add_subplot(gs[0:2,2:4])
    axes.append(ax)
    ax.plot(Ld,Ld_pred,'ko', ms=ms)
    ax.plot(Ld_fit, Ld_fit, 'r--', lw=lw)
    ax.set_aspect('equal')
    xlabel='$L_d$'
    ylabel='$N_\\mathrm{initial} \left( \Lambda_i + 2 \delta_{id} \\right)$'
    ax.set_xlabel(xlabel, fontsize='large')
    ax.set_ylabel(ylabel, fontsize='large')
    ax.set_title("Division size", fontsize='large')
    ax.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # distributions
    ndists = len(dists)
    ax = fig.add_subplot(gs[2,0])
    axes.append(ax)
    ax0 = axes[-1]
    for i in range(1,ndists):
        ax = fig.add_subplot(gs[2,i], sharey=axes[-i])
        axes.append(ax)

    for i,key in enumerate(['delta_ii','Lambda_i','delta_id','Ld']):
        dist = dists[key]
        hist = dist['hist']
        edges = dist['edges']
        mu = dist['mean']
        std = dist['std']
        xlabel = dist['xlabel']
        color = dist['color']
        label = "$\mu = {:.3f}$\nCV = {:.0f} %".format(mu, std/mu*100)

        ax = axes[-ndists+i]

        ax.plot(0.5*(edges[:-1]+ edges[1:]), hist, '-', lw=lw, color=color)
        # fit
        if 'fit' in dist.keys():
            mu_fit = dist['fit']['mean']
            std_fit = dist['fit']['std']
            x0 = np.min(edges)
            x1 = np.max(edges)
            Xfit = np.linspace(x0,x1,1000)
            delta = Xfit[1]-Xfit[0]
            Yfit = np.exp(-0.5*(Xfit-mu_fit)**2/std_fit**2)
            Z = np.sum(Yfit)*delta
            delta = edges[1]-edges[0]
            Z /= np.sum(hist)*delta
            Yfit /= Z
            ax.plot(Xfit, Yfit, 'r--', lw=lw)

        ax.annotate(label, xy=(1.,0.98), xycoords='axes fraction', va='top', ha='right', fontsize='medium')
        ax.set_xlabel(xlabel, fontsize='large')

        ax.tick_params(bottom=True, left=False, labelbottom=True, labelleft=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        if (i == 0):
            ax.tick_params(left=True, labelleft=True)
            ax.spines['left'].set_visible(True)

    # added sizes
    ## Li
    ax = fig.add_subplot(gs[3:5,0:2])
    axes.append(ax)
    ax.plot(mLAi,LAi,'ko', ms=ms, alpha=0.2)
    ax.errorbar(mLAi_binned, LAi_binned, yerr=LAi_binned_err, color='k', marker='s', ms=2*ms, ecolor='k', elinewidth=4*lw, lw=lw)
    acf = dists['Lambda_i']['fit']['acf']
    acf_sim = sst.pearsonr(mLAi,LAi)[0]
    ax.plot(LAi_fit, acf*LAi_fit + (1-acf)*dists['Lambda_i']['mean'] , 'r--', lw=2*lw, label='pred')
    ax.annotate("$\\rho = {:.2f}$\n$\\rho_\mathrm{{pred}} = {:.2f}$".format(acf_sim, acf), xy=(1.,0.02), xycoords='axes fraction', va='bottom', ha='right', fontsize='large')
    ax.set_aspect('equal')
    xlabel='$\Lambda_i^{(n-1)}$'
    ylabel='$\Lambda_i^{(n)}$'
    ax.set_xlabel(xlabel, fontsize='large')
    ax.set_ylabel(ylabel, fontsize='large')
    ax.set_title("Initiation correlation", fontsize='large')
    ax.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize='medium')

    ## Ld
    ax = fig.add_subplot(gs[3:5,2:4])
    axes.append(ax)
    ax.plot(mLd,Ld,'ko', ms=ms, alpha=0.2)
    acf = dists['Ld']['fit']['acf']
    acf_sim = sst.pearsonr(mLd,Ld)[0]
    ax.errorbar(mLd_binned, Ld_binned, yerr=Ld_binned_err, color='k', marker='s', ms=2*ms, ecolor='k', elinewidth=4*lw, lw=lw)
    ax.plot(Ld_fit, acf*Ld_fit + (1-acf)*dists['Ld']['mean'] , 'r--', lw=2*lw, label='pred')
    ax.annotate("$\\rho = {:.2f}$\n$\\rho_\mathrm{{pred}} = {:.2f}$".format(acf_sim, acf), xy=(1.,0.02), xycoords='axes fraction', va='bottom', ha='right', fontsize='large')
    ax.set_aspect('equal')
    xlabel='$L_d^{(n-1)}$'
    ylabel='$L_d^{(n)}$'
    ax.set_xlabel(xlabel, fontsize='large')
    ax.set_ylabel(ylabel, fontsize='large')
    ax.set_title("Division correlation", fontsize='large')
    ax.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize='medium')

    # formatting
    for i, ax in enumerate(axes):
        ax.tick_params(length=4)
#         ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    # figure title
    if not fig_title is None:
        fig.suptitle(fig_title,fontsize='x-large', x=0.5, ha='center')
    gs.tight_layout(fig, rect=[0.,0.,1.,0.93])
    return fig

### Replication origin homeostasis  re-analysis ###
def plot_replication_origins(df_dict, fig=None, lw=0.5, ms=2, ylim=None, fig_title=None):
    """
    Method used to plot information of replication origin homeostasis in simulations.

    """
    nsim = len(df_dict)

    # MAKE THE FIGURE
    if fig is None:
        figsize=(3*nsim,3*2)
        fig = plt.figure(num='none', facecolor='w',figsize=figsize)

    axes = []
    gs = mgs.GridSpec(2, nsim, hspace=0.1)
    for i in range(2):
        row = []
        ax = fig.add_subplot(gs[i,0])
        row.append(ax)
        for j in range(1,nsim):
            ax = fig.add_subplot(gs[i,j], sharey=row[0])
            row.append(ax)
        axes.append(row)
    for i in range(len(axes)):
        for j in range(nsim):
            ax = axes[i][j]
            ax.tick_params(length=4, bottom=True, left=True, labelbottom=True, labelleft=True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if (j > 0):
                ax.spines['left'].set_visible(False)
                ax.tick_params(left=False, labelleft=False)

    # make the plot
    noris = list(df_dict.keys())
    noris.sort()
    data_all = []
    for n in range(nsim):
        nori = noris[n]
        df = df_dict[nori]['df']

        columns = ['numori_born']
        X = df['numori_born'].dropna().to_numpy()
        data_all.append(X)
    numori_max = np.max(np.concatenate(data_all))
    pmax = int(np.ceil(np.log2(numori_max))) + 1
    ticks = [2**p for p in range(pmax+1)]

    for n in range(nsim):
        nori = noris[n]
        df = df_dict[nori]['df']
        color = df_dict[nori]['color']

        columns = ['born', 'numori_born', 'Ld_fit']
        data = df.loc[:,columns].dropna().to_numpy()
        Tb, X, Ld = data.T

        # plot the histogram
        bins = np.arange(2**pmax+1)
        hist, edges = np.histogram(X, bins=bins, density=False)
        ax = axes[0][n]
        ax.bar(edges[:-1], hist, np.diff(edges), color=color)
        ax.set_xlabel('# oriC at birth', fontsize='medium')
        ax.set_xticks(ticks)
        if (n == 0):
            ax.set_ylabel('cell count', fontsize='medium')
        ax.set_title("nori_init = {:d}".format(nori), fontsize='medium')


        # plot time trace
        ax = axes[1][n]
        ax.plot(Tb, X, 'o', ms=ms, color=color)
        ax.set_xlabel('time', fontsize='medium')
        if (n == 0):
            ax.set_ylabel('# oriC at birth', fontsize='medium')
            ax.set_yticks(ticks)


    # EXIT
    fig.suptitle(fig_title, fontsize='x-large', x=0.5, ha='center')
    gs.tight_layout(fig, rect=[0.,0.,1.,0.95])
    return fig

def plot_overlay_sd(df_dict, fig=None, lw=0.5, ms=2, ylim=None, fig_title=None):
    """
    Method used to plot an overlay of the distribution of cell size at division

    """
    nsim = len(df_dict)

    # MAKE THE FIGURE
    if fig is None:
        figsize=(4,4)
        fig = plt.figure(num='none', facecolor='w',figsize=figsize)

        ax = fig.gca()
        ax.tick_params(length=4, bottom=True, left=True, labelbottom=True, labelleft=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # make the plot
    noris = list(df_dict.keys())
    noris.sort()
    for n in range(nsim):
        nori = noris[n]
        df = df_dict[nori]['df']
        label = df_dict[nori]['label']
        color = df_dict[nori]['color']
        if 'binw' in df_dict[nori]:
            binw = df_dict[nori]['binw']
        else:
            binw = None

        columns = ['born', 'numori_born', ]
        Ld = df['Ld_fit'].dropna().to_numpy()
        edges = make_binning_edges(Ld, binw=binw)
        hist, edges = np.histogram(Ld, bins=edges, density=True)
        X = 0.5*(edges[:-1]+edges[1:])

        # plot the histogram
        ax.plot(X, hist, '-', lw=lw, color=color, label=label)

    ax.set_xlabel('$S_d$', fontsize='medium')
    ax.set_ylabel('pdf', fontsize='medium')

    # EXIT
    fig.suptitle(fig_title, fontsize='x-large', x=0.5, ha='center')
    fig.tight_layout(rect=[0.,0.,1.,0.95])
    return fig

### Figure 5 re-analysis ###
def process_df_EXP(df_):
    """
    Process experimental data from Witz and al.
    """
    df = copy.deepcopy(df_)

    # remove some columns
    del df['Ld']
    del df['Lb']
    del df['Li']
    del df['length']
    del df['full_cellcycle']
    del df['pearson_log']
    del df['Li_old']
    del df['period']

    # rename columns
    renamedict = {'Lb_fit': 'Lb', \
                  'Ld_fit': 'Ld', \
                  'Li_fit': 'LAi', \
                  'tau_fit': 'lambda_inv', \
                  'born': 'Tb', \
                  }
    df = df.rename(columns=renamedict)

    return df

def process_df_EXP2(df_):
    """
    Process experimental data from Si & Le Treut et al.
    """
    df = copy.deepcopy(df_)

    # remove some columns
    del df['lambda']
    del df['Delta_bd']
    del df['width']
    del df['tau']
    del df['daughter ID']
    del df['initiator ID']
    del df['initiator B']
    del df['ncycle']
    del df['nori init']
    del df['Si_fit']
    del df['Sb_fit']
    del df['Sd_fit']
    del df['delta_id_m1']
    del df['delta_id_m2']
    del df['delta_id_m3']

    # rename columns
    renamedict = {'Sb': 'Lb', \
                  'Sd': 'Ld', \
                  'Lambda_i': 'LAi', \
                  'delta_ii': 'dLi', \
                  'delta_id': 'dLdLi', \
                  'phi': 'rfact', \
                  'tau_eff': 'lambda_inv', \
                  'cell ID': 'cell_id', \
                  'mother ID': 'mother_id'

                  }
    df = df.rename(columns=renamedict)

    # set index
    df = df.set_index('cell_id')

    return df

def process_df_SIM(df_):
    """
    Process simulation data obtained with Witz and al. original code.
    """
    df = copy.deepcopy(df_)

    # rename columns
    renamedict = {'Lb_fit': 'Lb', \
                  'Ld_fit': 'Ld', \
                  'Li_fit': 'LAi', \
                  'DLi': 'dLi', \
                  'final_DLdLi': 'dLdLi', \
                  'tau_fit': 'lambda_inv', \
                  'born': 'Tb', \
                  }
    df = df.rename(columns=renamedict)

    # remove some columns
    del df['length']
    del df['mLi_fit']
    del df['mLd_fit']

    # some additional computations
    df['dLdLi'] = df['dLdLi'] / 2.

    return df

def plot_simulation_overlays(df_dict, binw_dict=None, \
                    fig=None, lw=0.5, ms=2, \
                    fig_title=None, bar_width=0.7):
    """
    Method used to make a comparative plot of the distributions, means, CVs and autocorrelation of several variables.
    """

    # MAKE FIGURE #
    nrows = 4
    ncols = 5
    if fig is None:
        figsize=(4*ncols,3*nrows)
        fig = plt.figure(facecolor='w',figsize=figsize)
    gs = mgs.GridSpec(nrows, ncols, hspace=0.1)

    axes = []

    # axes for distributions
    row = []
    ax0 = fig.add_subplot(gs[0,0])
    row.append(ax0)
    for i in range(1,ncols):
        #row.append(fig.add_subplot(gs[0,i], sharey=ax0))
        row.append(fig.add_subplot(gs[0,i]))
    axes.append(row)
    # means
    ax0 = fig.add_subplot(gs[1,:])
    axes.append([ax0])
    # CVs
    ax = fig.add_subplot(gs[2,:], sharex=ax0)
    axes.append([ax])
    # AC
    ax = fig.add_subplot(gs[3,:], sharex=ax0)
    axes.append([ax])

    # decorations
    ## all
    for ax_list in axes:
        for ax in ax_list:
            ax.tick_params(length=4, bottom=True, left=True, labelbottom=True, labelleft=True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    ## specific
    for ax in axes[1]+axes[2]:
        ax.tick_params(labelbottom=False, bottom=False)
    axes[3][0].tick_params(labelbottom=True, bottom=False)

    # FILL-IN FIGURE #
    file_list = list(df_dict.keys())
    nfiles = len(file_list)
    norm = mco.Normalize(vmin=0, vmax=nfiles-1)
    cmap = cm.rainbow
    for n in range(nfiles):
        fpath = file_list[n]
        print(fpath)
        if not ('color' in df_dict[fpath]):
            df_dict[fpath]['color'] = cmap(norm(n))
        df_dict[fpath]['nori_init'] = 1
        plot_simulation_single(fig, axes, df_dict[fpath], n, nfiles, lw=lw, ms=ms, bar_width=bar_width, binw_dict=binw_dict)

    # legend
    patches=[]
    for n in range(nfiles):
        fpath = file_list[n]
        if 'label' in df_dict[fpath]:
            label = df_dict[fpath]['label']
        else:
            label=os.path.basename(fpath)
            label = os.path.splitext(label)[0]
        color=df_dict[fpath]['color']
        patch = mpatches.Patch(color=color, label=label)
        patches.append(patch)
    #ncol = int(np.ceil(np.sqrt(nfiles)))
    ncol = int(nfiles)
    plt.figlegend(handles=patches, fontsize='large', ncol=ncol ,borderaxespad=0, borderpad=0, loc='lower center', frameon=False)

    # title
    if not fig_title is None:
        fig.suptitle(fig_title,fontsize='x-large', x=0.5, ha='center')
    gs.tight_layout(fig, rect=[0.,0.05,1.,0.95])
    return fig

def plot_simulation_single(fig, axes, data_dict, n_ind, n_tot, lw=0.5, ms=2, bar_width=0.7, binw_dict=None):

    # parameters and input
    df = data_dict['df']
    color = data_dict['color']
    nori_initial = data_dict['nori_init']

    # conditionning
    data_format = data_dict['format']
    if data_format == 'SIM':
        df = process_df_SIM(df)
    elif data_format == 'EXP2':
        df = process_df_EXP2(df)
    else:
        raise ValueError("Data format must be \'SIM\' or \'EXP2\'!")

    # extraction
    columns = ['Lb', 'Ld', 'LAi', 'dLi', 'dLdLi', 'mother_id', 'rfact']
    df = copy.deepcopy(df.loc[:, columns])
    if len(df.dropna()) == 0:
        raise ValueError("Problem with one of the columns")

    for key in ['mLd', 'mLAi']:
        columns.append(key)
        df[key] = np.nan
    for cid in df.index:
        mid = df.at[cid,'mother_id']
        if mid in df.index:
            df.at[cid, 'mLd'] = df.at[mid, 'Ld']
            df.at[cid, 'mLAi'] = df.at[mid, 'LAi']

    # prepare data
    data = df.loc[:, columns].dropna().to_numpy().astype('float64')
    Lb, Ld, LAi, delta_ii, delta_id, mid, rfact, mLd, mLAi = data.T
#    LAi = Li / nori_id
#    mLAi = mLi / mnori_id
#    delta_id = Delta_id / (2*nori_id)
#    delta_ii = Delta_ii / (2*nori_ii)

#    mu_ii = np.nanmean(delta_ii)
#    s_ii =  np.nanstd(delta_ii)
#    mu_id = np.nanmean(delta_id)
#    s_id =  np.nanstd(delta_id)
#    mu_i_pred = 2*mu_ii
#    std_i_pred = 2./np.sqrt(3) * s_ii
#    acf_i_pred = 0.5
#    acf_d_pred = 0.5 / (1. + 3.*s_id**2/s_ii**2)
#    mu_d_pred = nori_initial*2*(mu_ii+mu_id)
#    std_d_pred = nori_initial*2.*np.sqrt(s_ii**2 / 3. + s_id**2)

    # one last check
    ninit_cells = np.sum(rfact == 0.5)
    ntot_cells = len(rfact)
    print("Cells with rfact = 0.5: {:d} / {:d} <=> {:.2f} %".format(ninit_cells, ntot_cells, float(ninit_cells)/float(ntot_cells)*100))
    ##############
    # histograms #
    ##############
    xlabels = []
    keys = ['delta_ii', 'Lambda_i', 'delta_id', 'Ld', 'rfact']
    if binw_dict is None:
        binw_dict = {key: None for key in keys}
    else:
        for key in keys:
            if not key in binw_dict.keys():
                binw_dict[key] = None
    mus = []
    stds = []
    cvs = []

    # delta_ii
    ## computations
    xlabel = "$\delta_{ii}$"
    xlabels.append(xlabel)
    binw = binw_dict['delta_ii']
    X = delta_ii
    edges = make_binning_edges(X, x0=None, x1=None, binw=binw)
    x0 = edges[0]
    x1 = edges[-1]
    hist, edges = np.histogram(X, bins=edges, density=True)
    mu = np.nanmean(X)
    std = np.nanstd(X)
    cv = std/mu
    mus.append(mu)
    stds.append(std)
    cvs.append(cv)

    ## plot
    ax = axes[0][0]
    ax.plot(0.5*(edges[:-1]+ edges[1:]), hist, '-', lw=lw, color=color)

    #ax.annotate(label, xy=(1.,0.98), xycoords='axes fraction', va='top', ha='right', fontsize='medium')
    ax.set_xlabel(xlabel, fontsize='large')
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelleft=False, left=False)
    #ax.set_ylim(0.,None)

    # Lambda_i
    ## computations
    xlabel = "$\Lambda_{i}$"
    xlabels.append(xlabel)
    binw = binw_dict['Lambda_i']
    X = LAi
    edges = make_binning_edges(X, x0=None, x1=None, binw=binw)
    x0 = edges[0]
    x1 = edges[-1]
    hist, edges = np.histogram(X, bins=edges, density=True)
    mu = np.nanmean(X)
    std = np.nanstd(X)
    cv = std/mu
    mus.append(mu)
    stds.append(std)
    cvs.append(cv)

    ## plot
    ax = axes[0][1]
    ax.plot(0.5*(edges[:-1]+ edges[1:]), hist, '-', lw=lw, color=color)

    ax.set_xlabel(xlabel, fontsize='large')
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelleft=False, left=False)
    #ax.set_ylim(0.,None)

    # delta_id
    ## computations
    xlabel = "$\delta_{id}$"
    xlabels.append(xlabel)
    binw = binw_dict['delta_id']
    X = delta_id
    edges = make_binning_edges(X, x0=None, x1=None, binw=binw)
    x0 = edges[0]
    x1 = edges[-1]
    hist, edges = np.histogram(X, bins=edges, density=True)
    mu = np.nanmean(X)
    std = np.nanstd(X)
    cv = std/mu
    mus.append(mu)
    stds.append(std)
    cvs.append(cv)

    ## plot
    ax = axes[0][2]
    ax.plot(0.5*(edges[:-1]+ edges[1:]), hist, '-', lw=lw, color=color)

    ax.set_xlabel(xlabel, fontsize='large')
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelleft=False, left=False)
    #ax.set_ylim(0.,ymax)

    # Ld
    ## computations
    xlabel = "$S_{d}$"
    xlabels.append(xlabel)
    binw = binw_dict['Ld']
    X = Ld
    edges = make_binning_edges(X, x0=None, x1=None, binw=binw)
    x0 = edges[0]
    x1 = edges[-1]
    hist, edges = np.histogram(X, bins=edges, density=True)
    mu = np.nanmean(X)
    std = np.nanstd(X)
    cv = std/mu
    mus.append(mu)
    stds.append(std)
    cvs.append(cv)

    ## plot
    ax = axes[0][3]
    ax.plot(0.5*(edges[:-1]+ edges[1:]), hist, '-', lw=lw, color=color)

    ax.set_xlabel(xlabel, fontsize='large')
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelleft=False, left=False)
    #ax.set_ylim(0.,None)

    # septum positioning
    ## computations
    xlabel = "$\phi_{1/2}$"
    #xlabel = "rfact"
    xlabels.append(xlabel)
    binw = binw_dict['rfact']
    X = rfact
    #X = (1.-rfact)/rfact
    edges = make_binning_edges(X, x0=None, x1=None, binw=binw)
    x0 = edges[0]
    x1 = edges[-1]
    hist, edges = np.histogram(X, bins=edges, density=True)
    mu = np.nanmean(X)
    std = np.nanstd(X)
    cv = std/mu
    mus.append(mu)
    stds.append(std)
    cvs.append(cv)

    ## plot
    ax = axes[0][4]
    ax.plot(0.5*(edges[:-1]+ edges[1:]), hist, '-', lw=lw, color=color)

    ax.set_xlabel(xlabel, fontsize='large')
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelleft=False, left=False)
    #ax.set_ylim(0.,None)

    ##############
    # bar plots  #
    ##############
    bar_plots = []
    # means
    ax = axes[1][0]
    X = mus
    ylabel = 'means'
    xticks = np.arange(len(X))
    rects = ax.bar(xticks-0.5*bar_width+n_ind*bar_width/n_tot, X, bar_width/n_tot, color=color)
    autolabel_vertical(ax, rects, fontsize='medium')
    ax.set_ylabel(ylabel, fontsize='large')
    ax.set_xticks(xticks)
    ax.tick_params(axis='both', length=4)
    ax.tick_params(top=False, right=False)

    # CVs
    ax = axes[2][0]
    X = [el*100 for el in cvs]
    ylabel = 'CVs (%)'
    xticks = np.arange(len(X))
    rects = ax.bar(xticks-0.5*bar_width+n_ind*bar_width/n_tot, X, bar_width/n_tot, color=color)
    autolabel_vertical(ax, rects, fmt_str='{:.0f}', fontsize='medium')
    ax.set_ylabel(ylabel, fontsize='large')
    ax.set_xticks(xticks)
    ax.tick_params(axis='both', length=4)
    ax.tick_params(top=False, right=False)

    # AC
    ax = axes[3][0]
    acd_i = sst.pearsonr(mLAi, LAi)[0]
    acd_d = sst.pearsonr(mLd, Ld)[0]
    X = np.zeros(len(mus))
    X[1] = acd_i
    X[3] = acd_d
    ylabel = '$\\rho(n-1, n)$'
    xticks = np.arange(len(X))
    xpos = xticks-0.5*bar_width+n_ind*bar_width/n_tot
    idx = (X > 0.)
    X = X[idx]
    xpos = xpos[idx]
    rects = ax.bar(xpos, X, bar_width/n_tot, color=color)
    bar_plots.append(rects)
    autolabel_vertical(ax, rects, fontsize='medium')
    ax.set_ylabel(ylabel, fontsize='large')
    ax.set_xticks(xticks)
    ax.tick_params(axis='both', length=4)
    ax.tick_params(top=False, right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize='large')

def plot_adder_compare(df_dict, fig=None, lw=0.5, ms=2, ylim=None, fig_title=None, npred = -1):
    """
    Method used to compare the adder behavior of several datasets.

    """
    # MAKE THE FIGURE
    if fig is None:
        figsize=(4*3,4)
        fig = plt.figure(num='none', facecolor='w',figsize=figsize)

    axes = []
    gs = mgs.GridSpec(1, 2, hspace=0.1)
    for i in range(2):
        ax = fig.add_subplot(gs[0,i])
        axes.append(ax)
    for ax in axes:
        ax.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    file_list = list(df_dict.keys())
    nfiles = len(file_list)
    norm = mco.Normalize(vmin=0, vmax=nfiles-1)
    cmap = cm.rainbow
    for n in range(nfiles):
        fpath = file_list[n]
        if not ('color' in df_dict[fpath]):
            df_dict[fpath]['color'] = cmap(norm(n))
        df_dict[fpath]['nori_init'] = 1
        if n == npred:
            plot_pred = True
            print("Plotting predictions with statistics computed from {:s}".format(fpath))
        else:
            plot_pred = False
        plot_adder(axes, df_dict[fpath], lw=lw, ms=ms, plot_pred=plot_pred)


    ax = axes[0]
    ax.set_aspect('equal')
    xlabel='$S_d^{(n-1)}$'
    ylabel='$S_d^{(n)}$'
    ax.set_title("Division size correlation: $(S_d^{(n-1)}, S_d^{(n)})$", fontsize='medium')
    ax.set_xlabel(xlabel, fontsize='large')
    ax.set_ylabel(ylabel, fontsize='large')

    ax = axes[1]
    ax.set_aspect('equal')
    xlabel='$S_b$'
    ylabel='$S_d-S_b$'
    ax.set_xlabel(xlabel, fontsize='large')
    ax.set_ylabel(ylabel, fontsize='large')
    ax.set_title("Adder correlation: $(S_b, S_d-S_b)$", fontsize='medium')
    #ax.set_xlim(0,None)
    #ax.set_ylim(0,None)
    ax.legend(loc='upper left', fontsize='medium', bbox_to_anchor=(1.,0.98))

    # EXIT
    fig.suptitle(fig_title, fontsize='large', x=0.5, ha='center')
    fig.tight_layout(rect=[0.,0.,1.,0.95])
    return fig

def plot_adder(axes, data_dict, lw=-.5, ms=2, plot_pred=False):
    # parameters and input
    df = data_dict['df']
    color = data_dict['color']
    nori_initial = data_dict['nori_init']
    label = data_dict['label']
    if 'binw_Lb' in data_dict:
        binw_b = data_dict['binw_Lb']
    else:
        binw_b = None
    if 'binw_Ld' in data_dict:
        binw_d = data_dict['binw_Ld']
    else:
        binw_d = None

    # conditionning
    data_format = data_dict['format']
    if data_format == 'SIM':
        df = process_df_SIM(df)
    elif data_format == 'EXP':
        df = process_df_EXP(df)
    elif data_format == 'EXP2':
        df = process_df_EXP2(df)
    else:
        raise ValueError("Data format must be either \'SIM\' or \'EXP\' or \'EXP2\'!")

    # extraction
    ## if prediction computation is required
    if plot_pred:
        if data_format == 'EXP':
            raise ValueError("Not possible to compute \delta_id with experiment in general")

        df_bkp = copy.deepcopy(df)
        columns = ['Lb', 'Ld', 'LAi', 'dLi', 'dLdLi', 'mother_id', 'rfact']
        df = copy.deepcopy(df.loc[:, columns])
        if len(df.dropna()) == 0:
            raise ValueError("Problem with one of the columns")

        for key in ['mLd', 'mLAi']:
            columns.append(key)
            df[key] = np.nan
        for cid in df.index:
            mid = df.at[cid,'mother_id']
            if mid in df.index:
                df.at[cid, 'mLd'] = df.at[mid, 'Ld']
                df.at[cid, 'mLAi'] = df.at[mid, 'LAi']
        data = df.loc[:, columns].dropna().to_numpy().astype('float64')
        Lb, Ld, LAi, delta_ii, delta_id, mid, rfact, mLd, mLAi = data.T

        mu_ii = np.nanmean(delta_ii)
        s_ii =  np.nanstd(delta_ii)
        mu_id = np.nanmean(delta_id)
        s_id =  np.nanstd(delta_id)
        mu_i_pred = 2*mu_ii
        std_i_pred = 2./np.sqrt(3) * s_ii
        acf_i_pred = 0.5
        acf_d_pred = 0.5 / (1. + 3.*s_id**2/s_ii**2)
        mu_d_pred = nori_initial*2*(mu_ii+mu_id)
        std_d_pred = nori_initial*2.*np.sqrt(s_ii**2 / 3. + s_id**2)
        df = df_bkp

    ## regular computations
    columns = ['Lb', 'Ld', 'mother_id']
    df = copy.deepcopy(df.loc[:, columns])
    if len(df.dropna()) == 0:
        raise ValueError("Problem with one of the columns")

    for key in ['mLd']:
        columns.append(key)
        df[key] = np.nan
    for cid in df.index:
        mid = df.at[cid,'mother_id']
        if mid in df.index:
            df.at[cid, 'mLd'] = df.at[mid, 'Ld']

    # prepare data
    data = df.loc[:, columns].dropna().to_numpy().astype('float64')
    Lb, Ld, mid, mLd = data.astype('float64').T
    DL = Ld - Lb
    r_d = sst.pearsonr(mLd, Ld)[0]    # true mother/daughter correlation
    a_d = sst.pearsonr(2*Lb, Ld)[0]   # this is appropriate for the adder plot
    label += ", $r = {:.2f}$, $\\alpha = {:.2f}$".format(r_d, a_d)

    # division size correlation plot
    ax = axes[0]
    ## bin data
    edges = make_binning_edges(mLd, x0=None, x1=None, binw=binw_d)
    X_binned = 0.5*(edges[:-1] + edges[1:])
    nbins = len(edges)-1
    Y_binned_sets = get_binned(mLd,Ld,edges)
    Y_binned = np.zeros(nbins)*np.nan
    Y_counts = np.zeros(nbins)*np.nan
    Y_vars = np.zeros(nbins)*np.nan

    for i in range(nbins):
        Yi = Y_binned_sets[i]
        Zi = np.nansum(np.isfinite(Yi),axis=0)
        if (Zi == 0):
            continue
        m = np.nansum(Yi) / Zi
        v = np.nansum((Yi-m)**2) / Zi
        Y_counts[i] = Zi
        Y_binned[i] = m
        Y_vars[i] = v
    idx = np.isfinite(Y_counts) & (Y_counts > 2)
    mLd_binned = X_binned[idx]
    Ld_binned = Y_binned[idx]
    Ld_binned_err = np.sqrt(Y_vars[idx]/Y_counts[idx])
    mLd_fit = np.linspace(mLd_binned[0], mLd_binned[-1], 1000)
    mu_d = np.nanmean(Ld)

    # plot data
    #ax.plot(Lb, DL, 'o', ms=ms, color=color, lw=lw, alpha=0.2)
    ax.errorbar(mLd_binned, Ld_binned, yerr=Ld_binned_err, color=color, marker='s', ms=2*ms, ecolor=color, elinewidth=1*lw, lw=0*lw, label=label)
    #ax.plot(mLd_fit, r_d*mLd_fit + (1-r_d)*mu_d , '-', lw=2*lw, color=color)

    if plot_pred:
        ax.plot(mLd_fit, acf_d_pred*mLd_fit + (1-acf_d_pred)*mu_d_pred , '--', lw=2*lw, color='k', label='prediction, $r = \\alpha = {:.2f}$'.format(acf_d_pred), zorder=3)

    # adder plot
    ax = axes[1]
    ## bin data
    edges = make_binning_edges(Lb, x0=None, x1=None, binw=binw_b)
    X_binned = 0.5*(edges[:-1] + edges[1:])
    nbins = len(edges)-1
    Y_binned_sets = get_binned(Lb,DL,edges)
    Y_binned = np.zeros(nbins)*np.nan
    Y_counts = np.zeros(nbins)*np.nan
    Y_vars = np.zeros(nbins)*np.nan

    for i in range(nbins):
        Yi = Y_binned_sets[i]
        Zi = np.nansum(np.isfinite(Yi),axis=0)
        if (Zi == 0):
            continue
        m = np.nansum(Yi) / Zi
        v = np.nansum((Yi-m)**2) / Zi
        Y_counts[i] = Zi
        Y_binned[i] = m
        Y_vars[i] = v
    idx = np.isfinite(Y_counts) & (Y_counts > 2)
    Lb_binned = X_binned[idx]
    DL_binned = Y_binned[idx]
    DL_binned_err = np.sqrt(Y_vars[idx]/Y_counts[idx])
    Lb_fit = np.linspace(Lb_binned[0], Lb_binned[-1], 1000)
    mu_b = np.nanmean(Lb)

    # plot data
    #ax.plot(Lb, DL, 'o', ms=ms, color=color, lw=lw, alpha=0.2)
    ax.errorbar(Lb_binned, DL_binned, yerr=DL_binned_err, color=color, marker='s', ms=2*ms, ecolor=color, elinewidth=1*lw, lw=0*lw, label=label)
    #ax.plot(Lb_fit, (2*a_d-1.)*Lb_fit + 2*(1-a_d)*mu_b , '-', lw=2*lw, color=color)

    if plot_pred:
        ax.plot(Lb_fit, (2*acf_d_pred-1.)*Lb_fit + 2*(1-acf_d_pred)*mu_d_pred/2 , '--', lw=2*lw, color='k', label='prediction, $r = \\alpha = {:.2f}$'.format(acf_d_pred), zorder=3)

    return

### processing of Si & Le Treut et al experimental data ###

def process_fsglt(df):
    """
    Method to process experimental data from Si & Le Treut and add several attributes.
    """

    # add mother ids
    add_mother_id(df)

    # add initiation information
    add_initiation(df)
    add_si_fit(df)

    # add initiation-to-initiation adder
    add_delta_ii(df)

    # add initiation-to-division adder
    add_delta_id_method1(df)
    add_delta_id_method2(df)
    add_delta_id_method3(df)

    return

def backtrack_time(mydf, cell_id, delta_t, ngen=0):
    """
    Helper function to `backtrack_initiation`
    """
    # make sure the cell is in the data frame
    idx = mydf['cell ID'] == cell_id
    if np.sum(idx.to_list()) != 1:
#         print('ID', cell_id)
#         raise ValueError("Found a cell ID not in the data")
        return None, delta_t, ngen

    tau = mydf.loc[idx, 'tau'].iloc[0]
    delta_t_new = delta_t - tau
    if delta_t_new <= 0:
        return cell_id, -delta_t_new, ngen
    else:
        ngen += 1
        cell_id_mother = mydf.loc[idx, 'mother ID'].iloc[0]
        return backtrack_time(mydf, cell_id_mother, delta_t_new, ngen)

def backtrack_initiation(mydf, cell_id):
    """
    Recursive function to backtrack the cell in which an initiation event associated
    with a division event in the current cell originated.
    """
    # make sure the cell is in the data frame
    idx = mydf['cell ID'] == cell_id
    if np.sum(idx.to_list()) != 1:
        print('ID', cell_id)
        raise ValueError("Cell ID not in the data")

    # get the taucyc
    taucyc = mydf.loc[idx, 'tau_cyc'].iloc[0]

    return backtrack_time(mydf, cell_id, taucyc)

def add_mother_id(df):
    df['mother ID'] = None

    ndata = len(df)
    for i in range(ndata):
        subdf = df.iloc[i]
        cell_id = subdf['cell ID']
        daughter_id = subdf['daughter ID']

        idx = df['cell ID'] == daughter_id
        df.loc[idx,'mother ID'] = cell_id
    return

def add_initiation(df):
    ndata = len(df)
    df['initiator ID'] = None
    df['initiator B'] = None
    df['ncycle'] = None
    df['nori init'] = None      # number of oriC just before initiation

    for i in df.index:
#     print(row.loc['cell ID'])
        cell_id = df.at[i,'cell ID']
        init_id, B, ncycle = backtrack_initiation(df, cell_id)
        if not (init_id is None):
            df.at[i,'initiator ID'] = init_id
            df.at[i, 'ncycle'] = ncycle
            df.at[i, 'initiator B'] = B
            df.at[i, 'nori init'] = 2**ncycle
            idx = df['cell ID'] == init_id
            if np.sum(idx.to_list()) != 1:
                raise ValueError("Cell ori {:d} not in the data".format(init_id))
    return

def add_si_fit(df):
    """
    Add the fitted size at initiation using information on initiation
    """
    df['Si_fit'] = None
    df['Sb_fit'] = None
    df['Sd_fit'] = None

    for i in df.index:
        init_id = df.at[i,'initiator ID']
        age_init = df.at[i,'initiator B']
        if init_id is None:
            continue

        idx = df['cell ID'] == init_id
        if np.sum(idx.to_list()) != 1:
            raise ValueError("There should be exactly one initiator cell")

        # start the fit
        lam, sb, sd, tau = df.loc[idx, ['lambda','Sb','Sd','tau']].iloc[0].to_numpy()
        #print(i, df.at[i,'cell ID'], init_id, lam, sb, sd, age_init, tau)
        ## The fitted function is f(t) = sb_fit exp(lambda . t)
        ## sb_fit is determined so as to minimize (f(0) - sb)^2 + (f(tau) - sd)^2
        ## The solution is given below
        a = np.exp(lam*tau)
        sb_fit = (sb + sd*a)/(1.+a**2)
        sd_fit = sb_fit * np.exp(lam*tau)
        si_fit = sb_fit * np.exp(lam*age_init)

        df.loc[idx,['Sb_fit','Sd_fit']] = [sb_fit, sd_fit]
        df.at[i, 'Si_fit'] = si_fit
    return

def add_delta_ii(df):
    """
    Add delta_ii: added size from initiation to initiation per origin
    This is the definition consistent with:
      * the `cross_generation_construct` method in `decomposition.py` file.
      * the implementation `simul_doubleadder` in `coli_simulation.py` file.

    """
    df['delta_ii'] = None

    for i in df.index:
        cell_id = df.at[i, 'cell ID']
        Lambda_i = df.at[i, 'Lambda_i']
        mother_id = df.at[i, 'mother ID']

        if mother_id is None:
            continue

        idx = df['cell ID'] == mother_id
        if np.sum(idx.to_list()) != 1:
            raise ValueError("There should be exactly one mother cell")

        mLambda_i = df.loc[idx, 'Lambda_i'].iloc[0]

        df.at[i, 'delta_ii'] = Lambda_i - 0.5*mLambda_i
    # end loop on cells

    return

def add_delta_ii_forward(df):
    """
    Add delta_ii: added size from initiation to initiation per origin
    This definition is consistent with Si & Le Treut 2019.
    """
    df['delta_ii_forward'] = None

    for i in df.index:
        cell_id = df.at[i, 'cell ID']
        daughter_id = df.at[i, 'daughter ID']
        Lambda_i = df.at[i, 'Lambda_i']

        idx = df['cell ID'] == daughter_id
        if np.sum(idx.to_list()) != 1:
            continue

        Lambda_i_f = df.loc[idx, 'Lambda_i'].iloc[0]

        df.at[i, 'delta_ii_forward'] = Lambda_i_f - 0.5*Lambda_i
    # end loop on cells

    return

def add_delta_id_method1(df):
    """
    Add delta_id: added size from initiation to division per origin.
    """
    delta_key = 'delta_id_m1'
    df[delta_key] = None

    for i in df.index:
        Lambda_i = df.at[i, 'Lambda_i']
        sd = df.at[i, 'Sd']
        df.at[i,delta_key] = 0.5*(sd - Lambda_i)
    # end loop on cells
    return

def add_delta_id_method2(df):
    """
    Add delta_id: added size from initiation to division per origin.
    Here we find \delta_id by applying the formula:
      2 \delta_{id}^{(n)} = \sum_{k=1}^{p} \frac{ S_d^{(n-k)} - S_b^{(n-k)}} {2^k}
    where:
      * initiation occurs in generation n-p and division in generation n.
      * \phi^{(n)} is the division ratio associated to division (n, n+1)
    """
    delta_key = 'delta_id_m2'
    df[delta_key] = None

    for i in df.index:
        cell_id = df.at[i, 'cell ID']
        init_id = df.at[i, 'initiator ID']
        nori_init = df.at[i, 'nori init']
        Lambda_i = df.at[i, 'Lambda_i']

        if init_id is None:
            continue

        # initialize
        idx = df['cell ID'] == init_id
        if np.sum(idx.to_list()) != 1:
            raise ValueError("There should be exactly one initiator cell")

        sd = df.loc[idx, 'Sd'].iloc[0]
        cml_size = sd - nori_init*Lambda_i
        if (cml_size < 0.):
            print("Calculation of the number of oric is wrong for cell ID {:d}... ignoring it".format(cell_id))
            continue
            #raise ValueError("Calculation of the number of oric is wrong.")

        # start loop
        count=0
        count_MAX=5
        current_id = init_id
        while (current_id != cell_id):
            current_id = df.loc[idx, 'daughter ID'].iloc[0]

            idx = df['cell ID'] == current_id
            if np.sum(idx.to_list()) != 1:
                raise ValueError("There should be exactly one daughter cell")

            sb, sd = df.loc[idx, ['Sb', 'Sd']].iloc[0].to_numpy()

            cml_size /= 2.  # because one origin was lost at division
            cml_size += sd - sb

            count +=1
            if count > count_MAX:
                raise ValueError("Trapped in infinite loop")
        # end loop
        cml_size /= 2.
        df.at[i, delta_key] = cml_size
    # end loop on cells
    return

def add_delta_id_method3(df):
    """
    Add delta_id: added size from initiation to division per origin.
    Here we adopt the conventions of the simulations:
      * The total added size is determined at replication initiation once and for all.
      * In principle, division among cousin/daughter cells will be synchronized.
    Then we have:
      \frac{ S_d^{(n)} }{ \phi^{(n-1)} \phi^{(n-2)} ... \phi^{(n-p)}} = 2^p (\Lambda_i^{(n)}  + 2 \delta_{id}^{(n)},
    where:
      * initiation occurs in generation n-p and division in generation n.
      * \phi^{(n)} is the division ratio associated to division (n, n+1)
    """

    delta_key = 'delta_id_m3'
    df[delta_key] = None

    for i in df.index:
        cell_id = df.at[i, 'cell ID']
        init_id = df.at[i, 'initiator ID']
        nori_init = df.at[i, 'nori init']
        Lambda_i = df.at[i, 'Lambda_i']
        sd = df.at[i,'Sd']

        if init_id is None:
            continue

        # initialize
        idx = df['cell ID'] == init_id
        if np.sum(idx.to_list()) != 1:
            raise ValueError("There should be exactly one initiator cell")

        rfact = 1.
        # start loop
        count=0
        count_MAX=5
        current_id = init_id
        while (current_id != cell_id):
            current_id = df.loc[idx, 'daughter ID'].iloc[0]

            idx = df['cell ID'] == current_id
            if np.sum(idx.to_list()) != 1:
                raise ValueError("There should be exactly one daughter cell")

            phi = df.loc[idx, 'phi'].iloc[0]
            rfact *= (2*phi)

            count +=1
            if count > count_MAX:
                raise ValueError("Trapped in infinite loop")
        # end loop

        df.at[i, delta_key] = 0.5*(sd/rfact - Lambda_i)

    # end loop on cells
    return

def get_seq(df, cid, field, tid=None):
    """
    Return the divisions ratio from initiator generation to current generation.
    """
    # initialize
    idx = df['cell ID'] == cid
    if np.sum(idx.to_list()) != 1:
        raise ValueError("There should be exactly one matching cell")
    mid = df.loc[idx, 'mother ID'].iloc[0]
    val = df.loc[idx, field].iloc[0]

    if (tid is None):
        tid = df.loc[idx, 'initiator ID'].iloc[0]

    res = [val]
    while (cid != tid):
        cid = mid

        idx = df['cell ID'] == cid
        if np.sum(idx.to_list()) != 1:
            raise ValueError("There should be exactly one matching cell")

        mid = df.loc[idx, 'mother ID'].iloc[0]
        val = df.loc[idx, field].iloc[0]
        res.append(val)

    # sort from older to current generation
    res.reverse()
    return res

def add_allvariables(df):
    """
    This method add all physiological variables that will be used in the determinant scoring analysis.
    """

    columns = [
            'Si', \
            'Delta_bi', \
            'R_bd', \
            'R_bi', \
            'R_ii_b', \
            'R_ii_f', \
            'R_id', \
            'Lambda_i_b', \
            'Lambda_i_f', \
            'tau_ii_b', \
            'tau_ii_f', \
              ]
    for col in columns:
        df[col]=None

    for i in df.index:
        cid = df.at[i, 'cell ID']
        mid = df.at[i, 'mother ID']
        did = df.at[i, 'daughter ID']
        iid = df.at[i, 'initiator ID']
        nori_init = df.at[i, 'nori init']
        Lambda_i = df.at[i, 'Lambda_i']
        sd = df.at[i,'Sd']
        sb = df.at[i,'Sb']

        hasinit = not (iid is None)
        idx = df['initiator ID'] == cid
        if np.sum(idx.to_list()) == 1:  # if there is exactly one initiation
            # initiation size if there is initiation in current generation
            Si = df.loc[idx, 'nori init'].iloc[0] * df.loc[idx, 'Lambda_i'].iloc[0]
            df.at[i, 'Si'] = Si

            # Added size from birth to initiation
            Delta_bi = Si - sb
            df.at[i,'Delta_bi'] = Delta_bi

            # initiation-to-birth ratio
            if hasinit:
                R_bi = Si/sb
                df.at[i, 'R_bi'] = R_bi

        # division-to-birth ratio
        R_bd = sd/sb
        df.at[i, 'R_bd'] = R_bd

        # initiation size per ori in previous and next replication cycle
        if not (mid is None):
            idx = df['cell ID'] == mid
            if np.sum(idx.to_list()) == 1:
                df.at[i, 'Lambda_i_b'] = df.loc[idx, 'Lambda_i'].iloc[0]
        if not (did is None):
            idx = df['cell ID'] == did
            if np.sum(idx.to_list()) == 1:
                df.at[i, 'Lambda_i_f'] = df.loc[idx, 'Lambda_i'].iloc[0]

        # initiation-to-initiation duration -- backward
        mhasinit = False
        if not (mid is None):
            idx = df['cell ID'] == mid
            if np.sum(idx.to_list()) != 1:
                raise ValueError("There should be exactly one mother cell")
            miid = df.loc[idx, 'initiator ID'].iloc[0]
            mhasinit = not (miid is None)
        if hasinit and mhasinit:
            ## age at initiation
            Bs = get_seq(df, cid, 'initiator B', mid)
            mB, B = Bs

            taus = get_seq(df, iid, 'tau', miid)
            taus[0] = taus[0] - mB # subtract B period in mother initiator cell
            taus[-1] = B            # only count until initiation happens in current initiator cell

            ## total duration from previous initiation to current initiation
            tau_ii = np.sum(taus)
            df.at[i, 'tau_ii_b'] = tau_ii

        # initiation-to-initiation duration -- forward
        dhasinit = False
        if not (did is None):
            idx = df['cell ID'] == did
            if np.sum(idx.to_list()) == 1:
                diid = df.loc[idx, 'initiator ID'].iloc[0]
                dhasinit = not (diid is None)
        if hasinit and dhasinit:
            ## age at initiation
            Bs = get_seq(df, did, 'initiator B', cid)
            mB, B = Bs

            taus = get_seq(df, diid, 'tau', iid)
            taus[0] = taus[0] - mB # subtract B period in mother initiator cell
            taus[-1] = B            # only count until initiation happens in current initiator cell

            ## total duration from previous initiation to current initiation
            tau_ii = np.sum(taus)
            df.at[i, 'tau_ii_f'] = tau_ii

        # division to initiation ratio
        R_id = sd / Lambda_i
        df.at[i, 'R_id'] = R_id

        # initiation to initiation ratio -- backward
        if not (mid is None):
            idx = df['cell ID'] == mid
            if np.sum(idx.to_list()) != 1:
                raise ValueError("There should be exactly one mother cell")

            mLambda_i = df.loc[idx, 'Lambda_i'].iloc[0]
            df.at[i, 'R_ii_b'] = 2*Lambda_i / mLambda_i   # factor of 2 so that there is an exponential fit
                                            # from previous to current initiation

        # initiation to initiation ratio -- forward
        if not (did is None):
            idx = df['cell ID'] == did
            if np.sum(idx.to_list()) == 1:
                Lambda_i_f = df.loc[idx, 'Lambda_i'].iloc[0]
                df.at[i, 'R_ii_f'] = 2*Lambda_i_f / Lambda_i   # factor of 2 so that there is an exponential fit
                                                # from previous to current initiation

    # add other specific variables
    add_delta_ii_forward(df)

    return df

class ResultStruct:
    def __init__(self):
        self.x = None
        pass

def fit_normal_fsglt(xdata_, fit_range):
    """
    fit the input data to a Gaussian distribution
    Modified from Witz et al.
    """
    xdata = np.array(xdata_, dtype=np.float_)

    valbins, binmean = np.histogram(xdata, bins=fit_range)
    z = np.sum(valbins*np.diff(binmean))
    valbins = valbins/z
    bin_pos = 0.5*(binmean[:-1]+binmean[1:])
    #valbins = valbins/np.sum(valbins)*(binmean[1]-binmean[0])
    #bin_pos= np.array([0.5*(binmean[x]+binmean[x+1]) for x in range(len(binmean)-1)])

    mu = np.nanmean(xdata)
    std = np.nanstd(xdata)
    norm = 1./(np.sqrt(2.*np.pi)*std)

    res_fit = ResultStruct()
    res_fit.x = [norm, mu, std]

    return bin_pos, valbins, res_fit

def fit_lognormal_fsglt(xdata_, fit_range):
    """
    fit the input data to a Log-normal distribution
    Modified from Witz et al.
    """
    xdata = np.array(xdata_, dtype=np.float_)
    idx = xdata > 0.
    xdata = xdata[idx]


    return fit_normal_fsglt(np.log(xdata), fit_range)


def compute_determinant(mat):
    """
    See decomposition.py
    line 149

    INPUT
    -----
        mat: matrix where each row is the list of observations for one variable.
    """

    K = np.cov(mat)
    return np.linalg.det(K)/np.prod(np.diag(K))

def plot_Ivalues(table, label_mapping, nval=None, lw=0.5, ms=2, fig_title=None, figsize=None, fmt_str='{:.4f}', specials=[], color_default='black', color_special='red'):
    """
    This function plots the determinant values in the table
    """

    fig = plt.figure(num='none', facecolor='w',figsize=figsize)
    ax = fig.gca()
    ax.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)
    ax.tick_params(axis='both', which='both', length=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)


    labels = []
    values = []
    colors = []
    if nval is None:
        nval = len(table)
    for i in range(nval):
        comb = table[i][:-1]
        thevars = [label_mapping[v] for v in comb]
        labels.append(", ".join(thevars))
        values.append(table[i][-1])
        color = color_default
        for sp_comb in specials:
            if set(sp_comb) == set(comb):
                color = color_special
                print("Found special combination!")
                break
        colors.append(color)

    Y = np.arange(nval)
    rects = ax.barh(Y, values, color=colors, align='center')
    autolabel_horizontal(ax, rects, fontsize='medium', fmt_str=fmt_str)
    ax.set_yticks(Y)
    ax.set_yticklabels(labels,fontsize='medium')
    ax.invert_yaxis()
    ax.set_xlabel("I value", fontsize='medium')
    ax.set_xlim(0.,1.)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    #set_xticks(np.arange(11, dtype=np.float_)/10)

    rect=[0.,0.,1.,0.99]
    fig.tight_layout(rect=rect)
    if not fig_title is None:
        fig.suptitle(fig_title, fontsize='large', x=0.5, ha='center')
    return fig

def plot_Ivalues_all(table, lw=0.5, ms=1, fig_title=None, figsize=None, fmt_str='{:.4f}', specials=[], color_default='black', color_special='red'):
    """
    This function plots the determinant values in the table
    """

    fig = plt.figure(num='none', facecolor='w',figsize=figsize)
    ax = fig.gca()
    ax.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)
    ax.tick_params(axis='both', which='both', length=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)


    values_default = []
    values_special = []
    colors = []
    nval = len(table)
    for i in range(nval):
        comb = table[i][:-1]
        val = table[i][-1]
        isdefault=True
        for sp_comb in specials:
            if set(sp_comb) == set(comb):
                isdefault=False
                print("Found special combination!")
                break
        if isdefault:
            values_default.append(val)
            values_special.append(None)
        else:
            values_default.append(None)
            values_special.append(val)

    X = np.arange(nval)
    ax.plot(X, values_default, 'o', color=color_default)
    ax.plot(X, values_special, 'o', ms=4*ms, color=color_special)
    ax.set_ylabel("I value", fontsize='medium')
    ax.set_ylim(0.,1.)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    #set_xticks(np.arange(11, dtype=np.float_)/10)

    rect=[0.,0.,1.,0.99]
    fig.tight_layout(rect=rect)
    if not fig_title is None:
        fig.suptitle(fig_title, fontsize='large', x=0.5, ha='center')
    return fig

def load_table(fpath):
    """
    Load table of determinant analysis
    """
    table = []
    with open(fpath,'r') as fin:
        fin.readline()  # pass first line
        while True:
            line = fin.readline()
            if line == "":
                break
            if line == '\n':
                break
            tab = line.split()
            tab[-1] = float(tab[-1])
            table.append(tab)
    return table


