"""
This module allows to perform a statistical analysis called
decomposition analysis.
 """
# Author: Guillaume Witz, Biozentrum Basel, 2019
# License: MIT License

import re
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import colicycle.tools_GW as tgw
import colicycle.time_mat_operations as tmo


def renaming_modelGW(name):
    """Renaming variables. This function simply translates variable names
    used in the code into Latex style variables usable in plots. These 
    rules are valide for the double adder model
    
    Parameters
    ----------
    name : list of strings
        list of names to translate
        
    
    Returns
    -------
    name : list of strings
        translated list of names
    """
    name = [x.replace('tau_g','$\\alpha$') if type(x)==str else x for x in name]
    name = [x.replace('Lig2_fit','$\\Lambda_f$') if type(x)==str else x for x in name]
    name = [x.replace('Lb_fit','$\\Lambda_b$') if type(x)==str else x for x in name]
    name = [x.replace('Lig_fit','$\\Lambda_i$') if type(x)==str else x for x in name]
    name = [x.replace('DeltaLgi','$d\\Lambda_{if}$') if type(x)==str else x for x in name]
    name = [x.replace('DeltaLigb','$d\\Lambda_{ib}$') if type(x)==str else x for x in name]
    name = [x.replace('Tbg','$T_{ib}$') if type(x)==str else x for x in name]
    name = [x.replace('Tg','$T_{if}$') if type(x)==str else x for x in name]
    name = [x.replace('rLig','$R_{if}$') if type(x)==str else x for x in name]

    return name


def renaming_classic(name):
    """Renaming variables. This function simply translates variable names
    used in the code into Latex style variables usable in plots. These 
    rules are valide for the classic division centric model
    
    Parameters
    ----------
    name : list of strings
        list of names to translate
        
    
    Returns
    -------
    name : list of strings
        translated list of names
    """
    name = [x.replace('tau_fit','$\\lambda$') if type(x)==str else x for x in name]
    name = [x.replace('Li_fit','$L_i$') if type(x)==str else x for x in name]
    name = [x.replace('Ld_fit','$L_d$') if type(x)==str else x for x in name]
    name = [x.replace('Lb_fit','$L_b$') if type(x)==str else x for x in name]
    #name = [x.replace('Ti','$T_{bi}$') if type(x)==str else x for x in name]
    name = [re.sub(r'\bTi\b', '$T_{bi}$', x) if type(x)==str else x for x in name]
    name = [x.replace('Td','$T_{bd}$') if type(x)==str else x for x in name]
    #name = [x.replace('DeltaLib','$dL_{bi}$') if type(x)==str else x for x in name]
    name = [re.sub(r'\bDeltaLib\b', '$dL_{bi}$', x) if type(x)==str else x for x in name]
    #name = [x.replace('DeltaLid','$dL_{id}$') if type(x)==str else x for x in name]
    name = [re.sub(r'\bDeltaLid\b', '$dL_{id}$', x) if type(x)==str else x for x in name]
    name = [x.replace('DeltaTid','$T_{id}$') if type(x)==str else x for x in name]
    name = [x.replace('DeltaL','$dL$') if type(x)==str else x for x in name]
    name = [x.replace('rLdLb','$R_{db}$') if type(x)==str else x for x in name]
    
    return name


#Decomposition using the classic adder + initiation information (time, length)
def decompose_adder_init(colidata):
    """Decomposition of data using variables corresponding to the 
    division centric view of the cell cycle
    
    Parameters
    ----------
    colidata : Pandas dataframe
        dataframe with cell cycle information
        
    
    Returns
    -------
    coli_to_test : Pandas dataframe
        subset of data used for decomposition
    sorted_combinations : array of string lists
        each element of the array is a quadruplet of variable names
        corresponding to a decomposition. The list is sorted from 
        best to worst decomposition
    sorted_vals : array of floats
        independence I value for sorted decompositions
    table_to_plot : list of lists
        Each element of the list is a pair of values: the list of 
        variables and the corresponding independenc I value
        
    """
    
    #number of variables for decomposition
    numvar_to_use = 4
    
    #variable used in decomposition
    variables = ['Lb_fit','Ld_fit','Td','Ti','tau_fit','rLdLb','DeltaL','Li_fit','DeltaLib']
    
    #subselect used variables in dataframe
    coli_to_test = colidata.copy()
    coli_to_test['tau_fit'] = 1/coli_to_test['tau_fit']
    coli_to_test =     coli_to_test = coli_to_test[['Lb_fit','Ld_fit','Td','Ti','tau_fit','rLdLb','DeltaL','Li_fit','DeltaLib']].dropna()
    
    #define numeric code for each variable (1,10,100,1000 etc.)
    code = [10**i for i in range(len(variables))]
    
    #define groups of variables related by an equation (e.g. Ld = Lb*e^Td/tau_fit)
    equations = [['Ld_fit','Lb_fit','tau_fit','Td'],['DeltaL','Ld_fit','Lb_fit'],['rLdLb','Ld_fit','Lb_fit'],
    ['Li_fit','tau_fit','Lb_fit','Ti'],['DeltaLib','Li_fit','Lb_fit']]

    #Generate all possible combinations of variables by brute force
    combinations = []
    for j in range(1000):
        current = np.random.choice(variables,numvar_to_use,replace=False)
        cur_unchanged = current.copy()
        for i in range(10):
            for x in equations:
                if len(set(x) & set(current))==len(x)-1:

                    current = set.union(set(current),set(x))
        if len(current)==len(variables):
            combinations.append(cur_unchanged)

    #calculate numeric code for each combination. This is the same irrespective of the order of variables. Keep
    #only single occurrennces of each code
    finalset = set([np.sum([code[np.where(np.array(variables)==combinations[j][i])[0][0]] for i in range(numvar_to_use)]) for j in range(len(combinations))])
    
    #recover the variable names corresponding to the codes defined above
    combinations = [np.array(variables)[np.array([np.mod(np.floor_divide(list(finalset)[j],10**(i)),10) 
                                   for i in range(len(variables))])==1] for j in range(len(list(finalset)))]

    #calculate independence for each combination and sort them
    indep = [np.linalg.det(np.cov(coli_to_test[combinations[i]].T))/np.prod(np.diag(np.cov(coli_to_test[combinations[i]].T))) 
     for i in range(len(combinations))]
    sorted_combinations = np.flipud(np.array(combinations)[np.argsort(indep)])
    sorted_vals = np.flipud(np.array(indep)[np.argsort(indep)])

    '''
    #alternative way to calculate using directly correlation coeffficients
    indep2 = [np.linalg.det(np.corrcoef(coli_to_test[combinations[i]].T)) for i in range(len(combinations))]
    sorted_combinations2 = np.flipud(np.array(combinations)[np.argsort(indep2)])
    sorted_vals2 = np.flipud(np.array(indep2)[np.argsort(indep2)])'''
    

    #create output table 
    s = ", ";
    rows = [s.join(list(x)) for x in sorted_combinations]

    table_to_plot = [[rows[x], np.around(sorted_vals[x],3)] for x in range(len(sorted_combinations))]
    
    return coli_to_test, sorted_combinations, sorted_vals, table_to_plot

def decompose_double_adder(colidata):
    """Decomposition of data using variables corresponding to the 
    replication centric double adder
    
    Parameters
    ----------
    colidata : Pandas dataframe
        dataframe with cell cycle information
        
    
    Returns
    -------
    coli_to_test : Pandas dataframe
        subset of data used for decomposition
    sorted_combinations : array of string lists
        each element of the array is a quadruplet of variable names
        corresponding to a decomposition. The list is sorted from 
        best to worst decomposition
    sorted_vals : array of floats
        independence I value for sorted decompositions
    table_to_plot : list of lists
        Each element of the list is a pair of values: the list of 
        variables and the corresponding independenc I value
        
    """
    
    #number of variables for decomposition
    numvar_to_use = 4
    
    #variable used in decomposition
    variables = ['Lig_fit','Lig2_fit','Tg','Tbg','tau_g','rLig','DeltaLgi','Lb_fit','DeltaLigb']
    
    #subselect used variables in dataframe
    coli_to_test = colidata.copy()
    coli_to_test['tau_g'] = 1/coli_to_test['tau_g'] 
    coli_to_test = coli_to_test[['Lig_fit','Lig2_fit','Tg','Tbg','tau_g','rLig','DeltaLgi','Lb_fit','DeltaLigb']].dropna()
    
    #define numeric code for each variable (1,10,100,1000 etc.)
    code = [10**i for i in range(len(variables))]
    
    #define groups of variables related by an equation 
    equations = [['Lig2_fit','Lig_fit','tau_g','Tg'],['DeltaLgi','Lig2_fit','Lig_fit'],['rLig','Lig2_fit','Lig_fit'],
    ['Lb_fit','tau_g','Lig_fit','Tbg'],['DeltaLigb','Lb_fit','Lig_fit']]

    #Generate all possible combinations of variables by brute force
    combinations = []
    for j in range(1000):
        current = np.random.choice(variables,numvar_to_use,replace=False)
        cur_unchanged = current.copy()
        for i in range(10):
            for x in equations:
                if len(set(x) & set(current))==len(x)-1:

                    current = set.union(set(current),set(x))
        #print(current)
        #print(len(current))
        if len(current)==len(variables):
            combinations.append(cur_unchanged)

    #calculate numeric code for each combination. This is the same irrespective of the order of variables. Keep
    #only single occurrennces of each code
    finalset = set([np.sum([code[np.where(np.array(variables)==combinations[j][i])[0][0]] for i in range(numvar_to_use)]) for j in range(len(combinations))])
    
    #recover the variable names corresponding to the codes defined above
    combinations = [np.array(variables)[np.array([np.mod(np.floor_divide(list(finalset)[j],10**(i)),10) 
                                   for i in range(len(variables))])==1] for j in range(len(list(finalset)))]

    #calculate independence for each combination and sort them
    indep = [np.linalg.det(np.cov(coli_to_test[combinations[i]].T))/np.prod(np.diag(np.cov(coli_to_test[combinations[i]].T))) 
     for i in range(len(combinations))]
    sorted_combinations = np.flipud(np.array(combinations)[np.argsort(indep)])
    sorted_vals = np.flipud(np.array(indep)[np.argsort(indep)])

    s = ", ";
    rows = [s.join(list(x)) for x in sorted_combinations]

    table_to_plot = [[rows[x], np.around(sorted_vals[x],3)] for x in range(len(sorted_combinations))]

    return coli_to_test, sorted_combinations, sorted_vals, table_to_plot

def decompose_division_adder(colidata):
    """Decomposition of data only for the division cycle
    
    Parameters
    ----------
    colidata : Pandas dataframe
        dataframe with cell cycle information
        
    
    Returns
    -------
    coli_to_test : Pandas dataframe
        subset of data used for decomposition
    sorted_combinations : array of string lists
        each element of the array is a quadruplet of variable names
        corresponding to a decomposition. The list is sorted from 
        best to worst decomposition
    sorted_vals : array of floats
        independence I value for sorted decompositions
    table_to_plot : list of lists
        Each element of the list is a pair of values: the list of 
        variables and the corresponding independenc I value
        
    """
    
    #number of variables for decomposition
    numvar_to_use = 3
    
    #variable used in decomposition
    variables = ['Lb_fit','Ld_fit','Td','tau_fit','DeltaL']
    
    #subselect used variables in dataframe
    coli_to_test = colidata.copy()
    coli_to_test['tau_fit'] = 1/coli_to_test['tau_fit']
    coli_to_test = coli_to_test[['Lb_fit','Ld_fit','Td','tau_fit','DeltaL']].dropna()
    
    #define numeric code for each variable (1,10,100,1000 etc.)
    code = [10**i for i in range(len(variables))]
    
    #define groups of variables related by an equation (e.g. Ld = Lb*e^Td/tau_fit)
    equations = [['Ld_fit','Lb_fit','tau_fit','Td'],['DeltaL','Ld_fit','Lb_fit']]

    #Generate all possible combinations of variables by brute force
    combinations = []
    for j in range(1000):
        current = np.random.choice(variables,numvar_to_use,replace=False)
        cur_unchanged = current.copy()
        for i in range(10):
            for x in equations:
                if len(set(x) & set(current))==len(x)-1:

                    current = set.union(set(current),set(x))
        #print(current)
        #print(len(current))
        if len(current)==len(variables):
            combinations.append(cur_unchanged)

    #calculate numeric code for each combination. This is the same irrespective of the order of variables. Keep
    #only single occurrennces of each code
    finalset = set([np.sum([code[np.where(np.array(variables)==combinations[j][i])[0][0]] for i in range(numvar_to_use)]) for j in range(len(combinations))])
    
    #recover the variable names corresponding to the codes defined above
    combinations = [np.array(variables)[np.array([np.mod(np.floor_divide(list(finalset)[j],10**(i)),10) 
                                   for i in range(len(variables))])==1] for j in range(len(list(finalset)))]

    #calculate independence for each combination and sort them
    indep = [np.linalg.det(np.cov(coli_to_test[combinations[i]].T))/np.prod(np.diag(np.cov(coli_to_test[combinations[i]].T))) 
     for i in range(len(combinations))]
    sorted_combinations = np.flipud(np.array(combinations)[np.argsort(indep)])
    sorted_vals = np.flipud(np.array(indep)[np.argsort(indep)])

    #create output table
    s = ", ";
    rows = [s.join(list(x)) for x in sorted_combinations]

    table_to_plot = [[rows[x], np.around(sorted_vals[x],3)] for x in range(len(sorted_combinations))]
    return coli_to_test, sorted_combinations, sorted_vals, table_to_plot



def cross_generation_construct(colidata, datatype, time_scale):
    """Create a new replication centric cell cycle dataframe 
    
    Parameters
    ----------
    colidata : Pandas dataframe
        dataframe with cell cycle information¨
    datatype : str
        experimental (exp) or simulation (simul) data
    time_scale : int
        time scale
        
    
    Returns
    -------
    colidata : Pandas dataframe
        original dataframe completed with replication cenric variables
    
    """
    
    #place holders for new variables
    colidata['Tg'] = np.nan
    colidata['Lig_fit'] = np.nan
    colidata['Lig2_fit'] = np.nan
    colidata['tau_g'] = np.nan
    colidata['pearsonlog_g'] = np.nan
    colidata['pearsonlin_g'] = np.nan

    cross_gen_list = [[] for x in colidata.index]

    for ind,x in enumerate(colidata.index):
        if colidata.at[x,'mother_id']>0:
            daughter_Ti = colidata.at[x,'Ti']/time_scale
            mother_Ti = colidata.at[colidata.at[x,'mother_id'],'Ti']/time_scale

            if ~np.isnan(daughter_Ti):
                if daughter_Ti>0: 
                    daughter_len = colidata.at[x,'length']  

                    #find daughters id
                    sisters = colidata[colidata.mother_id == colidata.loc[x].mother_id]

                    if (mother_Ti>0):

                        if datatype == 'exp':
                            if len(sisters)!=2:
                                continue

                        #calculate division ratio for current cell
                        if datatype == 'exp':
                            if sisters.iloc[0].name == x:
                                rfact = sisters.iloc[0].Lb_fit/(sisters.iloc[0].Lb_fit+sisters.iloc[1].Lb_fit)
                            else:
                                rfact = sisters.iloc[1].Lb_fit/(sisters.iloc[0].Lb_fit+sisters.iloc[1].Lb_fit)
                        else:
                            rfact = colidata.at[x,'rfact']   

                        mother_len = colidata.at[colidata.at[x,'mother_id'],'length']

                        #create a length list combininig mother length from initiation to division multiplied 
                        #by division ratio (to avoid creating discontinuity) with the current cell length 
                        #from birth to initiation
                        crossgen_len = np.concatenate((rfact*mother_len[int(mother_Ti)::],daughter_len[0:int(daughter_Ti)]))
                        colidata.loc[x,'Tg'] = len(crossgen_len)
                        colidata.loc[x,'Tbg'] = len(mother_len[int(mother_Ti)::])
                        cross_gen_list[ind] = crossgen_len

                        #calculate an exponential fit on the cross-generation length and store parameters
                        ydata = crossgen_len
                        T = colidata.at[x, 'Tg']
                        Lig_fit, Lig2_fit, tau_g, pearson_lin, pearson_log = fit_exp(ydata,T)
                        colidata.at[x,'Lig_fit']= Lig_fit
                        colidata.at[x,'Lig2_fit']= Lig2_fit
                        colidata.at[x,'tau_g']= tau_g
                        colidata.at[x,'pearsonlin_g']= pearson_lin
                        colidata.at[x,'pearsonlog_g']= pearson_log
                else:
                    if mother_Ti>0:
                        mother_len = colidata.at[colidata.at[x,'mother_id'],'length']
                        crossgen_len = 0.5*mother_len[int(mother_Ti):len(mother_len)+int(daughter_Ti)]
                        colidata.loc[x,'Tg'] = len(crossgen_len)
                        colidata.loc[x,'Tbg'] = len(mother_len[int(mother_Ti)::])
                        cross_gen_list[ind] = crossgen_len
                        
                        if len(crossgen_len)<3:
                            continue
                        ydata = crossgen_len
                        T = colidata.at[x, 'Tg']
                        Lig_fit, Lig2_fit, tau_g, pearson_lin, pearson_log = fit_exp(ydata,T)
                        colidata.at[x,'Lig_fit']= Lig_fit
                        colidata.at[x,'Lig2_fit']= Lig2_fit
                        colidata.at[x,'tau_g']= tau_g
                        colidata.at[x,'pearsonlin_g']= pearson_lin
                        colidata.at[x,'pearsonlog_g']= pearson_log

    colidata['cross_gen_len'] = pd.Series(cross_gen_list,index=colidata.index)
    
    if datatype == 'simul':
        colidata = colidata[colidata.born>1000]
    else:
        colidata = colidata[colidata.period==1]
        colidata=colidata[colidata.pearson_log>0.95]
        colidata=colidata[colidata.tau_fit>0]
        
    #complete information table ready to be usd for decomposition analysis
    colidata.loc[:,'DeltaL']= colidata['Ld_fit']-colidata['Lb_fit']
    colidata.loc[:,'DeltaLi']= colidata['Li_fit']-colidata['Lb_fit']
    colidata.loc[:,'DeltaLid']= colidata['Ld_fit']-colidata['Li_fit']
    colidata.loc[:,'DeltaLib']= colidata['Li_fit']-colidata['Lb_fit']
    colidata.loc[:,'DeltaTid']= colidata['Td']-colidata['Ti']
    colidata.loc[:,'rLdLb']= colidata['Ld_fit']/colidata['Lb_fit']
    
    colidata.loc[:,'rLig']= colidata['Lig2_fit']/colidata['Lig_fit']
    colidata.loc[:,'DeltaLgi']= colidata['Lig2_fit']-colidata['Lig_fit']
    colidata.loc[:,'DeltaLigb'] = 0.5*colidata.apply(lambda row: tmo.mother_var(row, colidata,'DeltaLid'),axis = 1)

    return colidata


def fit_exp(ydata, T):
    """Exponential fit for growth curve
    
    Parameters
    ----------
    ydata : numpy array
        length data
    T : numpy array
        time data
        
    
    Returns
    -------
    Lig_fit : float
        "birth length"
    Lig2_fit : float
        "division length"
    tau_g : float
        inverse growth rate
    pearson_lin : float
        pearson for linear fit
    pearson_log : float
        pearson for log fit
    
    """
    
    xdata = range(len(ydata))
    tau0 = T
    popt, pcov = scipy.optimize.curve_fit(tgw.fun_expgrowht2, xdata, ydata, p0=[15,tau0])
    tau_g = popt[1]
    Lig_fit = popt[0]
    Lig2_fit = tgw.fun_expgrowht2(T,Lig_fit,tau_g)
    pearson_lin = scipy.stats.pearsonr(xdata, ydata)[0]
    pearson_log = scipy.stats.pearsonr(xdata, np.log(ydata))[0]
    
    return Lig_fit, Lig2_fit, tau_g, pearson_lin, pearson_log

def plot_combinations(coli_to_test, sorted_combinations, sorted_vals, comb_ind, renaming_fun, numvar):
    """Plot a given decomposition as a coloured matrix
    
    Parameters
    ----------
    coli_to_test : Pandas dataframe
        cell cycle dataframe
    sorted_combinations : array of string lists
        each element of the array is a quadruplet of variable names
        corresponding to a decomposition. The list is sorted from 
        best to worst decomposition
    sorted_vals : array of floats
        independence I value for sorted decompositions
    comb_ind : int
        index of decomposition to plot
    renaming_fun : str
        name of function to use for renaming variables
    numvar : number of variables of the decomposition
        
    
    Returns
    -------
    fig, ax : matplotlib handles
        matplotlib reference to plot
    
    """
    c = list(itertools.product(sorted_combinations[comb_ind], sorted_combinations[comb_ind]))
    c = [[str(x) for x in y] for y in c]
    pairwise = np.reshape([scipy.stats.pearsonr(coli_to_test[c[x][0]],coli_to_test[c[x][1]])[0] 
                         for x in np.arange(len(c))],(numvar,numvar))
    names = np.array(np.split(np.array([renaming_fun(x) for x in c]),numvar))
    fig, ax = plt.subplots(figsize=(7,7))
    plt.imshow(pairwise,cmap = 'seismic',vmin=-1,vmax = 1)
    for i in range(names.shape[0]):
        for j in range(names.shape[0]):
            if np.abs(pairwise[i,j])>0.5:
                 col = 'white'
            else:
                col = 'black'
            plt.text(x=i, y=j-0.1, s = names[i,j][0], color = col,size = 30, horizontalalignment='center')
            plt.text(x=i, y=j+0.3, s = names[i,j][1], color = col, size = 30, horizontalalignment='center')

    ax.set_axis_off()
    ax.set_title(', '.join(list(np.unique(names)))+
                 ',\n I: '+str(np.around(sorted_vals[comb_ind],3)),fontsize = 30)
    #plt.show()
    return fig,ax

def plot_combinations_9array(coli_to_test, sorted_combinations, sorted_vals, comb_ind, renaming_fun):
    """Plot the nine best decompositions of a given set with variables
    inside the matrix
    
    Parameters
    ----------
    coli_to_test : Pandas dataframe
        cell cycle dataframe
    sorted_combinations : array of string lists
        each element of the array is a quadruplet of variable names
        corresponding to a decomposition. The list is sorted from 
        best to worst decomposition
    sorted_vals : array of floats
        independence I value for sorted decompositions
    comb_ind : numpy array
        list of indices to plot
    renaming_fun : str
        name of function to use for renaming variables
        
    
    Returns
    -------
    fig : matplotlib handles
        matplotlib reference to plot
    
    """
    
    fig, axes = plt.subplots(figsize=(20,20))
    axes.set_axis_off()
    for ind, comb in enumerate(comb_ind):
        c = list(itertools.product(sorted_combinations[comb], sorted_combinations[comb]))
        c = [[str(x) for x in y] for y in c]
        pairwise = np.reshape([scipy.stats.pearsonr(coli_to_test[c[x][0]],coli_to_test[c[x][1]])[0] 
                             for x in np.arange(len(c))],(4,4))
        names = np.array(np.split(np.array([renaming_fun(x) for x in c]),4))

        ax = fig.add_subplot(3, 3, ind+1)
        ax.imshow(pairwise,cmap = 'seismic',vmin=-1,vmax = 1)
        for i in range(names.shape[0]):
            for j in range(names.shape[0]):
                if np.abs(pairwise[i,j])>0.5:
                     col = 'white'
                else:
                    col = 'black'
                plt.text(x=i-0.1, y=j-0.2, s = names[i,j][0], color = col,size = 25)
                plt.text(x=i-0.1, y=j+0.2, s = names[i,j][1], color = col, size = 25)

        ax.set_axis_off()
        ax.set_title(', '.join(list(np.unique(names)))+
                     ',\n I: '+str(np.around(sorted_vals[comb],3)),fontsize = 25)
    fig.subplots_adjust(hspace = 0.4)

    #plt.show()
    return fig

def plot_combinations_9array_v2(coli_to_test, sorted_combinations, sorted_vals, comb_ind, renaming_fun):
    """Plot the nine best decompositions of a given set with variables
    outside the matrix
    
    Parameters
    ----------
    coli_to_test : Pandas dataframe
        cell cycle dataframe
    sorted_combinations : array of string lists
        each element of the array is a quadruplet of variable names
        corresponding to a decomposition. The list is sorted from 
        best to worst decomposition
    sorted_vals : array of floats
        independence I value for sorted decompositions
    comb_ind : numpy array
        list of indices to plot
    renaming_fun : str
        name of function to use for renaming variables
        
    
    Returns
    -------
    fig : matplotlib handles
        matplotlib reference to plot
    
    """
    
    fig, axes = plt.subplots(figsize=(20,20))
    axes.set_axis_off()
    for ind, comb in enumerate(comb_ind):
        c = list(itertools.product(sorted_combinations[comb], sorted_combinations[comb]))
        c = [[str(x) for x in y] for y in c]
        pairwise = np.reshape([scipy.stats.pearsonr(coli_to_test[c[x][0]],coli_to_test[c[x][1]])[0] 
                             for x in np.arange(len(c))],(4,4))
        names = np.array(np.split(np.array([renaming_fun(x) for x in c]),4))

        ax = fig.add_subplot(3, 3, ind+1)
        ax.imshow(pairwise,cmap = 'seismic',vmin=-1,vmax = 1)
        for i in range(names.shape[0]):
            for j in range(names.shape[0]):
                if np.abs(pairwise[i,j])>0.5:
                     col = 'white'
                else:
                    col = 'black'
                #plt.text(x=i-0.1, y=j-0.15, s = names[i,j][0], color = col,size = 30)
                #plt.text(x=i-0.1, y=j+0.3, s = names[i,j][1], color = col, size = 30)
        
        for i in range(names.shape[0]):
            plt.text(x=-0.6, y=i+0.2, s = names[i,0][0], color = 'black',size = 35,
                           horizontalalignment = 'right')
            plt.text(x=i-0.0, y=-0.65, s = names[0,i][1], color = 'black',size = 35,
                    horizontalalignment = 'center') 
        ax.set_axis_off()
        #ax.set_title(', '.join(list(np.unique(names)))+
        #             ',\n I: '+str(np.around(sorted_vals[comb],3)),fontsize = 25, pad = 50)
        ax.set_title('I: '+str(np.around(sorted_vals[comb],3)),fontsize = 35, pad = 55)
    fig.subplots_adjust(hspace = 0.4)

    #plt.show()
    return fig

def plot_combinations_9array3x3(coli_to_test, sorted_combinations, sorted_vals, comb_ind, renaming_fun):
    """Plot the nine best decompositions of a given set with variables
    inside the matrix for a decomposition of 3 variables
    
    Parameters
    ----------
    coli_to_test : Pandas dataframe
        cell cycle dataframe
    sorted_combinations : array of string lists
        each element of the array is a quadruplet of variable names
        corresponding to a decomposition. The list is sorted from 
        best to worst decomposition
    sorted_vals : array of floats
        independence I value for sorted decompositions
    comb_ind : numpy array
        list of indices to plot
    renaming_fun : str
        name of function to use for renaming variables
        
    
    Returns
    -------
    fig : matplotlib handles
        matplotlib reference to plot
    
    """
    
    fig, axes = plt.subplots(figsize=(20,20))
    axes.set_axis_off()
    for ind, comb in enumerate(comb_ind):
        c = list(itertools.product(sorted_combinations[comb], sorted_combinations[comb]))
        c = [[str(x) for x in y] for y in c]
        pairwise = np.reshape([scipy.stats.pearsonr(coli_to_test[c[x][0]],coli_to_test[c[x][1]])[0] 
                             for x in np.arange(len(c))],(3,3))
        names = np.array(np.split(np.array([renaming_fun(x) for x in c]),3))

        ax = fig.add_subplot(3, 3, ind+1)
        ax.imshow(pairwise,cmap = 'seismic',vmin=-1,vmax = 1)

        for i in range(names.shape[0]):
            for j in range(names.shape[0]):
                if np.abs(pairwise[i,j])>0.5:
                     col = 'white'
                else:
                    col = 'black'
                plt.text(x=i-0.1, y=j-0.2, s = names[i,j][0], color = col,size = 20)
                plt.text(x=i-0.1, y=j+0.2, s = names[i,j][1], color = col, size = 20)
                
        ax.set_axis_off()
        ax.set_title(', '.join(list(np.unique(names)))+
                     ',\n indep: '+str(np.around(sorted_vals[comb],3)),fontsize = 20)
        if ind==7:
            break
    fig.subplots_adjust(hspace = 0.4)

    #plt.show()
    return fig

def plot_combinations_9array3x3_v2(coli_to_test, sorted_combinations, sorted_vals, comb_ind, renaming_fun):
    """Plot the nine best decompositions of a given set with variables
    outside the matrix for a decomposition of 3 variables
    
    Parameters
    ----------
    coli_to_test : Pandas dataframe
        cell cycle dataframe
    sorted_combinations : array of string lists
        each element of the array is a quadruplet of variable names
        corresponding to a decomposition. The list is sorted from 
        best to worst decomposition
    sorted_vals : array of floats
        independence I value for sorted decompositions
    comb_ind : numpy array
        list of indices to plot
    renaming_fun : str
        name of function to use for renaming variables
        
    
    Returns
    -------
    fig : matplotlib handles
        matplotlib reference to plot
    
    """
    
    fig, axes = plt.subplots(figsize=(20,20))
    axes.set_axis_off()
    for ind, comb in enumerate(comb_ind):
        c = list(itertools.product(sorted_combinations[comb], sorted_combinations[comb]))
        c = [[str(x) for x in y] for y in c]
        pairwise = np.reshape([scipy.stats.pearsonr(coli_to_test[c[x][0]],coli_to_test[c[x][1]])[0] 
                             for x in np.arange(len(c))],(3,3))
        names = np.array(np.split(np.array([renaming_fun(x) for x in c]),3))

        ax = fig.add_subplot(3, 3, ind+1)
        ax.imshow(pairwise,cmap = 'seismic',vmin=-1,vmax = 1)

        for i in range(names.shape[0]):
            for j in range(names.shape[0]):
                if np.abs(pairwise[i,j])>0.5:
                     col = 'white'
                else:
                    col = 'black'
                #plt.text(x=i-0.1, y=j-0.2, s = names[i,j][0], color = col,size = 30)
                #plt.text(x=i-0.1, y=j+0.2, s = names[i,j][1], color = col, size = 30)
                
        for i in range(names.shape[0]):
            plt.text(x=-0.6, y=i+0.2, s = names[i,0][0], color = 'black',size = 35,
                           horizontalalignment = 'right')
            plt.text(x=i-0.0, y=-0.6, s = names[0,i][1], color = 'black',size = 35,
                    horizontalalignment = 'center') 
        
        ax.set_axis_off()
        
        ax.set_title('I: '+str(np.around(sorted_vals[comb],3)),fontsize = 35, pad = 55)
                
        
        if ind==7:
            break
    fig.subplots_adjust(hspace = 0.4)

    #plt.show()
    return fig


def plot_exp_simul(coli_exp, coli_simul, sorted_combinations, sorted_vals_exp,sorted_vals_simul, 
                   renaming_fun, comb1 =0, comb2=0):
    """Plot the best decomposition both for experimental and simulation
    data in the same matrix using the lower left and upper right corners
    
    Parameters
    ----------
    coli_exp : Pandas dataframe
        cell cycle dataframe of experimental data
    coli_simul : Pandas dataframe
        cell cycle dataframe of simulation data 
    sorted_combinations : array of string lists
        each element of the array is a quadruplet of variable names
        corresponding to a decomposition. The list is sorted from 
        best to worst decomposition
    sorted_vals : array of floats
        independence I value for sorted decompositions
    renaming_fun : str
        name of function to use for renaming variables
    comb1 : int
        index of decomposition to plot for experimental data
        (to use if the best decomposition is not the same
        in experiments and simulations)
    comb2 : int
        index of decomposition to plot for simulation data
        (to use if the best decomposition is not the same
        in experiments and simulations)
        
    
    Returns
    -------
    fig : matplotlib handles
        matplotlib reference to plot
    
    """
    
    fig, axes = plt.subplots(figsize=(7,7))
    axes.set_axis_off()
    
    comb = 1
    #//////////////
    c = list(itertools.product(sorted_combinations[comb], sorted_combinations[comb]))
    c = [[str(x) for x in y] for y in c]
    pairwise1 = np.reshape([scipy.stats.pearsonr(coli_exp[c[x][0]],coli_exp[c[x][1]])[0] 
                         for x in np.arange(len(c))],(4,4))
    pairwise2 = np.reshape([scipy.stats.pearsonr(coli_simul[c[x][0]],coli_simul[c[x][1]])[0] 
                         for x in np.arange(len(c))],(4,4))
    names = np.array(np.split(np.array([renaming_fun(x) for x in c]),4))

    combined = np.zeros(pairwise1.shape)
    for i in range(names.shape[0]):
        for j in range(names.shape[0]):
            if i>j:
                combined[i,j] = pairwise1[i,j]
            elif i==j:
                combined[i,j] = 0
                
            else:
                combined[i,j] = pairwise2[i,j]

    #ax = fig.add_subplot(3, 3, ind+1)
    axes.imshow(combined,cmap = 'seismic',vmin=-1,vmax = 1)
    for i in range(names.shape[0]):
        for j in range(names.shape[0]):

            if i==j:
                col = 'black'
                t = plt.text(x=i-0.015*len(names[i,j][0]), y=j+0.1, s = names[i,j][0], color = col,size = 28, fontweight= 'bold')
                r = fig.canvas.get_renderer()
                bb = t.get_window_extent(renderer=r)
                width = bb.width
                height = bb.height
                t.set_x(i-0.005*width)

                #plt.text(x=i-0.1, y=j+0.2, s = names[i,j][1], color = col, size = 20)
            elif np.abs(combined[i,j])>0.99:
                col = 'white'
                plt.text(x=i-0.0, y=j-0.1, s = names[i,j][0], color = col,size = 25, horizontalalignment='center')
                plt.text(x=i-0.0, y=j+0.3, s = names[i,j][1], color = col, size = 25, horizontalalignment='center')
            else:
                col = 'black'
                plt.text(x=i-0.0, y=j-0.1, s = names[i,j][0], color = col,size = 25, horizontalalignment='center')
                plt.text(x=i-0.0, y=j+0.3, s = names[i,j][1], color = col, size = 25, horizontalalignment='center')

    
        
    #ax.set_axis_off()
    axes.set_title('---   Experimental I: '+str(np.around(sorted_vals_exp[comb1],3))+
                   '\n'+'$-$'+'   Simulation I: '+str(np.around(sorted_vals_simul[comb2],3)),fontsize = 25)
        #//////////////////
    #fig.subplots_adjust(hspace = 0.4)

    #plt.show()
    return fig