"""
This module allows to simulate E.coli cell cycles following
different models.
 """
# Author: Guillaume Witz, Biozentrum Basel, 2019
# License: MIT License


import numpy as np
import pandas as pd
import copy

def simul_doubleadder(nbstart, run_time, params, name, nori_init=1):
    """Simulate double-adder model

    Parameters
    ----------
    nbstart : int
        number of cells to simulate
    run_time: int
        number of iterations
    params: dict
        experimental parameters
    name: str
        name of runs

    Returns
    -------
    cells : list of dict
        Each element of the list is a cell cycle defined by a
        dictionary of features (Lb, Ld etc.)
    """

    #initialize birth length and growth rate
    L0 = np.exp(np.random.normal(params['Lb_logn_mu'],params['Lb_logn_sigma'],size=nbstart))
    tau = np.exp(np.random.normal(params['tau_logn_mu'], params['tau_logn_sigma'], size=nbstart))

    #standard value of growth rate. Used to scale the noise appropriately
    normval = np.exp(params['tau_logn_mu'])

    #initialize the inter-initiation adder (exact procedure doesn't really matter here)
    #as all cells start with n_ori = 1, there's no initiation to division adder running
    DLi = np.random.normal(params['DLi_mu'], params['DLi_sigma'], size=nbstart)*nori_init

    #initialize cell infos as a list of dictionaries. All cells start with n_ori = 1
    cells = {}
    for x in range(nbstart):
        dict1 = {'Lb': L0[x],'L':L0[x], 'gen': str(x), 'tau':tau[x], 'Lt': [[0,L0[x],1]], 'finish': False,
                'born':0, 'DLi': [[0,DLi[x]]],'DLdLi': [],'Li':[],'Ti':[],
                'numori': nori_init,'Ld':np.nan, 'numori_born': nori_init,'name': name,'mLi':np.nan, 'mLd':np.nan, 'rfact':0.5}
        cells[str(x)] = dict1


    for t in range(run_time):

        divide_cell = []

        for x in cells:
            if cells[x]['finish']==False:


                #update cell size
                cells[x]['L'] = cells[x]['L']*(2**(1/cells[x]['tau']))
                cells[x]['Lt'].append([t,cells[x]['L'],cells[x]['numori']])

                #increment the most recent inter-initiation adder
                cells[x]['DLi'][-1][0] = cells[x]['DLi'][-1][0]+(cells[x]['Lt'][-1][1]-cells[x]['Lt'][-2][1])

                #if at least one volume counter since RI is running, increment all of them
                if len(cells[x]['DLdLi'])>0:
                    cells[x]['DLdLi'] = [[k[0]+(cells[x]['Lt'][-1][1]-cells[x]['Lt'][-2][1]),k[1]] for k in cells[x]['DLdLi']]

                #if a volume counter has reached its limit divide
                if len(cells[x]['DLdLi'])>0:
                    if (cells[x]['numori']>1) and (cells[x]['DLdLi'][0][0]>cells[x]['DLdLi'][0][1]):
                        cells[x]['finish'] = True#tag cell as finished
                        cells[x]['Ld'] = cells[x]['L']
                        cells[x]['Td'] = len(cells[x]['Lt'])
                        cells[x]['Td_abs'] = t
                        cells[x]['d_Ld_Lb'] = cells[x]['L']-cells[x]['Lb']

                        #assign the correct adders (the oldest ones) to the cell that just divided
                        cells[x]['final_DLdLi'] = cells[x]['DLdLi'][0][1]
                        cells[x]['final_DLi'] = cells[x]['DLi'][0][1]
                        cells[x]['final_Li'] = cells[x]['Li'][0]

                        #for each accumulated variable suppress the oldest one
                        if len(cells[x]['DLdLi'])==1:
                            cells[x]['DLdLi'] = []
                        else:
                            cells[x]['DLdLi'].pop(0)

                        if len(cells[x]['DLi'])==1:
                            cells[x]['DLi'] = []
                        else:
                            cells[x]['DLi'].pop(0)

                        if len(cells[x]['Li'])==1:
                            cells[x]['Li'] = []
                        else:
                            cells[x]['Li'].pop(0)
                        divide_cell.append(x)

                #if the added volume has reached its limit make new RI
                if cells[x]['DLi'][-1][0]>cells[x]['DLi'][-1][1]:

                    #duplicate origin
                    cells[x]['numori'] = cells[x]['numori']*2

                    #define new adder
                    newdli = cells[x]['numori']*np.random.normal(params['DLi_mu'], params['DLi_sigma'])

                    cells[x]['DLi'].append([0,newdli])

                    cells[x]['Li'].append(cells[x]['L'])

                    #temporarilly store Ti as absolute time
                    cells[x]['Ti'].append(t)

                    #define new adder
                    new_dv = cells[x]['numori']*np.exp(np.random.normal(params['DLdLi_logn_mu'], params['DLdLi_logn_sigma']))

                    cells[x]['DLdLi'].append([0,new_dv])

        for x in divide_cell:

            #Draw division ratio
            rfact = 1/(1+np.random.normal(1,params['div_ratio']))

            #Create new cell using mother information
            new_tau = np.exp(correlated_normal(np.log(cells[x]['tau']), params['tau_logn_mu'], params['tau_logn_sigma'], params['tau_corr']))
            new_Lb = copy.deepcopy(rfact*cells[x]['L'])
            new_L = copy.deepcopy(rfact*cells[x]['L'])
            new_Lt = [[t,copy.deepcopy(rfact*cells[x]['L']),copy.deepcopy(cells[x]['numori'])/2]]
            new_DLi = copy.deepcopy([[rfact*y[0],rfact*y[1]] for y in cells[x]['DLi']])
            new_DLdLi = copy.deepcopy([[rfact*y[0],rfact*y[1]] for y in cells[x]['DLdLi']])
            new_Li = copy.deepcopy([rfact*y for y in cells[x]['Li']])
            new_numori = copy.deepcopy(cells[x]['numori'])/2
            mother_initL = copy.deepcopy(cells[x]['final_Li'])/2
            mother_Ld = copy.deepcopy(cells[x]['Ld'])

            dict1 = {'Lb': new_Lb,'L': new_L, 'gen': str(x)+'B', 'tau': new_tau,'Lt': new_Lt, 'finish': False,
                     'born':t, 'DLi': new_DLi,'DLdLi': new_DLdLi,'Li':new_Li,'Ti':[], 'numori':new_numori,
                     'numori_born':copy.deepcopy(new_numori),'Ld':np.nan, 'name': name,'mLi': mother_initL, 'mLd':mother_Ld,
                    'rfact':rfact}

            cells[x+'B'] = copy.deepcopy(dict1)


            #keep oldest timer as final timer and give daughter remaining ones. Caclulate initiation time based on cell birth.
            TL_S_val = copy.deepcopy(cells[x]['Ti'].pop(0))
            cells[x+'B']['Ti'] = copy.deepcopy(cells[x]['Ti'])
            cells[x]['Ti'] = TL_S_val-copy.deepcopy(cells[x]['born'])

    for x in cells:
        if len(cells[x]['Li'])>0:
            cells[x]['Li'] = np.nan

    return cells


def simul_growth_dinter_classicadder(nbstart, run_time, params, name, nori_init=1):
    """Simulate a model with inter-initiation per origin adder and
    classic division adder (Ld = Lb+dL)

    Parameters
    ----------
    nbstart : int
        number of cells to simulate
    run_time: int
        number of iterations
    params: dict
        experimental parameters
    name: str
        name of runs

    Returns
    -------
    cells : list of dict
        Each element of the list is a cell cycle defined by a
        dictionary of features (Lb, Ld etc.)
    """

    #initialize birth length and growth rate
    L0 = np.exp(np.random.normal(params['Lb_logn_mu'],params['Lb_logn_sigma'],size=nbstart))
    tau = np.exp(np.random.normal(params['tau_logn_mu'], params['tau_logn_sigma'], size=nbstart))

    #standard value of growth rate. Used to scale the noise appropriately
    normval = np.exp(params['tau_logn_mu'])

    #initialize the inter-initiation adder (exact procedure doesn't really matter here)
    #as all cells start with n_ori = 1, there's no initiation to division adder running
    DLi = np.random.normal(params['DLi_mu'], params['DLi_sigma'], size=nbstart)*nori_init

    #initialize classic adder
    dL = np.random.normal(params['dL_mu'], params['dL_sigma'], size=nbstart)

    #initialize cell infos as a list of dictionaries. All cells start with n_ori = 1
    cells = {}
    for x in range(nbstart):
        dict1 = {'Lb': L0[x],'L':L0[x], 'gen': str(x), 'tau':tau[x], 'Lt': [[0,L0[x],1]], 'finish': False,
                'born':0, 'DLi': [[0,DLi[x]]]*nori_init,'DLdLi': [[0,1]]*(nori_init-1),'Li':[0]*nori_init,'Ti':[0]*nori_init, 'dL': [0,dL[x]],
                'numori': nori_init, 'Ld':np.nan, 'numori_born': nori_init,'name': name,'mLi':np.nan, 'mLd':np.nan, 'rfact':0.5}
        cells[str(x)] = dict1


    for t in range(run_time):

        divide_cell = []

        for x in cells:
            if cells[x]['finish']==False:


                #update cell size
                cells[x]['L'] = cells[x]['L']*(2**(1/cells[x]['tau']))
                cells[x]['Lt'].append([t,cells[x]['L'],cells[x]['numori']])

                #increment the most recent inter-initiation adder
                cells[x]['DLi'][-1][0] = cells[x]['DLi'][-1][0]+(cells[x]['Lt'][-1][1]-cells[x]['Lt'][-2][1])

                #increment adder
                cells[x]['dL'][0] = cells[x]['dL'][0]+(cells[x]['Lt'][-1][1]-cells[x]['Lt'][-2][1])

                #if at least one volume counter since RI is running, increment all of them
                if len(cells[x]['DLdLi'])>0:
                    cells[x]['DLdLi'] = [[k[0]+(cells[x]['Lt'][-1][1]-cells[x]['Lt'][-2][1]),k[1]] for k in cells[x]['DLdLi']]


                if (cells[x]['numori']>1) and (cells[x]['dL'][0]>cells[x]['dL'][1]):
                    cells[x]['finish'] = True#tag cell as finished
                    cells[x]['Ld'] = cells[x]['L']
                    cells[x]['Td'] = len(cells[x]['Lt'])
                    cells[x]['Td_abs'] = t
                    cells[x]['d_Ld_Lb'] = cells[x]['L']-cells[x]['Lb']

                    #assign the correct adders (the oldest ones) to the cell that just divided
                    cells[x]['final_DLi'] = cells[x]['DLi'][0][1]
                    cells[x]['final_Li'] = cells[x]['Li'][0]
                    cells[x]['final_DLdLi'] = cells[x]['DLdLi'][0][0]

                    #for each accumulated variable suppress the oldest one
                    if len(cells[x]['DLi'])==1:
                        cells[x]['DLi'] = []
                    else:
                        cells[x]['DLi'].pop(0)

                    if len(cells[x]['Li'])==1:
                        cells[x]['Li'] = []
                    else:
                        cells[x]['Li'].pop(0)

                    if len(cells[x]['DLdLi'])==1:
                            cells[x]['DLdLi'] = []
                    else:
                        cells[x]['DLdLi'].pop(0)

                    divide_cell.append(x)

                #if the added volume has reached its limit make new RI
                if cells[x]['DLi'][-1][0]>cells[x]['DLi'][-1][1]:

                    #duplicate origin
                    cells[x]['numori'] = cells[x]['numori']*2

                    #Version where adder is noisy itself
                    newdli = cells[x]['numori']*np.random.normal(params['DLi_mu'], params['DLi_sigma'])

                    cells[x]['DLi'].append([0,newdli])

                    cells[x]['Li'].append(cells[x]['L'])

                    #temporarilly store TL_S as absolute time
                    cells[x]['Ti'].append(t)

                    cells[x]['DLdLi'].append([0,0])

        for x in divide_cell:

            #Draw division ratio
            rfact = 1/(1+np.random.normal(1,params['div_ratio']))

            #Create new cell using mother information
            new_tau = np.exp(correlated_normal(np.log(cells[x]['tau']), params['tau_logn_mu'], params['tau_logn_sigma'], params['tau_corr']))
            new_Lb = copy.deepcopy(rfact*cells[x]['L'])
            new_L = copy.deepcopy(rfact*cells[x]['L'])
            new_Lt = [[t,copy.deepcopy(rfact*cells[x]['L']),copy.deepcopy(cells[x]['numori'])/2]]
            new_DLi = copy.deepcopy([[rfact*y[0],rfact*y[1]] for y in cells[x]['DLi']])
            new_Li = copy.deepcopy([rfact*y for y in cells[x]['Li']])
            new_numori = copy.deepcopy(cells[x]['numori'])/2
            mother_initL = rfact*copy.deepcopy(cells[x]['final_Li'])
            mother_Ld = copy.deepcopy(cells[x]['Ld'])
            new_DLdLi = copy.deepcopy([[rfact*y[0],rfact*y[1]] for y in cells[x]['DLdLi']])

            new_dL = np.random.normal(params['dL_mu'], params['dL_sigma'])

            dict1 = {'Lb': new_Lb,'L': new_L, 'gen': str(x)+'B', 'tau': new_tau,'Lt': new_Lt, 'finish': False,
                     'born':t, 'DLi': new_DLi,'DLdLi': new_DLdLi,'Li':new_Li,'Ti':[], 'numori':new_numori,
                     'numori_born':copy.deepcopy(new_numori),'Ld':np.nan, 'name': name,'mLi': mother_initL, 'mLd':mother_Ld,
                    'rfact':rfact, 'dL': [0,new_dL]}

            cells[x+'B'] = copy.deepcopy(dict1)


            #keep oldest timer as final timer and give daughter remaining ones. Caclulate initiation time based on cell birth.
            TL_S_val = copy.deepcopy(cells[x]['Ti'].pop(0))
            cells[x+'B']['Ti'] = copy.deepcopy(cells[x]['Ti'])
            cells[x]['Ti'] = TL_S_val-copy.deepcopy(cells[x]['born'])

    for x in cells:
        if len(cells[x]['Li'])>0:
            cells[x]['Li'] = np.nan

    return cells


def simul_growth_ho_amir(nbstart, run_time, params, name):
    """Simulate the Ho and Amir model (Front. in Microbiol. 2015) with inter-initiation per origin adder and
    timer from initiation to division

    Parameters
    ----------
    nbstart : int
        number of cells to simulate
    run_time: int
        number of iterations
    params: dict
        experimental parameters
    name: str
        name of runs

    Returns
    -------
    cells : list of dict
        Each element of the list is a cell cycle defined by a
        dictionary of features (Lb, Ld etc.)
    """

    #initialize birth length and growth rate
    L0 = np.exp(np.random.normal(params['Lb_logn_mu'],params['Lb_logn_sigma'],size=nbstart))
    tau = np.exp(np.random.normal(params['tau_logn_mu'], params['tau_logn_sigma'], size=nbstart))

    #standard value of growth rate. Used to scale the noise appropriately
    normval = np.exp(params['tau_logn_mu'])

    #initialize the inter-initiation adder (exact procedure doesn't really matter here)
    #as all cells start with n_ori = 1, there's no initiation to division adder running
    DLi = np.random.normal(params['DLi_mu'], params['DLi_sigma'], size=nbstart)

    #time from initiation to division
    tid_mu = 90
    tid_var = 5
    Tid = np.random.normal(tid_mu, tid_var, size=nbstart)

    #initialize cell infos as a list of dictionaries. All cells start with n_ori = 1
    cells = {}
    for x in range(nbstart):
        dict1 = {'Lb': L0[x],'L':L0[x], 'gen': str(x), 'tau':tau[x], 'Lt': [[0,L0[x],1]], 'finish': False,
                'born':0, 'DLi': [[0,DLi[x]]],'DLdLi': [],'Li':[],'Ti':[],
                'numori':1,'Ld':np.nan, 'numori_born':1,'name': name,'mLi':np.nan,
                 'mLd':np.nan, 'rfact':0.5, 'Tid': [[0,Tid[x]]]}
        cells[str(x)] = dict1


    for t in range(run_time):

        divide_cell = []

        for x in cells:
            if cells[x]['finish']==False:


                #update cell size
                cells[x]['L'] = cells[x]['L']*(2**(1/cells[x]['tau']))
                cells[x]['Lt'].append([t,cells[x]['L'],cells[x]['numori']])

                #increment the most recent inter-initiation adder
                cells[x]['DLi'][-1][0] = cells[x]['DLi'][-1][0]+(cells[x]['Lt'][-1][1]-cells[x]['Lt'][-2][1])

                #if at least one volume counter since RI is running, increment all of them
                if len(cells[x]['DLdLi'])>0:
                    cells[x]['DLdLi'] = [[k[0]+(cells[x]['Lt'][-1][1]-cells[x]['Lt'][-2][1]),k[1]] for k in cells[x]['DLdLi']]
                    cells[x]['Tid'] = [[k[0]+1,k[1]] for k in cells[x]['Tid']]

                #if a volume counter has reached its limit divide
                if len(cells[x]['DLdLi'])>0:
                    if (cells[x]['numori']>1) and (cells[x]['Tid'][0][0]>cells[x]['Tid'][0][1]):
                        cells[x]['finish'] = True#tag cell as finished
                        cells[x]['Ld'] = cells[x]['L']
                        cells[x]['Td'] = len(cells[x]['Lt'])
                        cells[x]['Td_abs'] = t
                        cells[x]['d_Ld_Lb'] = cells[x]['L']-cells[x]['Lb']

                        #assign the correct adders (the oldest ones) to the cell that just divided
                        cells[x]['final_DLdLi'] = cells[x]['DLdLi'][0][0]
                        cells[x]['final_DLi'] = cells[x]['DLi'][0][1]
                        cells[x]['final_Li'] = cells[x]['Li'][0]
                        cells[x]['final_Tid'] = cells[x]['Tid'][0][1]

                        #for each accumulated variable suppress the oldest one
                        if len(cells[x]['DLdLi'])==1:
                            cells[x]['DLdLi'] = []
                        else:
                            cells[x]['DLdLi'].pop(0)

                        if len(cells[x]['Tid'])==1:
                            cells[x]['Tid'] = []
                        else:
                            cells[x]['Tid'].pop(0)

                        if len(cells[x]['DLi'])==1:
                            cells[x]['DLi'] = []
                        else:
                            cells[x]['DLi'].pop(0)

                        if len(cells[x]['Li'])==1:
                            cells[x]['Li'] = []
                        else:
                            cells[x]['Li'].pop(0)
                        divide_cell.append(x)

                #if the added volume has reached its limit make new RI
                if cells[x]['DLi'][-1][0]>cells[x]['DLi'][-1][1]:

                    #duplicate origin
                    cells[x]['numori'] = cells[x]['numori']*2

                    #Version where adder is noisy itself
                    newdli = cells[x]['numori']*np.random.normal(params['DLi_mu'], params['DLi_sigma'])

                    cells[x]['DLi'].append([0,newdli])

                    cells[x]['Li'].append(cells[x]['L'])

                    #temporarilly store TL_S as absolute time
                    cells[x]['Ti'].append(t)

                    #Version where adder itself is noisy
                    new_dv = cells[x]['numori']*np.exp(np.random.normal(params['DLdLi_logn_mu'], params['DLdLi_logn_sigma']))

                    cells[x]['DLdLi'].append([0,new_dv])

                    cells[x]['Tid'].append([0,np.random.normal(tid_mu, tid_var, size=1)])

        for x in divide_cell:

            #Draw division ratio
            rfact = 1/(1+np.random.normal(1,params['div_ratio']))

            #Create new cell using mother information
            new_tau = np.exp(correlated_normal(np.log(cells[x]['tau']), params['tau_logn_mu'], params['tau_logn_sigma'], params['tau_corr']))
            new_Lb = copy.deepcopy(rfact*cells[x]['L'])
            new_L = copy.deepcopy(rfact*cells[x]['L'])
            new_Lt = [[t,copy.deepcopy(rfact*cells[x]['L']),copy.deepcopy(cells[x]['numori'])/2]]
            new_DLi = copy.deepcopy([[rfact*y[0],rfact*y[1]] for y in cells[x]['DLi']])
            new_DLdLi = copy.deepcopy([[rfact*y[0],rfact*y[1]] for y in cells[x]['DLdLi']])
            new_Tid = copy.deepcopy(cells[x]['Tid'])
            new_Li = copy.deepcopy([rfact*y for y in cells[x]['Li']])
            new_numori = copy.deepcopy(cells[x]['numori'])/2
            mother_initL = copy.deepcopy(cells[x]['final_Li'])/2
            mother_Ld = copy.deepcopy(cells[x]['Ld'])

            dict1 = {'Lb': new_Lb,'L': new_L, 'gen': str(x)+'B', 'tau': new_tau,'Lt': new_Lt, 'finish': False,
                     'born':t, 'DLi': new_DLi,'DLdLi': new_DLdLi,'Tid': new_Tid, 'Li':new_Li,'Ti':[], 'numori':new_numori,
                     'numori_born':copy.deepcopy(new_numori),'Ld':np.nan, 'name': name,'mLi': mother_initL, 'mLd':mother_Ld,
                    'rfact':rfact}

            cells[x+'B'] = copy.deepcopy(dict1)


            #keep oldest timer as final timer and give daughter remaining ones. Caclulate initiation time based on cell birth.
            TL_S_val = copy.deepcopy(cells[x]['Ti'].pop(0))
            cells[x+'B']['Ti'] = copy.deepcopy(cells[x]['Ti'])
            cells[x]['Ti'] = TL_S_val-copy.deepcopy(cells[x]['born'])

    for x in cells:
        if len(cells[x]['Li'])>0:
            cells[x]['Li'] = np.nan

    return cells


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



def standardise_dataframe(simul):
    """Turns simulation output in structure similar to experiments

    Parameters
    ----------
    simul : list of dicts
        output of simulation function


    Returns
    -------
    simul_pd_exp : Pandas dataframe
        dataframe with same structure as experimental data
    """

    #transform list into dataframe
    simul_pd_or = pd.DataFrame(simul).T
    simul_pd = copy.deepcopy(simul_pd_or)

    #remove bad formatting
    simul_pd = simul_pd.apply(pd.to_numeric, errors='coerce')

    #add column with cell length over time
    simul_pd['length'] = simul_pd_or.Lt.apply(lambda x: np.array(x)[:,1])

    #change the genealogy-based index into a numerical index and create a mother_id column
    #similar to the one of the experimental data
    simul_pd['genealogy'] = simul_pd.index
    simul_pd.index = range(len(simul_pd.index))
    simul_pd['mother_id'] = simul_pd.apply(lambda row:
            int(simul_pd.index[simul_pd.genealogy == row.genealogy[0:-1]][0])
            if len(row.genealogy[0:-1])>0 else -1,axis = 1)
    simul_pd = simul_pd.astype({"mother_id": int})

    #rename fields to match experimental formatting
    simul_pd_exp = copy.deepcopy(simul_pd)
    if 'final_DLdLi' not in simul_pd_exp.keys():
        simul_pd_exp['final_DLdLi'] = -1.0
    simul_pd_exp = simul_pd_exp[['rfact','born','Lb','Ld','final_Li','tau','final_DLi','final_DLdLi','Td','Ti','mLi','mLd','numori_born','mother_id','length']]
    simul_pd_exp = simul_pd_exp.rename(columns = {'Lb':'Lb_fit','Ld':'Ld_fit','final_Li':'Li_fit',
        'tau':'tau_fit','final_DLi': 'DLi','mLi': 'mLi_fit','mLd': 'mLd_fit'})

    simul_pd_exp = simul_pd_exp[['rfact','born','Lb_fit','Ld_fit','Li_fit','tau_fit','DLi','Td','Ti','mLi_fit','mLd_fit','final_DLdLi','numori_born','mother_id','length']]

    return simul_pd_exp
