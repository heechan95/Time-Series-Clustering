"""
Clustering util version 2
1) ARIMA -- First, run ARIMA on each time series and extracts coeffiecients from the ARIMA results. Secondly, cluster time series using K-means.

2) RANDOM -- randomly cluster with each cluster of 3 time series.

3) dtw -- construct distance matrix using dtw and clustering data with AgglomerativeClustering.

4) ks-test -- construct distance matrix using 2 sample ks-test and clustering data with AgglomerativeClustering.

5) bayesian -- using MCMC(Hamiltonian MC) to approximate correlation matrix with Tensorflow probability library. *in progress

@author : Heechan Lee(lhc101020@unist.ac.kr)
"""


import numpy as np
import pandas as pd
import os
import warnings
from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

ROOT_DIR="./data/2019/07/31/"
 
# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.9)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error
 
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values, verbose=False):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    if verbose:
                        print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    return best_cfg, best_score

# Get best model oceffiecient
# TODO: p,d,q value is fixed, will be able to be flexble.
def get_best_model_coefs(df_list):
    print(len(df_list))
    p_values = [1, 2, 3, 4, 5]
    d_values = range(0, 2)
    q_values = range(0, 2)
    coef_list=list()
    for df in df_list:
        series=df['Price']
        best_cfg, best_score = evaluate_models(series.values, p_values, d_values, q_values)
        model = ARIMA(series, order=best_cfg)
        model_fit = model.fit(disp=0)
        coef_list.append(model_fit.params)
        
    return coef_list

def get_num_of_terms(coef_list):
    import re
    pattern_ar = 'ar*'
    arterm = re.compile(pattern_ar)
    pattern_ma = 'ma*'
    materm = re.compile(pattern_ma)

    armaterms = []
    for coefs in coef_list:
        armaterm = {'ar':0, 'ma':0}
        for term in coefs.index:
            if arterm.match(term) is not None:
                armaterm['ar'] += 1
            elif materm.match(term) is not None:
                armaterm['ma'] += 1

        armaterms.append(armaterm)
            
    return armaterms


#argument: coef_list, ar_ma_terms_list -- the result list from 'get_num_of_terms' function
#map each time series to data whose dimension is sum of maximum number of AR terms and MA terms
#fill in coefficient if a time series has it, otherwise, fill in zero.

def make_coef_collection(coef_list,ar_ma_terms_list):
    def calc_max_arma(terms_list):
        ar_max = 0
        ma_max = 0
        for terms in terms_list:
            if ar_max < terms['ar']:
                ar_max = terms['ar']
            if ma_max < terms['ma']:
                ma_max = terms['ma']
        return ar_max, ma_max
    ar_max, ma_max = calc_max_arma(ar_ma_terms_list)
    data = np.zeros((len(coef_list),(ar_max+ma_max)))
    for i, coefs in enumerate(coef_list):
        ar_n = ar_ma_terms_list[i]['ar']
        ma_n = ar_ma_terms_list[i]['ma']
        j=0

        for ar_j in range(0,ar_n):
            data[i,ar_j] = coef_list[i][j]
            j += 1

        for ma_j in range(0,ma_n):
            data[i,ar_max+ma_j] = coef_list[i][j]
            j += 1
                
                
    return data

def get_result_clustering(rdir=ROOT_DIR):
    
    ROOT_DIR = rdir
    filenames = []
    #filenames_te = []
    corporate_names = []
    for r,d,f in os.walk(ROOT_DIR):
        if d is not None:
            for company in d:
                corporate_names.append(company)
    
        for file in f:
            if file.endswith('.csv'):
                filenames.append(r+'/'+file)
    print(corporate_names) 
    df_list = [pd.read_csv(filename, delimiter=";", engine='python', encoding='utf-8') for filename in filenames]
    #p_values = [1, 2]
    #d_values = range(0, 1)
    #q_values = range(0, 1)
    warnings.filterwarnings("ignore")
    coef_list = get_best_model_coefs(df_list)
    armaterms = get_num_of_terms(coef_list)
    coef_list_ = [coefs.values[1:] for coefs in coef_list]
    a = make_coef_collection(coef_list_, armaterms)
    from sklearn.cluster import KMeans
    from sklearn.cluster import AgglomerativeClustering
    num_cluster=3
    #kmeans = KMeans(n_clusters=num_cluster, n_init=20, random_state=0).fit(a)
    kmeans = AgglomerativeClustering(num_cluster).fit(a)
    print(kmeans.labels_) 
    cluster_corp = []
    cluster_file_tr = []
    #cluster_file_te = []

    for i in range(num_cluster):
        cluster_corp.append(list(np.array(corporate_names)[kmeans.labels_==i]))
        cluster_file_tr.append(list(np.array(filenames)[kmeans.labels_==i]))
        #cluster_file_te.append(list(np.array(filenames_te)[kmeans.labels_==i]))

    return cluster_corp, cluster_file_tr #, cluster_file_te

def random_clustering(rdir=ROOT_DIR, num_elements=2):

    ROOT_DIR = rdir
    filenames = []
    names = []
    for r,d,f in os.walk(ROOT_DIR):
        if d is not None:
            for company in d:
                names.append(company)

        for file in f:
            if file.endswith('.csv'):
                filenames.append(r+'/'+file)
    #df_list = [pd.read_csv(filename) for filename in filenames_tr]
    #p_values = [1, 2]
    #d_values = range(0, 1)
    #q_values = range(0, 1)
    warnings.filterwarnings("ignore")
    zipped_list = list(zip(names, filenames))
    import random
    random.shuffle(zipped_list)
    names, filenames = zip(*zipped_list)
    
    cluster_corp = []
    cluster_file = []
    length = len(names)
    idx=0
    while idx+num_elements <= length:
        cluster_corp.append(list(names[idx:idx+num_elements]))
        cluster_file.append(list(filenames[idx:idx+num_elements]))
        idx += num_elements
    
    for i in range(idx,length):
        cluster_corp[-1].append(names[i])
        cluster_file[-1].append(filenames[i])
    #if idx+num_elements > length:
    #    cluster_corp[-1].append(corporate_names[-1])
    #    cluster_file_tr[-1].append(filenames_tr[-1])
    #    cluster_file_te[-1].append(filenames_te[-1])

    return cluster_corp, cluster_file


def dtw_clustering(rdir=ROOT_DIR):

    ROOT_DIR = rdir
    filenames = []
    names = []
    for r,d,f in os.walk(ROOT_DIR):
        if d is not None:
            for company in d:
                names.append(company)

        for file in f:
            if file.endswith('.csv'):
                filenames.append(r+'/'+file)

    df_list = [pd.read_csv(filename, delimiter=";", engine='python', encoding='utf-8').sort_index(axis=0, ascending=False).reset_index() for filename in filenames]
    df_list = [df[df['Vol.'] != '-'].reset_index() for df in df_list]
    df_list = [df['Price'] for df in df_list]
    df_list = [(df-df.min())/(df.max()-df.min()) for df in df_list]

    from tslearn.metrics import dtw
    sz = len(filenames)
    distance_matrix = np.zeros((sz,sz))
    for i in range(sz):
        for j in range(sz):
            distance_matrix[i,j] = dtw(df_list[i], df_list[j])

    from sklearn.cluster import AgglomerativeClustering
    num_cluster = 3
    clustering = AgglomerativeClustering(num_cluster, 'precomputed',linkage='complete')
    clustering.fit(distance_matrix)
    
    print(clustering.labels_)
    
    cluster_corp = []
    cluster_files = [] 
    for i in range(num_cluster):
        cluster_corp.append(list(np.array(names)[clustering.labels_==i]))
        cluster_files.append(list(np.array(filenames)[clustering.labels_==i]))

    return cluster_corp, cluster_files #, cluster_file_te


def ks_clustering(rdir=ROOT_DIR):

    ROOT_DIR = rdir
    filenames = []
    names = []
    for r,d,f in os.walk(ROOT_DIR):
        if d is not None:
            for company in d:
                names.append(company)

        for file in f:
            if file.endswith('.csv'):
                filenames.append(r+'/'+file)

    df_list = [pd.read_csv(filename, delimiter=";", engine='python', encoding='utf-8').sort_index(axis=0, ascending=False).reset_index() for filename in filenames]
    df_list = [df[df['Vol.'] != '-'].reset_index() for df in df_list]
    df_list = [df['Price'] for df in df_list]
    df_list = [(df-df.min())/(df.max()-df.min()) for df in df_list]

    from scipy import stats
    sz = len(filenames)
    distance_matrix = np.zeros((sz,sz))
    for i in range(sz):
        for j in range(sz):
            distance_matrix[i,j] = stats.ks_2samp(df_list[i].values, df_list[j].values)[0]

    from sklearn.cluster import AgglomerativeClustering
    num_cluster = 3
    clustering = AgglomerativeClustering(num_cluster, 'precomputed',linkage='complete')
    clustering.fit(distance_matrix)

    print(clustering.labels_)

    cluster_corp = []
    cluster_files = []
    for i in range(num_cluster):
        cluster_corp.append(list(np.array(names)[clustering.labels_==i]))
        cluster_files.append(list(np.array(filenames)[clustering.labels_==i]))

    return cluster_corp, cluster_files #, cluster_file_te

def bayesian_clustering(rdir=ROOT_DIR):
    ROOT_DIR = rdir
    filenames = []
    names = []
    for r,d,f in os.walk(ROOT_DIR):
        if d is not None:
            for company in d:
                names.append(company)

        for file in f:
            if file.endswith('.csv'):
                filenames.append(r+'/'+file)

    df_list = [pd.read_csv(filename, delimiter=";", engine='python', encoding='utf-8').sort_index(axis=0, ascending=False).reset_index() for filename in filenames]
    df_list = [df[df['Vol.'] != '-'].reset_index() for df in df_list]
    df_list = [df['Price'] for df in df_list]
    df_list = [(df-df.min())/(df.max()-df.min()) for df in df_list]
    df_list = [(df.diff() / df).dropna().reset_index(drop=True) for df in df_list]

    #TODO: import Tensorflow Probability, and MCMC


    from scipy import stats
    sz = len(filenames)
    distance_matrix = np.zeros((sz,sz))
    for i in range(sz):
        for j in range(sz):
            distance_matrix[i,j] = stats.ks_2samp(df_list[i].values, df_list[j].values)[0]

    from sklearn.cluster import AgglomerativeClustering
    num_cluster = 3
    clustering = AgglomerativeClustering(num_cluster, 'precomputed',linkage='complete')
    clustering.fit(distance_matrix)

    print(clustering.labels_)

    cluster_corp = []
    cluster_files = []
    for i in range(num_cluster):
        cluster_corp.append(list(np.array(names)[clustering.labels_==i]))
        cluster_files.append(list(np.array(filenames)[clustering.labels_==i]))

    return cluster_corp, cluster_files #, cluster_file_te

