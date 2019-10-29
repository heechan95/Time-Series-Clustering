"""
Create srkl dataset given csv datasets
There are several options for clustering methods, including one with ARIMA, with DTW, and with KS TEST
you can specify which method to use as follows:

python3 create_srkl_data.py --cluster ARIMA/dtw/ks/random

The result .mat file will be saved in ./srkl-data folder. if you want to use it, move it to ../srkl-data folder
@author : Heechan Lee(lhc101020@unist.ac.kr, SAIL lab)
"""


import pandas as pd
import numpy as np
import scipy.io
import argparse
import os, glob
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta


from clustering_utils_v2 import get_result_clustering
from clustering_utils_v2 import random_clustering
from clustering_utils_v2 import dtw_clustering
from clustering_utils_v2 import ks_clustering

parser = argparse.ArgumentParser()
parser.add_argument('--cluster', type=str, 
                    help='clustering methodi ARIMA or random', 
                    default='random')


args = parser.parse_args()
# python3 naver_finance.py


code_df = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13', header=0)[0]

code_df = code_df[['회사명','종목코드']]

code_df = code_df.rename(columns={'회사명':'name', '종목코드':'code'})

code_df.code=code_df.code.map('{:06d}'.format)
#print(code_df.head())
def get_url(item_name, code_df): 
    code = code_df.query("name=='{}'".format(item_name))['code'].to_string(index=False)
    url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code)
    print("Request URL = {}".format(url))
    return url

def get_df(item_name,pages):
    url = get_url(item_name, code_df)
    df = pd.DataFrame()
    for page in range(1,pages+1):
        pg_url = '{url}&page={page}'.format(url=url, page=page)
        df = df.append(pd.read_html(pg_url,header=0)[0], ignore_index=True)
    df.dropna(inplace=True)
    return df


def clean_df(df):
    df = df.rename(columns= {'날짜': 'date', '종가': 'close', '전일비': 'diff', '시가': 'open', '고가': 'high', '저가': 'low', '거래량': 'volume'})
    df[['close','diff','open','high','low','volume']] \
        = df[['close','diff','open','high','low','volume']].astype(int)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['date'], ascending=True)
    return df

stocks = {'삼성전자':'samsungelectronics','SK하이닉스':'skhynix' ,'현대자동차':'hyundaimotor' ,'LG화학':'lgchem' ,'현대모비스':'hyundaimobis' ,'삼성물산':'samsungcnt' ,'한국전력공사':'koreaelectric' ,'삼성에스디에스':'samsungsds','아모레퍼시픽':'amorepacific' ,'LG생활건강':'lghousehold'}

DATE_FORMAT='%Y-%m-%d'

def date2real(date_str, format='%b %d, %Y'):
    """
    :param date_str:
    :param format:
    :return:
    """
    year = datetime(2019,12,31)-datetime(2018,12,31)
    t = datetime.strptime(date_str, format)
    delta = t-datetime(t.year,1,1)
    return (delta/year + t.year)
    #t = datetime.strptime(date_str, format)
    #return t.timestamp()

def csv2mat(rdir=None):
    
    if rdir is not None:
        if args.cluster == 'ARIMA':
            corp_list_list, tr_list_list = get_result_clustering(rdir)
        elif args.cluster == 'random':
            corp_list_list, tr_list_list = random_clustering(rdir,3)
        elif args.cluster == 'dtw':
            corp_list_list, tr_list_list = dtw_clustering(rdir)
        elif args.cluster == 'ks':
            corp_list_list, tr_list_list = ks_clustering(rdir)
        else:
            raise Exception("Not Supported")
    else:
        if args.cluster == 'ARIMA':
            corp_list_list, tr_list_list = get_result_clustering()
        elif args.cluster == 'random':
            corp_list_list, tr_list_list = random_clustering(num_elements=3)
        elif args.cluster == 'dtw':
            corp_list_list, tr_list_list = dtw_clustering()
        elif args.cluster == 'ks':
            corp_list_list, tr_list_list = ks_clustering()
        else:
            raise Exception("Not Supported")

    for i, csvfiles_training in enumerate(tr_list_list):
        srkl_data = {'S':[],'X':[], 'y':[] }
        flag = True
        for j,csvfile_training in enumerate(csvfiles_training):
            csvdata_training = pd.read_csv(csvfile_training, delimiter=";", engine='python', encoding='utf-8')
            csvdata_training.fillna(method='bfill', inplace=True)
            csvdata_training=csvdata_training.sort_index(axis=0, ascending=False).reset_index()
            csvdata_training=csvdata_training[csvdata_training['Vol.'] != '-'].reset_index()
            newdata = {}
            if flag:
                newdata['X'] = np.expand_dims(np.asarray(csvdata_training['Date'].apply(date2real)), axis=1)
            newdata['y'] = np.expand_dims(np.asarray(csvdata_training['Price']).transpose(), axis=1)

            if flag:
                srkl_data['X'].append(newdata['X'])
            srkl_data['y'].append(newdata['y'])

            flag=False
            srkl_data['S'].append(corp_list_list[i][j])

        
        for k,v in srkl_data.items():
            if k == 'S': continue
            srkl_data[k] = np.hstack(v)

        if not os.path.exists('./srkl-data/{}'.format(datetime.now().strftime(DATE_FORMAT))):
            os.system('mkdir -p ./srkl-data/{}'.format(datetime.now().strftime(DATE_FORMAT)))
        scipy.io.savemat('./srkl-data/{}/stocks-{}.mat'.format(datetime.now().strftime(DATE_FORMAT),i), srkl_data)


'''
today = datetime.now()
tr_enddate = today - relativedelta(weeks=2)
te_begindate = tr_enddate + relativedelta(days=1)
tr_begindate = today - relativedelta(years=1)

datapath = './data/{:04d}/{:02d}/{:02d}'.format(today.year,today.month,today.day)
csvfile_tr_list = []
csvfile_te_list = []
for k,v in stocks.items():
    print('Data downloaded to {0}/{1}/'.format(datapath,v))
    base_df = get_df(k,30)
    base_df = clean_df(base_df)
    df_tr = base_df[base_df['date']<tr_enddate.strftime(DATE_FORMAT)]
    df_tr = df_tr[df_tr['date']>tr_begindate.strftime(DATE_FORMAT)]
    df_te = base_df[base_df['date']>tr_enddate.strftime(DATE_FORMAT)]
    if not os.path.exists(datapath+'/{}'.format(v)):
        os.system('mkdir -p {}/{}'.format(datapath,v))
    df_tr.to_csv('{1}/{0}/{0}_tr.csv'.format(v,datapath))
    df_te.to_csv('{1}/{0}/{0}_te.csv'.format(v,datapath))

    csvfile_tr_list.append('{1}/{0}/{0}_tr.csv'.format(v,datapath))
    csvfile_te_list.append('{1}/{0}/{0}_te.csv'.format(v,datapath))
'''

#TODO be provided arguments about which datasets are made into srkl dataset
#change to your datapath
today = datetime.now()
datapath = './data/{:04d}/{:02d}/{:02d}'.format(today.year,today.month,today.day)
#datapath = './data/2019/07/31/'
print("Create srkl_data.mat")
csv2mat(datapath)
print("Done")


#df = get_df(stock,10)
#df = clean_df(df)
#print(df.head()) 
#reference: https://excelsior-cjh.tistory.com/109 [EXCELSIOR]

