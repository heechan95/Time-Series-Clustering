from lxml.etree import fromstring
import pandas as pd
import numpy as np
import calendar, requests
from indices.indices import *
from indices.stocks import *
from datetime import datetime
from dateutil.relativedelta import relativedelta
import csv
import os, glob
import scipy.io

DATE_FORMAT='%b %d, %Y'

# set https header parameters
headers = {
	'User-Agent': 'Mozilla/5.0', #required 
	'referer': "https://www.investing.com",
	'host' : 'www.investing.com',
	'X-Requested-With' : 'XMLHttpRequest'
}

# now=datetime.now()-relativedelta(days=3)
now=datetime.now()
enddate=now.strftime('%m/%d/%Y')
a_year_ago=now-relativedelta(years=1)
startdate=a_year_ago.strftime('%m/%d/%Y')

class IndiceHistoricalData():	
	
	def __init__(self, API_url):
		self.API_url = API_url

	#set https header for request
	def setHeaders(self, headers):
		self.headers = headers 
	
	#set indice data (indices.py) 
	def setFormData(self, data):
		self.data = data 

	#prices frequency, possible values: Monthly, Weekly, Daily		
	def updateFrequency(self, frequency):
		self.data['frequency'] = frequency	 

	#desired time period from/to
	def updateStartingEndingDate(self, startingDate, endingDate):
		self.data['st_date'] = startingDate	 
		self.data['end_date'] = endingDate	 

	#possible values: 'DESC', 'ASC'
	def setSortOreder(self, sorting_order):
		self.data['sort_ord'] = sorting_order	 
    
	def downloadData(self):
		self.response = requests.post(self.API_url, data=self.data, headers=self.headers).content
		#parse tables with pandas - [0] probably there is only one html table in response
		self.observations = pd.read_html(self.response)[0]
		return self.observations

	#print retrieved data
	def printData(self):
		print(self.observations)

	#print retrieved data
	def saveDataCSV(self):
		self.observations.to_csv(self.data['realname']+'.csv', sep=';', encoding='utf-8')

def date2real(date_str, format=DATE_FORMAT):
    """
    :param date_str:
    :param format:
    :return:
    """
    t = datetime.strptime(date_str, format)
    return t.timestamp()
def csv2mat(csvfile):
    csvdata = pd.read_csv(csvfile, delimiter=";", engine='python', encoding='utf-8')
    csvdata.fillna(method='bfill', inplace=True)
    csvdata=csvdata.sort_index(axis=0, ascending=False).reset_index()
    newdata = {}
    newdata['X'] = np.expand_dims(np.asarray(csvdata['Date'].apply(date2real)), axis=1) 
    newdata['y'] = np.expand_dims(np.asarray(csvdata['Price']).transpose(), axis=1)
    today_str = csvdata['Date'][len(csvdata['Date'])-1]
    nxt_week_str = datetime.strftime(datetime.strptime(today_str, DATE_FORMAT) + relativedelta(weeks=4), 
                                    DATE_FORMAT)
    newdata['Xtest'] = np.expand_dims(np.asarray([date2real(nxt_week_str)]), axis=1)
    newdata['ytest'] = np.expand_dims(np.asarray([0.]), axis=1)
    scipy.io.savemat(csvfile.rsplit('.',1)[0]+'.mat', newdata)    
    
stocks={ 
   
'samsungelectronics':SamsungElectronics,
'skhynix':Skhynix,
'celltrion':Celltrion,
'hyundaimotor':Hyundaimotor,
'lgchem':Lgchem,
'samsungbiologics':Samsungbiologics,
'sktelecom':Sktelecom,
'posco':Posco,
'koreaelectric':Koreaelectric,
'kbfinancial':Kbfinancial
}
        
if __name__ == "__main__":

    datapath = './data/{:04d}/{:02d}/{:02d}'.format(now.year, now.month, now.day)
    resultpath='./results/{:04d}/{:02d}/{:02d}'.format(now.year, now.month, now.day)
    for k, v in stocks.items():
        try:
            #first set Headers and FormData	
            ihd = IndiceHistoricalData('https://www.investing.com/instruments/HistoricalDataAjax')
            ihd.setHeaders(headers)
            ihd.setFormData(v)

            #second set Variables
            ihd.updateFrequency('Monthly')
            ihd.updateStartingEndingDate(startdate, enddate)
            ihd.setSortOreder('ASC')
            ihd.downloadData()
            ihd.printData()
            ihd.saveDataCSV()

            if not os.path.exists(datapath+'/{}'.format(k)): 
                os.system('mkdir -p {}/{}'.format(datapath,k))
            if not os.path.exists(resultpath+'/{}'.format(k)): 
                os.system('mkdir -p {}/{}'.format(resultpath,k))
            cmd1 = 'cp {1}.csv {2}/{1}/{1}.csv'.format(v,k, resultpath)
            print('Data copied to {1}/{0}/{0}.csv'.format(k, resultpath))
            cmd = 'mv {1}.csv {2}/{1}/{1}.csv'.format(v,k, datapath)
            print('Data downloaded to {1}/{0}/{0}.csv'.format(k, datapath))
            os.system(cmd1)
            os.system(cmd)
            #csv2mat('{1}/{0}/{0}.csv'.format(k,datapath))
            #print('{1}/{0}/{0}.csv converted to {1}/{0}/{0}.mat'.format(k, datapath))
        except Exception as e:
            print(e)

