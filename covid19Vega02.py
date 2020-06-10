#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of COVID19 spread from public data
released by covid19.org for Indian data and 
John Hopkins University data for global data

@author: Abhijit Bhattacharyya
         Nuclear Physics Division
         Bhabha Atomic Research Centre
         Mumbai 400 085
         EMAIL:abhihere@gmail.com, vega@barc.gov.in
@File: covid19VEGA02.py
@author: Abhijit Bhattacharyya
Created on Tue May 26 23:29:22 2020
"""

import sys
import os 
import gc
from tabulate import tabulate
import argparse
import pandas as pd
from pandas.plotting import table
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from matplotlib import ticker
import pycountry_convert as pc
import folium
import branca
from datetime import datetime, timedelta, date
from scipy.interpolate import make_interp_spline, BSpline
import weasyprint as wsp
import PIL as pil
import docx


import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "xgridoff"

import glob, json, requests
import calmap
from bs4 import BeautifulSoup

from keras.layers import Input, Dense, Activation, LeakyReLU
from keras import models
from keras.optimizers import RMSprop, Adam

import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------
# -------------------  FUNCTION definitions ----------------------------
out = ""#+"output/"
cmdParser = argparse.ArgumentParser() 

#------------------------------- Get World Data over Net starts 
def getDataOverNet():
    print("\n You have chosen to download the latest Data from the internet....\n")
    req = requests.get('https://www.worldometers.info/coronavirus/')
    soup = BeautifulSoup(req.text, "lxml")
    #is_captcha = soup.find('div', id='recaptcha') is not None
    # print(is_captcha)

    #print(soup)
    """
    import pycurl
    from io import BytesIO
    
    b_obj = BytesIO()
    crl = pycurl.Curl()
    crl.setopt(pycurl.USERAGENT, 'Mozilla/5.0 (Windows; U; Windows NT 6.1; it; rv:1.9.2.3) Gecko/20100401 Firefox/3.6.3 (.NET CLR 3.5.30729)')
    crl.setopt(crl.URL, 'https://www.worldometers.info/coronavirus/')
    crl.setopt(crl.WRITEDATA, b_obj)
    crl.perform()
    # crl.close() 
    get_body = b_obj.getvalue().decode('utf8')
    #print(get_body)
    crl.close()
    df_country = get_body.find('div',attrs={"id" : "nav-tabContent"}).find('table',attrs={"id" : "main_table_countries_today"}).find_all('tr')
    print(df_country.head(5))
    exit(0)
    """
    df_country = soup.find('div',attrs={"id" : "nav-tabContent"}).find(
        'table',attrs={"id" : "main_table_countries_today"}).find_all('tr')
    arrCountry = []

    for i in range(8,len(df_country)-1):
        tmp = df_country[i].find_all('td')
        country=""
    
        if (tmp[0].string) is not  None:        
            if (tmp[0].string.find('<a') == -1):
                country = [tmp[0].string]
            else:
                country = [tmp[0].a.string] # Country
        else:
            continue
    

        for j in range(1,13):
            if (str(tmp[j].string) == 'None' or str(tmp[j].string) == ' '):
                country = country + [0]
            else:            
                country = country + [(tmp[j].string.replace(',','').replace('+',''))]   
    
        #print(country)   # diagnosis of retrieved lines population could not be found as it was relinked again and so ommitted
        country.pop(0)
        arrCountry.append(country)
        
        df_worldinfor = pd.DataFrame(arrCountry)
        df_worldinfor.columns = ['Country','Total Cases','New Cases','Total Deaths','New Deaths','Total Recovered','Active Cases',
                         'Serious Critical','Total Cases/1M pop','Deaths/1M pop','Total Test','Tests/1M pop']
                         #'Population','Continent']

    for i in range(0,len(df_worldinfor)):
        df_worldinfor['Country'].iloc[i] = df_worldinfor['Country'].iloc[i].strip()

    df_worldinfor = df_worldinfor.set_index('Country')

    # Saving the donwloaded data in CSV file for future use
    df_worldinfor.to_csv('DATASET/worldometers_info.csv')
    return df_worldinfor

#------------------------------- Get World Data over Net ends --------------------------------

# ---------------------------------   FUNCTION DEF ENDS HERE ---------------------------------


# -------------------------  Process comamnd line arguments ------------------
opt1 = opt2 = opt3 = 0
if (((len(sys.argv) > 1) and (sys.argv[1] == '-h')) or (len(sys.argv) == 1)):
    print("Command: python covid19Vega02.py -o <opt1>")
    print(" opt1: 0  - read stored data and continue processing....")
    print(" opt1: 1  - download data from net, store to disk and continue processing.")
    print(" opt1: 2  - download data from net and EXIT without processing.") 
    print("\n")

    if (len(sys.argv) == 1):
        print(" ........ please run again with options.")
    sys.exit(0)
    
if (len(sys.argv) > 1):
    cmdParser.add_argument('-o', type = str)
    cmdArguments = cmdParser.parse_args()
    optList = cmdArguments.o.split(',')

if (int(optList[0]) >= 2):
    getDataOverNet()
    exit(0)
elif (int(optList[0]) == 1):   # means opt1 = 1 i.e. get data from net and store
    df_worldinfor = getDataOverNet()
elif (int(optList[0]) <= 0):
    root_path_world_data = './DATASET'
    worldometerData = f'{root_path_world_data}/worldometers_info.csv'
    # df_worldinfor = pd.read_csv(worldometerData, index_col='Country') #None)    
    #df_worldinfor = pd.read_csv(worldometerData, index_col='None')    
    df_worldinfor = pd.read_csv(worldometerData)    
    for i in range(0,len(df_worldinfor)):
        df_worldinfor['Country'].iloc[i] = df_worldinfor['Country'].iloc[i].strip()
    
# ------------------------ Command line processing ends here---------------------------------------


# --------------------------------------- Main code starts here ---------------------------------------------
df_worldinfor1=df_worldinfor.apply(pd.to_numeric,errors='ignore')
df_worldinfor1['Total Recovered']= pd.to_numeric(df_worldinfor1['Total Recovered'],errors='coerce')
df_worldinfor1 = df_worldinfor1[df_worldinfor1.Country != 'Total:']
df_worldinfor1.index = df_worldinfor1['Country']
df_worldinfor1 = df_worldinfor1.drop(['Country'], axis=1)

# Validating testing data around the world **************************************

df_testing1 = df_worldinfor1.drop(['Total Cases','New Cases','Total Deaths','New Deaths','Total Recovered','Active Cases',
                                'Serious Critical','Total Cases/1M pop','Deaths/1M pop'], axis = 1)

df_testingS1 = df_testing1.sort_values('Tests/1M pop', ascending=False)
#print(tabulate(df_testingS1, headers = df_testingS1.columns.tolist(), tablefmt='psql'))

fig = plt.figure(figsize=(22,18))
fig.add_subplot(111)
#plt.axes(axisbelow=True)
#plt.barh(df_testingS1.sort_values('Tests/1M pop')["Tests/1M pop"].index[-50:], 
#         df_testingS1.sort_values('Tests/1M pop')["Tests/1M pop"].values[-50:], color="crimson")
plt.bar(df_testingS1.sort_values('Tests/1M pop')["Tests/1M pop"].index[-50:], 
         df_testingS1.sort_values('Tests/1M pop')["Tests/1M pop"].values[-50:], color="crimson")
plt.tick_params(size=5, labelsize=13)
plt.xticks(rotation=70)
plt.ylabel("Test/1M pop", fontsize=18)
plt.title("Top Countries (Tests / 1M pop )", fontsize=20)
plt.grid(alpha = 0.2)
plt.tight_layout()
plt.savefig(out+'PLOT/code2/testsPer1M.png')
# plt.show()
plt.close()

# Computing mortality Rate on Testing *******************************************

df_testing2 = df_worldinfor1.drop(
    ['New Cases','New Deaths','Total Recovered','Active Cases','Serious Critical','Deaths/1M pop','Total Cases/1M pop'], axis = 1)

df_testing2["MortalityRate"] = np.round(100 * df_testing2["Total Deaths"] / df_testing2["Total Cases"], 2)
df_testing2["Positive"] = np.round(100 * df_testing2["Total Cases"] / df_testing2["Total Test"], 2)

#df_testing2 = df_testing2.sort_values('Total Cases', ascending=False)
df_testing2 = df_testing2.sort_values('Country', ascending=True)

df_testing2.style.background_gradient(cmap='Blues',subset=["Total Test"])\
                        .background_gradient(cmap='Reds',subset=["Tests/1M pop"])\
                        .background_gradient(cmap='Greens',subset=["Total Cases"])\
                        .background_gradient(cmap='Purples',subset=["Total Deaths"])\
                        .background_gradient(cmap='YlOrBr',subset=["MortalityRate"])\
                        .background_gradient(cmap='bone_r',subset=["Positive"])

#print(tabulate(df_testing2, headers = df_testing2.columns.tolist(), tablefmt='psql'))

docFName = 'PLOT/code2/mortalityTable.docx'
nCol = len(df_testing2.columns.tolist())
# myDoc = docx.Document(docFName)   # to read old file
myDoc = docx.Document()             # to create new file

df_testing2.reset_index(drop=False, inplace=True)

myDocHead = myDoc.add_heading('Country-wise tests and mortality report \n',1)
#myDoc.add_heading('  ',2)
myDocHead.bold = True
myDocHead.underline = True

tt = myDoc.add_table(df_testing2.shape[0] + 1, df_testing2.shape[1])   # row, col
tt.style = 'LightShading-Accent1'

for j in range(df_testing2.shape[-1]):
    tt.cell(0, j).text = df_testing2.columns[j]      # header

for i in range(df_testing2.shape[0]):
    for j in range(df_testing2.shape[-1]):
        tt.cell(i+1, j).text = str(df_testing2.values[i, j])  # rest part of the dataframe
        
myDoc.save(docFName)
