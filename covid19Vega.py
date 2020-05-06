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
@File: covid19VEGA.py
@author: Abhijit Bhattacharyya
Created on Tue May  5 12:00:41 2020
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pycountry_convert as pc
import folium
import branca
from datetime import datetime, timedelta, date
from scipy.interpolate import make_interp_spline, BSpline
import plotly.express as px
import glob, json, requests
import calmap

from keras.layers import Input, Dense, Activation, LeakyReLU
from keras import models
from keras.optimizers import RMSprop, Adam

import warnings
warnings.filterwarnings('ignore')

# <------------------- FUNCTION DEFINITION ----------------------------->
def plot_params(ax,axis_label= None, plt_title = None,label_size=15, axis_fsize = 15, title_fsize = 20, scale = 'linear' ):
    # Tick-Parameters
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which='both', width=1,labelsize=label_size)
    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3, color='0.8')
    
    # Grid
    plt.grid(lw = 1, ls = '-', c = "0.7", which = 'major')
    plt.grid(lw = 1, ls = '-', c = "0.9", which = 'minor')

    # Plot Title
    plt.title( plt_title,{'fontsize':title_fsize})
    
    # Yaxis sacle
    plt.yscale(scale)
    plt.minorticks_on()
    # Plot Axes Labels
    xl = plt.xlabel(axis_label[0],fontsize = axis_fsize)
    yl = plt.ylabel(axis_label[1],fontsize = axis_fsize)
    
    
def visualize_covid_cases(confirmed, deaths, continent=None , country = None , state = None, period = None, figure = None, scale = "linear"):
    x = 0
    if figure == None:
        f = plt.figure(figsize=(10,10))
        # Sub plot
        ax = f.add_subplot(111)
    else :
        f = figure[0]
        # Sub plot
        ax = f.add_subplot(figure[1],figure[2],figure[3])
    ax.set_axisbelow(True)
    plt.tight_layout(pad=10, w_pad=5, h_pad=5)
    
    stats = [confirmed, deaths]
    label = ["Confirmed", "Deaths"]
    
    if continent != None:
        params = ["Continent",continent]
    elif country != None:
        params = ["Country",country]
    else: 
        params = ["All", "All"]
    color = ["darkcyan","crimson"]
    marker_style = dict(linewidth=3, linestyle='-', marker='o',markersize=4, markerfacecolor='#ffffff')
    for i,stat in enumerate(stats):
        if params[1] == "All" :
            cases = np.sum(np.asarray(stat.iloc[:,5:]),axis = 0)[x:]
        else :
            cases = np.sum(np.asarray(stat[stat[params[0]] == params[1]].iloc[:,5:]),axis = 0)[x:]
        date = np.arange(1,cases.shape[0]+1)[x:]
        plt.plot(date,cases,label = label[i]+" (Total : "+str(cases[-1])+")",color=color[i],**marker_style)
        plt.fill_between(date,cases,color=color[i],alpha=0.3)

    if params[1] == "All" :
        Total_confirmed = np.sum(np.asarray(stats[0].iloc[:,5:]),axis = 0)[x:]
        Total_deaths = np.sum(np.asarray(stats[1].iloc[:,5:]),axis = 0)[x:]
    else :
        Total_confirmed =  np.sum(np.asarray(stats[0][stat[params[0]] == params[1]].iloc[:,5:]),axis = 0)[x:]
        Total_deaths = np.sum(np.asarray(stats[1][stat[params[0]] == params[1]].iloc[:,5:]),axis = 0)[x:]
        
    text = "From "+stats[0].columns[5]+" to "+stats[0].columns[-1]+"\n"
    text += "Mortality rate : "+ str(int(Total_deaths[-1]/(Total_confirmed[-1])*10000)/100)+"\n"
    text += "Last 5 Days:\n"
    text += "Confirmed : " + str(Total_confirmed[-1] - Total_confirmed[-6])+"\n"
    text += "Deaths : " + str(Total_deaths[-1] - Total_deaths[-6])+"\n"
    text += "Last 24 Hours:\n"
    text += "Confirmed : " + str(Total_confirmed[-1] - Total_confirmed[-2])+"\n"
    text += "Deaths : " + str(Total_deaths[-1] - Total_deaths[-2])+"\n"
    
    plt.text(0.02, 0.78, text, fontsize=15, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes,bbox=dict(facecolor='white', alpha=0.4))
    
    # Plot Axes Labels
    # axis_label = ["Days ("+df_confirmed.columns[5]+" - "+df_confirmed.columns[-1]+")","No of Cases"]
    axis_label = ["Days ("+confirmed.columns[5]+" - "+confirmed.columns[-1]+")","No of Cases"]
    
    # Plot Parameters
    plot_params(ax,axis_label,scale = scale)
    
    # Plot Title
    if params[1] == "All" :
        plt.title("COVID-19 Cases World",{'fontsize':25})
    else:   
        plt.title("COVID-19 Cases for "+params[1] ,{'fontsize':25})
        
    # Legend Location
    l = plt.legend(loc= "best",fontsize = 15)
    
    if figure == None:
        plt.show()
        
def get_total_cases(cases, country = "All"):
    if(country == "All") :
        return np.sum(np.asarray(cases.iloc[:,5:]),axis = 0)[-1]
    else :
        return np.sum(np.asarray(cases[cases["Country"] == country].iloc[:,5:]),axis = 0)[-1]
    
def get_mortality_rate(confirmed,deaths, continent = None, country = None):
    if continent != None:
        params = ["Continent",continent]
    elif country != None:
        params = ["Country",country]
    else :
        params = ["All", "All"]
    
    if params[1] == "All" :
        Total_confirmed = np.sum(np.asarray(confirmed.iloc[:,5:]),axis = 0)
        Total_deaths = np.sum(np.asarray(deaths.iloc[:,5:]),axis = 0)
        mortality_rate = np.round((Total_deaths/(Total_confirmed+1.01))*100,2)
    else :
        Total_confirmed =  np.sum(np.asarray(confirmed[confirmed[params[0]] == params[1]].iloc[:,5:]),axis = 0)
        Total_deaths = np.sum(np.asarray(deaths[deaths[params[0]] == params[1]].iloc[:,5:]),axis = 0)
        mortality_rate = np.round((Total_deaths/(Total_confirmed+1.01))*100,2)
        
    return np.nan_to_num(mortality_rate)


def dd(date1,date2):
    return (datetime.strptime(date1,'%m/%d/%y') - datetime.strptime(date2,'%m/%d/%y')).days


out = ""#+"output/"

#<-------------------------------------------------------------------->


#Declare root paths for DATA
root_path_JHU_T = './DATASET/JohnHopkinsU_CSSE/csse_covid_19_data/csse_covid_19_time_series'
root_path_JHU_D = './DATASET/JohnHopkinsU_CSSE/csse_covid_19_data/csse_covid_19_daily_reports'
root_path_IND_C = './DATASET/APIcovid19indiaorg/CSV'
root_path_IND_J = './DATASET/APIcovid19indiaorg/JSON'

globalTSC = f'{root_path_JHU_T}/time_series_covid19_confirmed_global.csv'
globalTSD = f'{root_path_JHU_T}/time_series_covid19_deaths_global.csv'
globalTSR = f'{root_path_JHU_T}/time_series_covid19_recovered_global.csv'

indiaRaw1 = f'{root_path_IND_C}/raw_data1.csv'
indiaRaw2 = f'{root_path_IND_C}/raw_data2.csv'
indiaRaw3 = f'{root_path_IND_C}/raw_data3.csv'
indiaDR1 = f'{root_path_IND_C}/death_and_recovered1.csv'
indiaDR2 = f'{root_path_IND_C}/death_and_recovered2.csv'
indiaState = f'{root_path_IND_C}/state_wise.csv'
indiaDist = f'{root_path_IND_C}/district_wise.csv'

#Load data file
gTSCDF = pd.read_csv(globalTSC, parse_dates=True)
gTSDDF = pd.read_csv(globalTSD, parse_dates=True)
gTSRDF = pd.read_csv(globalTSR, parse_dates=True)
iCDF1 = pd.read_csv(indiaRaw1, index_col = 1, parse_dates=True)
iCDF2 = pd.read_csv(indiaRaw2, index_col = 1, parse_dates=True)
iCDF3 = pd.read_csv(indiaRaw3, index_col = 1, parse_dates=True)   #Raw3 is not matched with Raw1, Raw2

iDRDF1 = pd.read_csv(indiaDR1, index_col = 1, parse_dates=True)
iDRDF1.insert(iDRDF1.shape[1], "Unnamed: 15", '')
iDRDF2 = pd.read_csv(indiaDR2, index_col = 1, parse_dates=True)


# Preprocessing of data
gTSCDF = gTSCDF.replace(np.nan, '', regex=True)     # Global Confirmed cases
gTSCDF = gTSCDF.rename(columns={"Province/State" : "State", "Country/Region" : "Country"})
gTSDDF = gTSDDF.replace(np.nan, '', regex=True)     # Global Deceased Cases
gTSDDF = gTSDDF.rename(columns={"Province/State" : "State", "Country/Region" : "Country"})
gTSRDF = gTSRDF.replace(np.nan, '', regex=True)     # Global Recovered Cases
gTSRDF = gTSRDF.rename(columns={"Province/State" : "State", "Country/Region" : "Country"})

iCDF3.insert(2, "dummy", 0)  # this treatment is required as the data RAW3 is not organized at par with RAW1/2
iCDF31 = iCDF3[iCDF3.columns[[19, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 11, 16, 17, 18, 13, 14, 15, 9]]]
iCDF = (iCDF1.append(iCDF2)).append(iCDF31)
iCDF = iCDF.drop(['dummy', 'Num Cases'], axis=1)
iDRDF = iDRDF1.append(iDRDF2)
iCDF = iCDF.rename(columns={"Patient Number" : "PID", "Age Bracket" : "Age", "Detected City" : "City", "Detected District" : "District", "Detected State" : "State", "State code" : "STcode"})
iDRDF = iDRDF.rename(columns={"Patient Number" : "PID", "Age Bracket" : "Age", "Detected City" : "City", "Detected District" : "District", "Detected State" : "State", "State code" : "STcode"})
iDRDF = iDRDF.drop('Unnamed: 15', axis=1)
iRDF = iDRDF.query('Patient_Status == "Recovered"')  # recovered dataset
iDDF = iDRDF.query('Patient_Status == "Deceased"')   # Deceased dataset


# start analysis with JHU data 

#----------------------------------------------------------------------------------------------------------------------------
# Spread Trends in few affected Countries
gCDF_Countr = gTSCDF.groupby(["Country"]).sum()
gCDF_Countr = gCDF_Countr.sort_values(gCDF_Countr.columns[-1], ascending = False)
lCountries = gCDF_Countr[gCDF_Countr[gCDF_Countr.columns[-1]] >= 4000].index
cols = 2
rows = int(np.ceil(lCountries.shape[0]/cols))
myFig = plt.figure(figsize = (18, 7 * rows))
for i, iCountry in enumerate(lCountries):
    visualize_covid_cases(gTSCDF, gTSDDF, country = iCountry, figure = [myFig, rows, cols, i + 1])

plt.savefig(out+'PLOT/Global_Report.png')    
plt.show()


#------------------------------------------------------------------------------------------------------------------------
# Trend comparison for Confirmed Cases
gTSTMPC = gTSCDF.groupby('Country').sum().drop(["Lat", "Long"], axis=1).sort_values(gTSCDF.columns[-1], ascending=False)

threshold = 50
nLast = 0
dLast = ''
myCountry = ''
f = plt.figure(figsize=(10,12))
ax = f.add_subplot(111)
for i, Country in enumerate(gTSTMPC.index):
    if i >= 9:
        if Country != "India" and Country != "Japan":
            continue
    days = 60
    gTST1 = gTSTMPC.loc[gTSTMPC.index == Country].values[0]
    gTST1 = gTST1[gTST1 > threshold][:days]
    date = np.arange(0, len(gTST1[:days]))
    if Country == "India":
        nLast = gTST1[-1]
        dLast = date.max()
        myCountry = Country
    xnew = np.linspace(date.min(), date.max(), 30)
    spl = make_interp_spline(date, gTST1, k = 1)
    power_smooth = spl(xnew)
    if Country != "India":
        plt.plot(xnew, power_smooth, '-o', label = Country, linewidth = 3, markevery = [-1])
    else:
        marker_style = dict(linewidth = 4, linestyle = '-', marker = 'o', markersize = 10, markerfacecolor = '#ffffff')
        plt.plot(date, gTST1, "-.", label = Country, **marker_style)

print("Date: ", dLast, "Country: ", myCountry, "  Num: ", nLast, "\n");
plt.tick_params(labelsize = 14)        
plt.xticks(np.arange(0, days, 7), [ "Day " + str(i) for i in range(days)][::7])     

# Reference lines 
x = np.arange(0, 18)
y = 2**(x + np.log2(threshold))
plt.plot(x, y, "--", linewidth =2, color = "gray")
plt.annotate("No. of cases doubles every day", (x[-2], y[-1]), xycoords = "data", fontsize = 14, alpha = 0.5)

x = np.arange(0, int(days - 12))
y = 2**(x / 2 + np.log2(threshold))
plt.plot(x, y, "--", linewidth = 2, color = "gray")
plt.annotate(".. every second day", (x[-3], y[-1]), xycoords = "data", fontsize = 14, alpha = 0.5)

x = np.arange(0, int(days - 4))
y = 2**(x / 7 + np.log2(threshold))
plt.plot(x, y, "--", linewidth = 2, color = "gray")
plt.annotate(".. every week", (x[-3], y[-1]), xycoords = "data", fontsize = 14, alpha = 0.5)

x = np.arange(0, int(days - 4))
y = 2**(x / 30 + np.log2(threshold))
plt.plot(x, y, "--", linewidth = 2, color = "gray")
plt.annotate(".. every month", (x[-3], y[-1]), xycoords = "data", fontsize = 14, alpha = 0.5)
 
x = np.arange(0, int(days - 5))
y = 2**(x / 4 + np.log2(threshold))
plt.plot(x, y, "--", linewidth = 2, color = "Red")
plt.annotate(".. every 4 days", (x[-3], y[-1]), color = "Red", xycoords = "data", fontsize = 14, alpha = 0.8)

# plot Params
plt.xlabel("Days",fontsize = 17)
plt.ylabel("Number of Confirmed Cases",fontsize = 17)
plt.title("Trend Comparison of Different Countries\n and India (Confirmed) ",fontsize = 22)
plt.legend(loc = "upper left")
plt.yscale("log")
plt.grid(which = "both")
plt.savefig(out+'PLOT/Trend_Comparison_with_India_Confirmed.png')
plt.show()


# Trend comparison for Deceased  Cases
gTSTMPD = gTSDDF.groupby('Country').sum().drop(["Lat", "Long"], axis=1).sort_values(gTSDDF.columns[-1], ascending=False)
threshold = 50
f = plt.figure(figsize=(10,12))
ax = f.add_subplot(111)
for i, Country in enumerate(gTSTMPD.index):
    if i >= 9:
        if Country != "India" and Country != "Japan":
            continue
    days = 60
    gTST2 = gTSTMPD.loc[gTSTMPD.index == Country].values[0]
    gTST2 = gTST2[gTST2 > threshold][:days]
    date = np.arange(0, len(gTST2[:days]))
    xnew = np.linspace(date.min(), date.max(), 30)
    spl = make_interp_spline(date, gTST2, k = 1)
    power_smooth = spl(xnew)
    if Country != "India":
        plt.plot(xnew, power_smooth, '-o', label = Country, linewidth = 3, markevery = [-1])
    else:
        marker_style = dict(linewidth = 4, linestyle = '-', marker = 'o', markersize = 10, markerfacecolor = '#ffffff')
        plt.plot(date, gTST2, "-.", label = Country, **marker_style)

plt.tick_params(labelsize = 14)        
plt.xticks(np.arange(0, days, 7), [ "Day " + str(i) for i in range(days)][::7])     

# Reference lines 
x = np.arange(0, 18)
y = 2**(x + np.log2(threshold))
plt.plot(x, y, "--", linewidth =2, color = "gray")
plt.annotate("No. of cases doubles every day", (x[-2], y[-1]), xycoords = "data", fontsize = 14, alpha = 0.5)

x = np.arange(0, int(days - 12))
y = 2**(x / 2 + np.log2(threshold))
plt.plot(x, y, "--", linewidth = 2, color = "gray")
plt.annotate(".. every second day", (x[-3], y[-1]), xycoords = "data", fontsize = 14, alpha = 0.5)

x = np.arange(0, int(days - 4))
y = 2**(x / 7 + np.log2(threshold))
plt.plot(x, y, "--", linewidth = 2, color = "gray")
plt.annotate(".. every week", (x[-3], y[-1]), xycoords = "data", fontsize = 14, alpha = 0.5)

x = np.arange(0, int(days - 4))
y = 2**(x / 30 + np.log2(threshold))
plt.plot(x, y, "--", linewidth = 2, color = "gray")
plt.annotate(".. every month", (x[-3], y[-1]), xycoords = "data", fontsize = 14, alpha = 0.5)
 
x = np.arange(0, int(days - 5))
y = 2**(x / 4 + np.log2(threshold))
plt.plot(x, y, "--", linewidth = 2, color = "Red")
plt.annotate(".. every 4 days", (x[-3], y[-1]), color = "Red", xycoords = "data", fontsize = 14, alpha = 0.8)

# plot Params
plt.xlabel("Days",fontsize = 17)
plt.ylabel("Number of Deceased Cases",fontsize = 17)
plt.title("Trend Comparison of Different Countries\n and India (Deceased) ",fontsize = 22)
plt.legend(loc = "upper left")
plt.yscale("log")
plt.grid(which = "both")
plt.savefig(out+'PLOT/Trend_Comparison_with_India_Deceased.png')
plt.show()