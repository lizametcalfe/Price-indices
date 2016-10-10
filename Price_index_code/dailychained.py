# -*- coding: utf-8 -*-
"""
Created on Friday 20th May

@author: Liz

Date:   20/05/2016
Verion: 1

Inputs:  Price data 
outputs: DataFrame with chained index for all time periods
changes - 
    added updated aggregation code
    
"""

import os
import pandas as pd
import numpy as np
import math
from scipy.stats import gmean

#%%
#Import Dataset
#
#os.chdir('D:/webscraped/New_Data')

####monthly (updated) ########
x = pd.read_csv('/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/august_offer.csv',encoding="utf-8")

#x["month"] = x["yearweekno"]

x=x[x['item_price_num']>0]


def weekly(df, monthday):
    df["monthday"]=monthday.apply(lambda x: str(x))
    df["day"] = df["monthday"].str[6:].astype(str)
    df["day"]=df["day"].apply(lambda x: str(x)[0:2])
    df["monthss"]=df["monthday"].str[4:6].astype(str)
    df["year"]=df["monthday"].str[:4].astype(str)
    df["dates"]=pd.to_datetime(df.day + df.monthss + df.year, format="%d%m%Y")
    df['datessp'] = df['dates'].map(lambda x: x.isocalendar()[0])
    df['datess'] = df['dates'].map(lambda x: x.isocalendar()[1])
    df["datess"]=df["datess"].apply(lambda x: "0"+str(x) if len(str(x)) == 1 else str(x))
    df["yearweekno"]=df['datessp'].astype(str)+df["datess"].astype(str)
    df["yearweekno"] = df["yearweekno"].apply(lambda x: int(x))
    return df

def oddeven(num):
    num = int(num)
    if (num % 2) == 0:
        num =num
    else:
        num = num - 1
    return num

x["idvar"]=x["product_name"]+"_"+x["store"]


#%%
def chaineddaily(data,idvar,classvar,datevar,pricevar):
    classvalue = pd.unique(data[classvar])
    x2 = pd.DataFrame({"ID":data[idvar],"Date":data[datevar],"Price":data[pricevar]})
    x2 = x2.sort_values(['ID','Date'])
    T = len(np.unique(x2["Date"]))
    x2['lagprice'] = x2.groupby('ID')['Price'].shift(+1)
    x2["daily_rel"]=x2['Price']/x2['lagprice']
    x2['pr_log'] = x2['daily_rel'].apply(math.log)
    date=np.unique(x2['Date'])
    x2 = x2.groupby('Date')
    x2 = x2['pr_log'].apply(np.mean).apply(np.exp)*100
    C = x2.reset_index()['pr_log']
    df = pd.DataFrame({'Daily': C}) 
    df["Chained"] = np.cumprod(df['Daily']/100, axis=None, dtype=None, out=None)
    Chained = df['Chained']*100
    Daily = df['Daily']
    df1 = pd.DataFrame({"i" : range(0,len(date)),"period":date,"ChainedDaily":Chained,"Daily":Daily, "ons_item_number":classvalue[0]})
    df1.index = range(T)
    df1 = df1.fillna(100)
    return df1

#%%
a =  []
#%time 
a.append(x.groupby('ons_item_no').apply(lambda L: chaineddaily(L, 'idvar', 'ons_item_no','monthday','item_price_num')))

####for monthly: aveactprice

#%%
Results = np.concatenate(a, axis=0)  # axis = 1 would append things as new columns
results2=pd.DataFrame(Results)
results2.columns=["ChainedDaily","Daily","i","ons_item_no","period"]

results2.to_csv("/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/DC_daily_itemlevel.csv")

#aggregate
#%%
weights=pd.read_excel('/home/mint/my-data/Web_scraped_CPI/Code_upgrade/New_version/Copy of weights-lager.xls')
def aggregation(indices,weights,frequency):
    data=pd.merge(indices,weights,left_on="ons_item_no",right_on="join",how="outer")
    data = data.dropna()
    index=data['Unnamed: 0']
    data["weighted_index"]=0
    if frequency=="month" or frequency=="fortnight" or frequency == "weekly":
        period=np.floor(data['period']/100)
        for x in index:
            if period[x]==2014:
                data.loc[x,"weighted_index"]=data.loc[x,"ChainedDaily"]*data.loc[x,'weight_1_2014']*data.loc[x,'weight_2_2014']
            elif period[x]==2015:
                data.loc[x,"weighted_index"]=data.loc[x,"ChainedDaily"]*data.loc[x,'weight_1_2015']*data.loc[x,'weight_2_2015']
            elif period[x]==2016:
                data.loc[x,"weighted_index"]=data.loc[x,"GEKSJ"]*data.loc[x,'weight_1_2016']*data.loc[x,'weight_2_2016']
    elif frequency=="daily":
        period=np.floor(data['period']/10000)
        for x in index:
            if period[x]==2014:
                data.loc[x,"weighted_index"]=data.loc[x,"ChainedDaily"]*data.loc[x,'weight_1_2014']*data.loc[x,'weight_2_2014']
            elif period[x]==2015:
                data.loc[x,"weighted_index"]=data.loc[x,"ChainedDaily"]*data.loc[x,'weight_1_2015']*data.loc[x,'weight_2_2015']
            elif period[x]==2016:
                data.loc[x,"weighted_index"]=data.loc[x,"ChainedDaily"]*data.loc[x,'weight_1_2016']*data.loc[x,'weight_2_2016']
    #WGEKSJ=np.sum(df["weighted_index"])
    #print(WGEKSJ)
    #out=pd.DataFrame({"period":pd.unique(data["period"]),"GEKSJ":WGEKSJ,"Division":pd.unique(data['Top'])})
    b=data.groupby(by=["level_3","period"])["weighted_index"].sum()
    b=b.reset_index()
    return(b)
   

#%%

##change to daily chained
Dail_Imputed=pd.read_csv("/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/DC_daily_itemlevel.csv")
Dail_Agg_imputed=aggregation(Dail_Imputed,weights,"daily")
Dail_Agg_imputed.to_csv("/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/DC_daily_agg.csv")

df= weekly(Dail_Imputed,Dail_Imputed["period"])
df1 = weekly(Dail_Agg_imputed,Dail_Agg_imputed["period"])

df["yearweekno"]=df["yearweekno"].apply(lambda x: 201610 if x == 201609 else x)
df["yearweekno"]=df["yearweekno"].apply(lambda x: 201609 if x == 201608 else x)
df["yearweekno"]=df["yearweekno"].apply(lambda x: 201608 if x == 201607 else x)
df["yearweekno"]=df["yearweekno"].apply(lambda x: 201607 if x == 201606 else x)
df["yearweekno"]=df["yearweekno"].apply(lambda x: 201606 if x == 201605 else x)
df["yearweekno"]=df["yearweekno"].apply(lambda x: 201605 if x == 201604 else x)
df["yearweekno"]=df["yearweekno"].apply(lambda x: 201604 if x == 201603 else x)
df["yearweekno"]=df["yearweekno"].apply(lambda x: 201603 if x == 201602 else x)
df["yearweekno"]=df["yearweekno"].apply(lambda x: 201602 if x == 201601 else x)
df["yearweekno"]=df["yearweekno"].apply(lambda x: 201601 if x == 201553 else x)

df1["yearweekno"]=df1["yearweekno"].apply(lambda x: 201610 if x == 201609 else x)
df1["yearweekno"]=df1["yearweekno"].apply(lambda x: 201609 if x == 201608 else x)
df1["yearweekno"]=df1["yearweekno"].apply(lambda x: 201608 if x == 201607 else x)
df1["yearweekno"]=df1["yearweekno"].apply(lambda x: 201607 if x == 201606 else x)
df1["yearweekno"]=df1["yearweekno"].apply(lambda x: 201606 if x == 201605 else x)
df1["yearweekno"]=df1["yearweekno"].apply(lambda x: 201605 if x == 201604 else x)
df1["yearweekno"]=df1["yearweekno"].apply(lambda x: 201604 if x == 201603 else x)
df1["yearweekno"]=df1["yearweekno"].apply(lambda x: 201603 if x == 201602 else x)
df1["yearweekno"]=df1["yearweekno"].apply(lambda x: 201602 if x == 201601 else x)
df1["yearweekno"]=df1["yearweekno"].apply(lambda x: 201601 if x == 201553 else x)

#assign weekly and fortnightly index, order is important here!
df["yearfortno"] = df['yearweekno'].apply(lambda x: oddeven(x))
df1["yearfortno"] = df1['yearweekno'].apply(lambda x: oddeven(x))
df=df[df["yearweekno"]>201422]

df["months"] = df["period"].apply(lambda x: str(x)[0:6])
df1["months"] = df1["period"].apply(lambda x: str(x)[0:6])

def averagedaily(data,datevar, pricevar, classvar):
#    x2 = pd.DataFrame({"Date":data[datevar],"Price":data[pricevar]})
    classvalue = pd.unique(data[classvar])
    alc=pd.DataFrame()
    alc["month"]=data[datevar]
    alc["indices"] = data[pricevar]
    alc = alc.groupby("month")
    alc1 = alc.agg(gmean)
    date = alc1.reset_index()["month"]
    the = alc1["indices"][0:1]
    alc1["rebasedindices"] = alc1["indices"].apply(lambda x: (x/the)*100)
    Daily = alc1["rebasedindices"]
    df1 = pd.DataFrame({"Chainedaveragerebased":Daily, "level_3":classvalue[0]})
    return df1.reset_index()

b=[]
b.append(df.groupby('ons_item_no').apply(lambda L: averagedaily(L, 'yearweekno','ChainedDaily','ons_item_no')))
b1=[]
b1.append(df1.groupby('level_3').apply(lambda L: averagedaily(L, 'yearweekno','weighted_index','level_3')))


c=[]
c.append(df.groupby('ons_item_no').apply(lambda L: averagedaily(L, 'months','ChainedDaily','ons_item_no')))

c1=[]
c1.append(df1.groupby('level_3').apply(lambda L: averagedaily(L, 'months','weighted_index','level_3')))



d=[]
d.append(df.groupby('ons_item_no').apply(lambda L: averagedaily(L, 'yearfortno','ChainedDaily','ons_item_no')))
d1=[]
d1.append(df1.groupby('level_3').apply(lambda L: averagedaily(L, 'yearfortno','weighted_index','level_3')))

#take average and re-base

#e = b

Resultsav = np.concatenate(b, axis=1)  # axis = 1 would append things as new columns
resultsav2=pd.DataFrame(Resultsav)
resultsav2.columns=["month","Chainedaverage","level"]

resultsav2.to_csv("/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/DC_weekly_itemlevel.csv")

Resultsavmonth = np.concatenate(c, axis=1)  # axis = 1 would append things as new columns
Resultsavmonth=pd.DataFrame(Resultsavmonth )
Resultsavmonth.columns=["month","Chainedaverage","level"]
Resultsavmonth.to_csv("/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/DC_monthly_itemlevel.csv")

Resultsavfort = np.concatenate(d, axis=1)  # axis = 1 would append things as new columns
Resultsavfort=pd.DataFrame(Resultsavfort)
Resultsavfort.columns=["month","Chainedaverage","level"]
Resultsavfort.to_csv("/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/DC_fortnightly_itemlevel.csv")

Resultsav1 = np.concatenate(b1, axis=1)  # axis = 1 would append things as new columns
resultsav21=pd.DataFrame(Resultsav1)
resultsav21.columns=["month","Chainedaverage","level"]

resultsav21.to_csv("/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/DC_weekly_agg.csv")

Resultsavmonth1 = np.concatenate(c1, axis=1)  # axis = 1 would append things as new columns
Resultsavmonth1=pd.DataFrame(Resultsavmonth1 )
Resultsavmonth1.columns=["month","Chainedaverage","level"]
Resultsavmonth1.to_csv("/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/DC_monthly_agg.csv")

Resultsavfort1 = np.concatenate(d1, axis=1)  # axis = 1 would append things as new columns
Resultsavfort1=pd.DataFrame(Resultsavfort1)
Resultsavfort1.columns=["month","Chainedaverage","level"]
Resultsavfort1.to_csv("/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/DC_fortnightly_agg.csv")
