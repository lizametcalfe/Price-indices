# -*- coding: utf-8 -*-
"""
Created on Wed May  4 13:07:39 2016

@author: liz metcalfe

Date:   07/04/2016
Verion: 0.1

Inputs:  Price data 
outputs: DataFrame with chained index for all time periods    
"""

import os
import pandas as pd
import numpy as np
import math
from scipy.stats import gmean

####web scraped price data
x = pd.read_csv('.../pricedata.csv',encoding="latin_1")
x=x[x['item_price_num']>0]

x["idvar"]=x["product_name"]+"_"+x["store"]


x["months"] = x["monthday"].apply(lambda x: str(x)[0:6])

def unitdate(data, date, datevar, pricevar):
    data_prices = data.loc[data[datevar].astype(int) == int(date)]
    data_prices = data_prices[["product_name",pricevar]]
    data_prices = data_prices.groupby(["product_name"])
    data_prices = data_prices[pricevar].agg({"gmean":gmean})
    data_prices.reset_index(inplace=True)
    return data_prices

def unitprice(data,idvar,classvar,datevar,pricevar,basedate, year):
    classvalue = pd.unique(data[classvar])[0]
    data["years"] = data[datevar].apply(lambda x: str(x)[0:4])
    dataa = data[data["years"] == str(year)]
    datab = data[data[datevar] == str(year-1)+"12"]
    data = pd.concat([dataa,datab],axis=0)
    date = pd.unique(data[datevar])
    df1 = pd.DataFrame({"i" : range(0,len(date)),"period":date,"ons_item_number":classvalue})
    df1["unit"]="empty"
    base = unitdate(data, basedate, datevar, pricevar)
    base["base_price"] = base["gmean"]
    del base["gmean"]
    for i in date:
        try:
            datamerged=pd.merge(base,unitdate(data,int(i), datevar, pricevar), how='inner', on='product_name')
            datamerged.loc[:,'price_relative'] = datamerged.loc[:,'gmean']/datamerged.loc[:,'base_price']
            datamerged['pr_log'] = datamerged['price_relative'].apply(math.log)
            datamerged["groups"] = 1
            test1 = datamerged.groupby('groups')
            lopp = test1['pr_log'].apply(np.mean).apply(np.exp)*100
            if lopp.empty:
                lopp = 100
            else:
                lopp = float(lopp)
            df1["unit"][df1["period"]==i] = lopp
        except:
            df1["unit"][df1["period"]==i] = 100
#    print(df1["ons_item_number"])
    return df1.sort_values("period")

#%%
a14=[]
a14.append(x.groupby('ons_item_no').apply(lambda L: unitprice(L,'idvar', 'ons_item_no','months','item_price_num', '201406',2014)))

a15=[]
a15.append(x.groupby('ons_item_no').apply(lambda L: unitprice(L,'idvar', 'ons_item_no','months','item_price_num', '201501',2015)))

a16=[]
a16.append(x.groupby('ons_item_no').apply(lambda L: unitprice(L,'idvar', 'ons_item_no','months','item_price_num', '201601',2016)))

#For each year 2014, 2015, 20016
#%%
Results = np.concatenate(a14, axis=0)  # axis = 1 would append things as new columns
results2=pd.DataFrame(Results)
results2.columns=["i",  "ons_item_number", "period",     "unit"]
results2.to_csv(".../unit2014.csv")

Results15 = np.concatenate(a15, axis=0)  # axis = 1 would append things as new columns
results215=pd.DataFrame(Results15)
results215.columns=["i",  "ons_item_number", "period",     "unit"]
results215.to_csv(".../unit2015.csv")

Results16 = np.concatenate(a16, axis=0)  # axis = 1 would append things as new columns
results216=pd.DataFrame(Results16)
results216.columns=["i",  "ons_item_number", "period",     "unit"]
results216.to_csv(".../unit2016.csv")

#The results are the product level price indices, run double chain link code to aggregate up price indices and link together the different years.
