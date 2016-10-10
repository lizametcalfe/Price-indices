# -*- coding: utf-8 -*-
"""
Created on Wed May  4 13:07:39 2016

@author: Liz Metcalfe

Date:   07/04/2016
Verion: 0.2

Inputs:  WS price data 
outputs: DataFrame with chained index for all time periods
changes - 
    added updated aggregation code
    
"""

import os
import pandas as pd
import numpy as np
import math
from scipy.stats import gmean
from fuzzywuzzy import fuzz

pd.options.display.max_seq_items = 2000
pd.options.display.max_colwidth = 1000

x = pd.read_csv('/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/august_offer.csv',encoding="utf-8")
x=x[x['item_price_num']>0]

def _removeNonAscii(s): return "".join(i for i in s if ord(i)<128)

x["product_name"]=x["product_name"].apply(lambda s: _removeNonAscii(s))

x["idvar"]=x["product_name"]+" "+x["store"]

x = x[["ons_item_no","product_name","store","monthday","item_price_num","idvar"]]

unwanted_words =["tesco","sainsbury","sainsbury's","tesco's","sainsburies","waitrose","waitrose's"]
x["product_name"] = x["product_name"].apply(lambda s: ' '.join(filter(lambda x: x.lower() not in unwanted_words,  s.split())))

fulldata = x


dates14 = pd.DataFrame()
dates14["date"] = [20140610,20140715, 20140812, 20140919, 20141014, 20141111, 20141209, 20150113]
dates15 = pd.DataFrame()
dates15["date"] = [20150113, 20150217, 20150317, 20150414, 20150512, 20150609, 20150714, 20150811, 20150908, 20151013, 20151110,20151215, 20160112]
dates16 = pd.DataFrame()
dates16["date"] = [20160112, 20160216, 20160315, 20160412, 20160510, 20160614, 20160712, 20160816]

fx14=fulldata[fulldata.monthday.isin(dates14["date"])]
fx15=fulldata[fulldata.monthday.isin(dates15["date"])]
fx16=fulldata[fulldata.monthday.isin(dates16["date"])]

def compreplace(data,sda,i):
    row=data[data["Missing"]=="Yes"][i-1:i]
    if row.empty:
        data =data[["product_name","ons_item_no","idvar","store","monthday","item_price_num","Missing"]]
        return data.sort_values("monthday")
    else:
        sda = sda[sda["ons_item_no"]==int(row["ons_item_no"])]
        k1 = sda[sda["monthday"] .isin(row["monthday"])]
        k1 = k1[k1["store"] .isin(row["store"])]
        d = k1.apply(lambda x: fuzz.ratio(x['product_name'], row['product_name']), axis=1)
        d = d[d >= 60]
        if len(d) != 0:
            v = k1.ix[d.idxmax(), ['store','monthday','item_price_num','product_name','idvar','ons_item_no']].values
            v= pd.Series(v, index=['store', 'monthday','item_price_num','product_name','idvar','ons_item_no'])
            v= pd.DataFrame([{"product_name":v["product_name"],"store":v["store"],"ons_item_no":v["ons_item_no"],"idvar":v["idvar"],"monthday":v["monthday"],"item_price_num":v["item_price_num"],"Missing":"Replaced"}])
            data = data[data["monthday"] != int(v["monthday"])] 
            data =data[["product_name","store","monthday","item_price_num","Missing","ons_item_no","idvar"]]
            data = data.append(v).sort_values("monthday") 
        else:
            data =data[["product_name","store","monthday","item_price_num","Missing","ons_item_no","idvar"]]
            data = data.sort_values("monthday")
    row2 = data[data["monthday"]==int(row["monthday"])]["idvar"].item()
    sdb = sda[sda["idvar"]==str(row2)]
    sdb["Missing"]= "Replaced"
    data2 = pd.concat([data.loc[data['item_price_num'].isnull()],sdb])
    data3 = pd.concat([data.loc[data['item_price_num'].notnull()],data2.drop_duplicates(subset='monthday',keep='last')])
    data3 = data3.drop_duplicates(subset='monthday',keep='first')
    return data3.sort_values("monthday")

def runfuzz(data,fx2):
    for i in range(1,6):
        the1 = compreplace(data,fx2,i)
        the2 = compreplace(the1,fx2,i)
        the3 = compreplace(the2,fx2,i)
        the4= compreplace(the3,fx2,i)
        the5 = compreplace(the4,fx2,i)
        the6 = compreplace(the5,fx2,i)
    return the6

def LCPI(x1,fx2, dates):
    dates["dateplus"]= dates["date"].apply(lambda s: int(s)+1)
    dates["datemin"]= dates["date"].apply(lambda s: int(s)-1)
    x2=x1[x1.monthday.isin(dates["date"])]
    if x2.empty:
        print("empty")
        return None
    xplus=x1[x1.monthday.isin(dates["dateplus"])]
    xmin=x1[x1.monthday.isin(dates["datemin"])]
#add missing column saying no
    x2["Missing"] = "No"
#mydata2["std_price_origin"] = mydata2["std_price"]
    xplus["Missing"] = "Plus replace"
    xmin["Missing"] = "Min replace"
    xplus["monthday"]=xplus["monthday"]-1
    xmin["monthday"]=xmin["monthday"]+1
    bplus=x2.append(xplus,ignore_index = True)
    bplus2= bplus.drop_duplicates(subset='monthday', take_last=False)
    bmin=bplus2.append(xmin,ignore_index = True)
    x3 = bmin.drop_duplicates(subset='monthday', take_last=False).sort_values("monthday")
    del dates["dateplus"]
    del dates["datemin"]
    table = pd.merge(x3, dates, how='outer', left_on='monthday', right_on='date')
    del table["monthday"]
    table["monthday"] = table["date"]
    del table["date"]
    table["ons_item_no"]  = table["ons_item_no"][table["ons_item_no"].notnull()].iloc[0]
    table["idvar"]  = table["idvar"][table["idvar"].notnull()].iloc[0]
    table["product_name"]  = table["product_name"][table["product_name"].notnull()].iloc[0]
    table["idvar"]  = table["idvar"][table["idvar"].notnull()].iloc[0]
    table["store"]  = table["store"][table["store"].notnull()].iloc[0]
    table["Missing"] = table["Missing"].apply(lambda x: "Yes" if pd.isnull(x) else x)
    data =table.sort_values("monthday")
    take =runfuzz(data,fx2)
    take2 =runfuzz(take,fx2)
    take3 =runfuzz(take2,fx2)
    take4 =runfuzz(take3,fx2)
    take4=take4.fillna(method="pad",limit=1)
    print(table["idvar"].ix[0])
    take4 =take4[["product_name","store","monthday","item_price_num","Missing","ons_item_no","idvar"]]
    return take4
print "done"

def runthrough(data, dates):
    a = []
    data = data.sort_values("idvar", ascending=False)
    for i in np.unique(data["idvar"]):
        a.append(LCPI(data[data["idvar"] == i], data, dates))
        aa = np.concatenate(a, axis=0)
        aa=pd.DataFrame(aa)
        aa.columns=["product_name","store","monthday","item_price_num","Missing","ons_item_no","idvar"]
    return aa

x14 = runthrough(fx14, dates14)
x14.to_csv("\home\mint\my-data\Web_scraped_CPI\Code_upgrade\data\LCPI14.csv")

x15 = runthrough(fx15, dates15)
x15.to_csv("osu\home\mint\my-data\Web_scraped_CPI\Code_upgrade\data\LCPI15.csv")

x16 = runthrough(fx16, dates16)
x16.to_csv("\home\mint\my-data\Web_scraped_CPI\Code_upgrade\data\LCPI16.csv")

