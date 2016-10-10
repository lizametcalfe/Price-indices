
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


x=x[x['item_price_num']>0]
#####daily (updated) ######
#x = pd.read_csv('L:/Branch folders/Index Numbers/Research - web scraping/Data April 16/data_imputed_20160504.csv',encoding="latin_1")
#x=x[x['item_price_num']>0]
#########weekly#########

####fortnightly##########

x["idvar"]=x["product_name"]+"_"+x["store"]

##for single item run this part####
#x1=x[x["ons_item_no"]==212720]
#%%

#x1=x[x["ons_item_no"]==212720]

#%%

####monthly (updated) ########

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

x= weekly(x,x["monthday"])

x["yearweekno"]=x["yearweekno"].apply(lambda x: 201610 if x == 201609 else x)
x["yearweekno"]=x["yearweekno"].apply(lambda x: 201609 if x == 201608 else x)
x["yearweekno"]=x["yearweekno"].apply(lambda x: 201608 if x == 201607 else x)
x["yearweekno"]=x["yearweekno"].apply(lambda x: 201607 if x == 201606 else x)
x["yearweekno"]=x["yearweekno"].apply(lambda x: 201606 if x == 201605 else x)
x["yearweekno"]=x["yearweekno"].apply(lambda x: 201605 if x == 201604 else x)
x["yearweekno"]=x["yearweekno"].apply(lambda x: 201604 if x == 201603 else x)
x["yearweekno"]=x["yearweekno"].apply(lambda x: 201603 if x == 201602 else x)
x["yearweekno"]=x["yearweekno"].apply(lambda x: 201602 if x == 201601 else x)
x["yearweekno"]=x["yearweekno"].apply(lambda x: 201601 if x == 201553 else x)


x=x[x["yearweekno"]>201422]

x["months"]=x["yearweekno"]
del x["yearweekno"]


x14 = x[x["months"]<201502]
x15 = x[x["months"]>=201501]
x15 = x15[x15["months"]<201602]
x16 = x[x["months"]>=201601]

del x14["month"]
del x15["month"]
del x16["month"]



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
    datab = data[data[datevar].astype(int) == int(str(year+1)+"01")]
    data = pd.concat([dataa,datab],axis=0)
    date = pd.unique(data[datevar])
    df1 = pd.DataFrame({"i" : range(0,len(date)),"period":date,"ons_item_no":classvalue})
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

#%time 
#Index1 = chaineddaily(x1,"idvar","ons_item_no","month","item_price_num")

def runthrough(data,basedate,date):
    a = []
    for i in np.unique(data["ons_item_no"]):
        a.append(unitprice(data[data["ons_item_no"] == i],  'idvar', 'ons_item_no','months','item_price_num', basedate,date))
    aa = np.concatenate(a, axis=0)  # axis = 1 would append things as new columns
    aa=pd.DataFrame(aa)
    aa.columns=["i",  "ons_item_number", "period", "unit"]
 
    return aa

aa=pd.DataFrame()
a = runthrough(x14, 201423, 2014)
a.to_csv("/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/unitweekly2014.csv")

b = runthrough(x15, 201501, 2015)
b.to_csv("/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/unitweekly2015.csv")

c = runthrough(x16, 201601, 2016)
c.to_csv("/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/unitweekly2016.csv")