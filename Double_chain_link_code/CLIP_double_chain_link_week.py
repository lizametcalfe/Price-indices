import pandas as pd
import numpy as np

unit2014= pd.read_csv("/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/CLIPweek2014.csv")
unit2015= pd.read_csv("/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/CLIPweek2015.csv")
unit2016= pd.read_csv("/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/CLIPweek2016.csv")

#rebase to June 2014 if neccessary
def rebasetojune(data,datevar,basedate,index):
    date = np.unique(data[datevar])
    basedate = data[data[datevar]==basedate][index]
    data["CLIP"] = data["CLIP"].apply(lambda x: (x/basedate)*100)
    return data    

unit14=[]
unit14.append(unit2014.groupby('ons_item_name').apply(lambda L: rebasetojune(L,"period",201423,"CLIP")))

unit14 = np.concatenate(unit14, axis=0)  # axis = 1 would append things as new columns
unit14=pd.DataFrame(unit14)
unit2014.columns=["Unnamed: 0",  "CLIP","i", "ons_item_name", "period"]

#weight up item level
weights=pd.read_excel('/home/mint/my-data/Web_scraped_CPI/Code_upgrade/New_version/Copy of weights-lager.xls')
weights["ons_item_name"]=weights["ons_item_name"].apply(lambda x: x.lower())
weightsupper = pd.read_excel('/home/mint/my-data/Web_scraped_CPI/Code_upgrade/New_version/Weights_upper.xls')
def aggregation(indices, priceindices,weights,frequency, weights14,weights15, weights16, agglevel,weightjoinvar):
    data=pd.merge(indices,weights,left_on=weightjoinvar,right_on="ons_item_name",how="outer").reset_index()
    index=data['index']
    if frequency=="month" or frequency=="fortnight" or frequency == "weekly":
        period=np.floor(data['period'].astype(float)/100)
        for x in index.dropna():
            if period[x]==2014:
                data.loc[x,"weighted_index"]=data.loc[x,priceindices]*data.loc[x,weights14]
            elif period[x]==2015:
                data.loc[x,"weighted_index"]=data.loc[x,priceindices]*data.loc[x,weights15]
            if period[x]==2016:
                data.loc[x,"weighted_index"]=data.loc[x,priceindices]*data.loc[x,weights16]
    elif frequency=="daily":
        period=np.floor(data['period']/10000)
        for x in index:
            if period[x]==2014:
                data.loc[x,"weighted_index"]=data.loc[x,priceindices]*data.loc[x,weights14]
            elif period[x]==2015:
                data.loc[x,"weighted_index"]=data.loc[x,priceindices]*data.loc[x,weights15]
            elif period[x]==2016:
                data.loc[x,"weighted_index"]=data.loc[x,priceindices]*data.loc[x,weights16]
    b=data.groupby(by=[agglevel,"period"])["weighted_index"].sum()
    return b.reset_index()



#double chain link in two parts
#chain jan to dec at coicop4 level
def itemchain(df,rebaseto,baseperiod):
    df=df.reset_index()
    df=df.reset_index()
    index=df['level_0']
    if rebaseto == "Dec":
        df.loc[len(df.index)-1,"weighted_index"] = df.loc[len(df.index)-1,"weighted_index"]/df.loc[len(df.index)-2,"weighted_index"]*100
        df["year"] = str(df.loc[len(df.index)-1,"period"])[0:4]
    elif rebaseto == "Jan":
        for x in index:
            period = df["period"]
            if period[x]==baseperiod:
                per = x
                df.loc[x,"weighted_index"] = float(df.loc[x,"weighted_index"])*float(df.loc[x-1,"weighted_index"])/100 if df.loc[x,"period"]==baseperiod else df.loc[x,"weighted_index"]
            if period[x] > baseperiod:
                df.loc[x,"weighted_index"] = float(df.loc[x,"weighted_index"])*float(df.loc[df["period"]==baseperiod,"weighted_index"])/100

                #        df.loc[len(df.index)-1,"weighted_index"] = df.loc[len(df.index)-1,"weighted_index"]*df.loc[len(df.index)-2,"weighted_index"]/100
#        df["weighted_index"] = df.apply(lambda row: float(row["weighted_index"])*float(df.loc[df["period"]==201501,"weighted_index"])/100 if row["year"]=="2015" else row["weighted_index"], axis=1)
    else:
        df = "Wrong"
    return df


def doublechainlink(originalyear, chainedyear,chainedyear2, basedate,basedate2,freq):
    originalyear = aggregation(originalyear, "CLIP",weights,"month","weight_2_2014","weight_2_2015","weight_2_2016","level_2","ons_item_name")
    chainedyear = aggregation(chainedyear, "CLIP",weights,"month","weight_2_2014","weight_2_2015","weight_2_2016","level_2","ons_item_name")
    chainedyear2 = aggregation(chainedyear2, "CLIP",weights,"month","weight_2_2014","weight_2_2015","weight_2_2016","level_2","ons_item_name")
    #single chain link for new year added
    a14=[]
    a14.append(originalyear.groupby('level_2').apply(lambda L: itemchain(L,"Dec",basedate)))
    a14 = np.concatenate(a14, axis=0)
    a14 = pd.DataFrame(a14)
    a14.columns=["Unnamed: 0","Unnamed: 1",  "level_2", "period","CLIP","year"]
    a15=[]
    a15.append(chainedyear.groupby('level_2').apply(lambda L: itemchain(L,"Dec",basedate2)))
    a15 = np.concatenate(a15, axis=0)
    a15 = pd.DataFrame(a15)
    a15.columns=["Unnamed: 0","Unnamed: 1",  "level_2", "period","CLIP","year"]
    unit2014agg2 = aggregation(a14, "CLIP",weightsupper,freq,"weight_1_2014","weight_1_2015","weight_1_2016","level_3","level_2")
    unit2015agg2 = aggregation(a15, "CLIP",weightsupper,freq,"weight_1_2014","weight_1_2015","weight_1_2016","level_3","level_2")
    unit2016agg2 = aggregation(chainedyear2, "weighted_index",weightsupper,freq,"weight_1_2014","weight_1_2015","weight_1_2016","level_3","level_2")
    unit2015agg2 = unit2015agg2[unit2015agg2["period"]!=basedate]
    unit2016agg2 = unit2016agg2[unit2016agg2["period"]!=basedate2]
    aaa1415=pd.concat([unit2014agg2,unit2015agg2],axis=0)
    aaa15=[]
    aaa15.append(aaa1415.groupby('level_3').apply(lambda L: itemchain(L,"Jan",basedate)))
    aaa15 = np.concatenate(aaa15, axis=0)
    aaa15 = pd.DataFrame(aaa15)
    aaa15.columns=["Unnamed: 0", "Unnamed: 1", "level_3", "period","weighted_index"]
    aaa1516=pd.concat([aaa15,unit2016agg2],axis=0)
    aaa16=[]
    aaa16.append(aaa1516.groupby('level_3').apply(lambda L: itemchain(L,"Jan",basedate2)))
    aaa16 = np.concatenate(aaa16, axis=0)
    aaa16 = pd.DataFrame(aaa16)
    aaa16.columns=["Unnamed: 0", "Unnamed: 1","Unnamed: 2", "Unnamed: 3", "level_3", "period","weighted_index"]
    return aaa16


upto2016 = doublechainlink(unit2014, unit2015,unit2016, 201501,201601,"month")

upto2016.to_csv("/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/CLIPdoublechainedweek.csv")