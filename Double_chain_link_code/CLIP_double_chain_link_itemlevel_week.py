import pandas as pd
import numpy as np

unit2014= pd.read_csv("/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/CLIPweek2014.csv")
unit2015= pd.read_csv("/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/CLIPweek2015.csv")
unit2016= pd.read_csv("/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/CLIPweek2016.csv")

#rebase to June 2014 if neccessary

#rebase to June 2014 if neccessary
def rebasetojune(data,datevar,basedate,index):
    date = np.unique(data[datevar])
    basedate = data[data[datevar]==basedate][index]
    data["CLIP"] = data["CLIP"].apply(lambda x: (x/basedate)*100)
    return data    


def runthrough(data,basedate,date, index):
    a = []
    for i in np.unique(data["ons_item_name"]):
        a.append(rebasetojune(data[data["ons_item_name"] == i],  date, basedate, index))
    aa = np.concatenate(a, axis=0)  # axis = 1 would append things as new columns
    aa=pd.DataFrame(aa)
    aa.columns=["i","CLIP","ii", "ons_item_name", "period"]
    del aa["ii"]
    return aa

unit2014 = runthrough(unit2014,201424,"period","CLIP")

#double chain link in two parts
#chain jan to dec at coicop4 level

#double chain link in two parts
#chain jan to dec at coicop4 level
def itemchain(df,baseperiod,indices):
    df=df.reset_index()
    df=df.reset_index()
    index=df['level_0']
    for x in index:
        period = df["period"]
        if period[x] > baseperiod:
            df.loc[x,indices] = float(df.loc[x,indices])*float(df.loc[df["period"]==baseperiod,indices])/100
                #        df.loc[len(df.index)-1,"weighted_index"] = df.loc[len(df.index)-1,"weighted_index"]*df.loc[len(df.index)-2,"weighted_index"]/100
#        df["weighted_index"] = df.apply(lambda row: float(row["weighted_index"])*float(df.loc[df["period"]==201501,"weighted_index"])/100 if row["year"]=="2015" else row["weighted_index"], axis=1)
    return df


def singlechainlink(originalyear, chainedyear,chainedyear2, basedate,basedate2,freq,priceindex):
    #single chain link for new year added
    unit2015 = chainedyear[chainedyear["period"]!=basedate]
    unit2016 = chainedyear2[chainedyear2["period"]!=basedate2]
    aaa1415=pd.concat([originalyear,unit2015],axis=0)
    aaa15=[]
    aaa15.append(aaa1415.groupby('ons_item_name').apply(lambda L: itemchain(L,basedate,priceindex)))
    aaa15 = np.concatenate(aaa15, axis=0)
    aaa15 = pd.DataFrame(aaa15)
    aaa15.columns=["Unnamed: 0","Unnamed: 1", priceindex, "NaN", "Unnamed: 2","ons_item_name","period"]
    aaa15 = aaa15[[priceindex, "ons_item_name","period"]]
    aaa1516=pd.concat([aaa15,unit2016],axis=0)
    aaa16=[]
    aaa16.append(aaa1516.groupby('ons_item_name').apply(lambda L: itemchain(L,basedate2,priceindex)))
    aaa16 = np.concatenate(aaa16, axis=0)
    aaa16 = pd.DataFrame(aaa16)
    aaa16.columns=["Unnamed: 0", "Unnamed: 1",priceindex, "Unnamed: 3","Unnamed: 4", "ons_item_name", "period"]
    aaa16 = aaa16[["ons_item_name", "period", priceindex]]
    return aaa16

upto2016 = singlechainlink(unit2014, unit2015,unit2016, 201501,201601,"month","CLIP")


upto2016.to_csv("/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/CLIPdoublechaineditemlevelweek.csv")