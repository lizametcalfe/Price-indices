# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 16:02:31 2016

@author: tomhuntersmith
"""

#####This code derives features from the "offer" variable and stores them in a new variable "offer_cat"

import sys    # sys.setdefaultencoding is cancelled by site.py
reload(sys)    # to re-enable sys.setdefaultencoding()
sys.setdefaultencoding('utf-8')


import pandas as pd
import datetime
import time


data = pd.read_csv("/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/august26_imputed_all.csv",encoding ="utf-8")
def offer_check(x):
    terms = ["buy","purchase","the"]
    for j in terms:
        if j in x: 
            return "BOGO"         
    terms = ["off","was","save","half","now","only"] # perhaps add,only??
    for j in terms:
        if j in x:
            return "Discount"
    terms = ["special purchase"]
    for j in terms:
        if j in x: 
            return "Special Purchase"
    terms = ["add","get"]
    for j in terms:
        if j in x: 
            return "add more"
    terms = ["clear","reduced"]
    for j in terms:
        if j in x:
            return "reduced to clear"

def produce_features(df):
    df["offer"] =df["offer"].apply(lambda x: str(x))
    df["offer"] =df["offer"].apply(lambda x: x.lower())
    df["offer_cat"] =df["offer"].apply(lambda x: offer_check(x))
    df.to_csv("/home/mint/my-data/Web_scraped_CPI/Code_upgrade/data/august_offer.csv", encoding="utf-8",index = False)


    
    
start_time = time.time()
data2 = produce_features(data)
print("Time taken to format: --- %s seconds ---" % (time.time() - start_time))

  
#%% if you do need to split the data this code will patch it back togther
    
