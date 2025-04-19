#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 10:51:00 2022

@author: emilypaulin
"""
#Look at the bottom section with the replace functions to see the problem. 
import pandas as pd
import psycopg2

#Connect to the database.
conn = psycopg2.connect() #redacted login
cur = conn.cursor()

#Retrieve data
cur.execute("SELECT respid FROM rutracker.responses")
respid = cur.fetchall()
cur.execute("SELECT qid FROM rutracker.responses")
qid=cur.fetchall()
cur.execute("SELECT answer FROM rutracker.responses")
answer=cur.fetchall()

#Populate original dataframe
d = {
     "respid":respid,
     "qid":qid,
     "answer":answer
     }
df = pd.DataFrame(d, columns=["respid", "qid","answer"])

#Pivot dataframe to go from long to wide data set
df2 = df.pivot(index="respid", columns="qid", values="answer")
print("Pivoted DF:")
print(df2)

#Take average of support in census cells
 
#question is qid=14, gender is qid=10, region is qid=7 (column labels as well)
#dataframe with city data to make data groupable by region
cur.execute("SELECT region_name FROM ruregions.regionmap")
region_name = cur.fetchall()
cur.execute("SELECT rgnid FROM ruregions.regionmap")
rgnid = cur.fetchall()
e = {
     "rgnid":(rgnid),
     "region_name":(region_name)
     }
cities = pd.DataFrame(e, columns = ["rgnid", "region_name"])

#Join df2 and cities dataframes for regionid instead of cityname
df3 = df2.join(cities.set_index("region_name"), on=[(7,)], how='left', sort=False)
print(df3[14,])

#Join newdf and df3 to make averageable values
#import data from supportvalues which will hopefully make this doable.
cur.execute("SELECT word FROM ruregions.supportvalues")
word = cur.fetchall()
cur.execute("SELECT number FROM ruregions.supportvalues")
Number = cur.fetchall()
h = {
     "word":(word),
     "number":(Number)
     }
newdf = pd.DataFrame(h)
print(newdf)

df4 = df3.join(newdf.set_index("word"), on=[(14,)], how="left", sort=False)
print(df4)

#Make numerical to average!!!
df4["number"] = df4["number"].astype(str).str.replace("\(","")
df4["number"] = df4["number"].astype(str).str.replace(",\)","")
df4["number"] = pd.to_numeric(df4["number"], errors='coerce')
df4["rgnid"] = df4["rgnid"].astype(str).str.replace("\(","")
df4["rgnid"] = df4["rgnid"].astype(str).str.replace(",\)","")
df4.rename(columns={(10,):"sex", (2,):"date"}, inplace=True)
df4["sex"] = df4["sex"].astype(str).str.replace("\(","")
df4["sex"] = df4["sex"].astype(str).str.replace(",\)","")
df4["sex"] = df4["sex"].astype(str).str.replace("\'","")
df4["date"] = df4["date"].astype(str).str.replace("\(","")
df4["date"] = df4["date"].astype(str).str.replace(",\)","")
df4["date"] = df4["date"].astype(str).str.replace("\'","")
df4["date"] = pd.to_datetime(df4["date"], format="%Y/%m/%d %H:%M:%S")
df4["date"] = df4["date"].dt.date

#Group data and apply function
supportbycell = df4.groupby(["date", "rgnid", "sex"])["number"].mean()
supportbycell = supportbycell.reset_index()
print(supportbycell)
supportbycell.to_csv("supportbycell.csv", index=False)

#not part of original average of support. This was to create the training df with support.
train_df = pd.DataFrame(columns=["text", "support"])
for chunk in pd.read_csv("train.csv", chunksize = 25000, encoding="utf-8"):
    train_df = pd.concat([train_df, chunk])
del train_df["Unnamed: 0"]
train_df["sex"] = train_df["sex"].astype(str).str.replace("1.0","female")
train_df["sex"] = train_df["sex"].astype(str).str.replace("2.0","male")
train_df["date"] = pd.to_datetime(train_df["date"])

test_df = pd.DataFrame(columns=["text","support"])
for chunk in pd.read_csv("test.csv", chunksize = 25000, encoding="utf-8"):
    test_df = pd.concat([test_df, chunk])
del test_df["Unnamed: 0"]
test_df["sex"] = test_df["sex"].astype(str).str.replace("1.0","female")
test_df["sex"] = test_df["sex"].astype(str).str.replace("2.0","male")
test_df["date"] = pd.to_datetime(test_df["date"])
    
supportbycell = pd.DataFrame(columns=["date","rgnid","sex","number"])
for chunk in pd.read_csv("supportbycell.csv", chunksize = 25000, encoding="utf-8"):
    supportbycell = pd.concat([supportbycell, chunk])
supportbycell["date"] = pd.to_datetime(supportbycell["date"])
supportbycell["rgnid"] = pd.to_numeric(supportbycell["rgnid"])
supportbycell["number"] = pd.to_numeric(supportbycell["number"])
    
train_df = train_df.merge(supportbycell, how="outer",on=["date", "rgnid", "sex"]) #adds support numbers to the train_df
train_df.to_csv("trainf.csv", index=False)

test_df = test_df.merge(supportbycell, how="outer", on=["date","rgnid","sex"]) #Added 1655 rows, only 34 matches
test_df.to_csv("testf.csv", index=False)



