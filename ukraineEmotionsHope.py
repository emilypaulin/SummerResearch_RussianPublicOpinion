#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:45:40 2022

@author: emilypaulin
"""

#This is training based on the ukr_emotions_hope data point (11219 responses to this particular question)
#This uses the binary model. 

import sklearn
import pandas as pd
import numpy as np
import psycopg2
import tensorflow as tf
import os
import tensorflow_hub as hub
from tensorflow import keras
import tensorflow_text as text
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from datetime import datetime
import re
conn = psycopg2.connect() #login redacted
cur = conn.cursor()
import boto3
s3 = boto3.resource('s3')
s3_client = boto3.client('s3')
bucket = s3.Bucket('sm-support-model')

rutracker = pd.DataFrame()
for chunk in pd.read_csv("/home/ec2-user/environment/rutracker.csv", chunksize=25000, encoding="utf-8"):
    rutracker = pd.concat([rutracker, chunk])

print(rutracker)

rutracker["ukr_emotions_hope"] = rutracker["ukr_emotions_hope"].astype(str).str.replace("1.0", "0")
rutracker["ukr_emotions_hope"] = rutracker["ukr_emotions_hope"].astype(str).str.replace("2.0", "0")
rutracker["ukr_emotions_hope"] = rutracker["ukr_emotions_hope"].astype(str).str.replace("3.0", "1") #Check with Jacob what number mean
rutracker["ukr_emotions_hope"] = rutracker["ukr_emotions_hope"].astype(str).str.replace("4.0", "1")
rutracker["ukr_emotions_hope"] = rutracker["ukr_emotions_hope"].astype(str).str.replace("5.0", "1")
rutracker["ukr_emotions_hope"] = rutracker["ukr_emotions_hope"].astype(str).str.replace("6.0", "1")
rutracker["ukr_emotions_hope"] = rutracker["ukr_emotions_hope"].astype(str).str.replace("nan", "")
rutracker["ukr_emotions_hope"] = pd.to_numeric(rutracker["ukr_emotions_hope"], errors="coerce")
rutracker["survey_date"] = pd.to_datetime(rutracker["survey_date"])

support = pd.DataFrame(columns = ["survey_date","rgnid","gender","ukr_emotions_hope"])
support["survey_date"] = rutracker["survey_date"]
support["rgnid"] = rutracker["rgnid"]
support["gender"] = rutracker["gender"]
support["ukr_emotions_hope"] = pd.to_numeric(rutracker["ukr_emotions_hope"])
support #5091 cells

supportbycell = rutracker.groupby(["survey_date", "rgnid", "gender"])["ukr_emotions_hope"].mean()
supportbycell = supportbycell.reset_index()
print(supportbycell)
#Can use same code as other ukraine_support from now on

users = pd.DataFrame(columns=["userid", "city", "sex"])
for chunk in pd.read_csv("users.csv", chunksize = 25000, encoding="utf-8"):
    users = pd.concat([users, chunk])

del users["Unnamed: 0"]

cities = pd.DataFrame(columns = ["cityname", "region_name", "rgnid"])
for chunk in pd.read_csv("cities.csv", chunksize = 25000, encoding="utf-8"):
    cities = pd.concat([cities, chunk])

del cities["Unnamed: 0"]
userdata = users.merge(cities, how="left", left_on=["city"], right_on=["cityname"])
del users
del cities
userdata["sex"] = userdata["sex"].astype(str).str.replace("1","female")
userdata["sex"] = userdata["sex"].astype(str).str.replace("2","male")
userdata["rgnid"] = pd.to_numeric(userdata["rgnid"])
userdata["userid"] = pd.to_numeric(userdata["userid"])
del userdata["city"]
del userdata["cityname"]

'''cur.execute("SELECT date FROM public.start_page_posts")
date = cur.fetchall()
cur.execute("SELECT from_id FROM public.start_page_posts")
from_id = cur.fetchall()
cur.execute("SELECT post_text, link_title, link_description FROM public.start_page_posts")
txt = cur.fetchall()
d = {
     "date":date,
     "from_id":from_id,
     "text":txt
     }
df = pd.DataFrame(d, columns=["date", "from_id", "text"])
print(df)
df["date"] = df["date"].astype(str).str.replace("\(", "")
df["date"] = df["date"].astype(str).str.replace(",\)","")
df["date"] = pd.to_numeric(df["date"])
df["date"] = pd.to_datetime(df["date"], unit = "s")
df["from_id"] = df["from_id"].astype(str).str.replace("\(", "")
df["from_id"] = df["from_id"].astype(str).str.replace(",\)","")
df["from_id"] = pd.to_numeric(df["from_id"])
df["text"] = df["text"].astype(str).str.replace("\(\'", "")
df["text"] = df["text"].astype(str).str.replace("\', \'"," ")
df["text"] = df["text"].astype(str).str.replace("\'\)","")
'''
df = pd.DataFrame(columns = ["date", "from_id", "txt"])
for chunk in pd.read_csv("allposts.csv", chunksize = 25000, encoding="utf-8"):
    df = pd.concat([df, chunk])
    
del df["Unnamed: 0"]

joinedposts = userdata.merge(df, how="inner", left_on=["userid"], right_on=["from_id"])
del joinedposts["userid"]
del joinedposts["from_id"]
del joinedposts["region_name"]
joinedposts["date"] = pd.to_datetime(joinedposts["date"])
joinedposts["date"] = joinedposts["date"].dt.date #1552610 total
 
df2 = joinedposts.groupby(["date", "rgnid", "sex"])["text"].apply(lambda x: ','.join(x))
aggregatedposts = df2.reset_index()
aggregatedposts["date"] = pd.to_datetime(aggregatedposts["date"]) #107234 total cells

aggposts = aggregatedposts.merge(support, left_on=["date","rgnid","sex"], right_on=["survey_date","rgnid","gender"]) #1324
#deletes extra columns added by merge
del aggposts["survey_date"]
del aggposts["gender"] 

aggposts["text"] = aggposts["text"].replace("  ,  ", np.nan)
aggposts.dropna(subset=["text", "ukr_emotions_hope"], inplace=True) #1083 now

train_df, test_df = sklearn.model_selection.train_test_split(aggposts, test_size=0.3, train_size=0.7, random_state=None, shuffle=True, stratify=None)
train_df.to_csv(path_or_buf="hopetrain.csv") #758
test_df.to_csv(path_or_buf="hopetest.csv") #32

#Add csvs to bucket and run on EC2
train_df = pd.DataFrame(columns=["text"])
for chunk in pd.read_csv("/home/ec2-user/hopetrain.csv", chunksize = 25000, encoding="utf-8"): #change depedning on system
    train_df = pd.concat([train_df, chunk])
    
del train_df["Unnamed: 0"]
train_df.dropna(subset=["text", "ukr_emotions_hope"], inplace=True)

train = train_df
del train["date"]
del train["sex"]
del train["rgnid"]
del train["text"] #Leaving the number column with the support from the survey
support_arr = pd.DataFrame.to_numpy(train)
support_arr

train_df = pd.DataFrame(columns=["text"])
for chunk in pd.read_csv("/home/ec2-user/hopetrain.csv", chunksize = 25000, encoding="utf-8"):
    train_df = pd.concat([train_df, chunk])
    
del train_df["Unnamed: 0"]
train_df.dropna(subset=["text", "ukr_emotions_hope"], inplace=True)
train = train_df
del train["date"]
del train["sex"]
del train["rgnid"]
del train["ukr_emotions_hope"] #Leaving the text column
train_arr = pd.DataFrame.to_numpy(train)
train_arr

embedding_model = keras.models.load_model("/home/ec2-user/embedding_model2")
batch_size = 32

embedding_model.fit(
    x=train_arr,
    y=support_arr,
    batch_size=batch_size,
    epochs=4,
    verbose="auto",
    callbacks=None,
    validation_split=0.0,
    validation_data=None,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
)
embedding_model.save("embedding_model2") #BE VERY CAREFUL ABOUT WHICH ONE SAVING: binary is 2, normal is blank
    
    
test_df = pd.DataFrame(columns=["text"])
for chunk in pd.read_csv("/home/ec2-user/hopetest.csv", chunksize = 25000, encoding="utf-8"):
    test_df = pd.concat([test_df, chunk])
    
del test_df["Unnamed: 0"]    
test_df.dropna(subset=["text", "ukr_emotions_hope"], inplace=True)    
test = test_df #85
del test["date"]
del test["sex"]
del test["rgnid"]
del test["ukr_emotions_hope"]
test_arr = pd.DataFrame.to_numpy(test)
print(test_arr)

test_df = pd.DataFrame(columns=["text"])
for chunk in pd.read_csv("/home/ec2-user/hopetest.csv", chunksize = 25000, encoding="utf-8"):
    test_df = pd.concat([test_df, chunk])
    
test_df.dropna(subset=["text", "ukr_emotions_hope"], inplace=True)    
del test_df["Unnamed: 0"]
test=test_df
del test["date"]
del test["sex"]
del test["rgnid"]
del test["text"]
support_arr = pd.DataFrame.to_numpy(test)
print(support_arr)

embedding_model = keras.models.load_model("/home/ec2-user/embedding_model2")
batch_size = 32
del test_df
embedding_model.evaluate(
    x=test_arr,
    y=support_arr,
    batch_size=batch_size,
    verbose="auto",
    sample_weight=None,
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
    return_dict=False
)

aggposts = pd.DataFrame(columns=["text"])
for chunk in pd.read_csv("/home/ec2-user/environment/aggregatedtext.csv", chunksize = 25000, encoding="utf-8"):
    aggposts = pd.concat([aggposts, chunk])
    
aggposts=aggposts[:10000]

del aggposts["date"]
del aggposts["sex"]
del aggposts["rgnid"]

agg_arr = pd.DataFrame.to_numpy(aggposts)
embedding_model = keras.models.load_model("/home/ec2-user/embedding_model2")
batch_size = 200

predictions = embedding_model.predict(
    agg_arr,
    batch_size=batch_size
    )
pd.DataFrame(predictions).to_csv(path_or_buf="hopepredictions.csv") 

#
# Actually binary is just making the model binary and running the same code as above
#embedding_model2 though
