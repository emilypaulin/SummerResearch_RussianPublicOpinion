#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:48:29 2022

@author: emilypaulin
"""

#Filter data based on key words and use categorical classification
import sklearn
from sklearn.model_selection import train_test_split
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
conn = psycopg2.connect() #redacted login
cur = conn.cursor()
import boto3
s3 = boto3.resource('s3')
s3_client = boto3.client('s3')
bucket = s3.Bucket('sm-support-model')

# Binary classification model data formatting
rutracker = pd.DataFrame()
for chunk in pd.read_csv("/home/ec2-user/environment/rutracker.csv", chunksize=25000, encoding="utf-8"):
    rutracker = pd.concat([rutracker, chunk])

print(rutracker)

rutracker["ukraine_approval"] = rutracker["ukraine_approval"].astype(str).str.replace("1.0", "1")
rutracker["ukraine_approval"] = rutracker["ukraine_approval"].astype(str).str.replace("2.0", "1")
rutracker["ukraine_approval"] = rutracker["ukraine_approval"].astype(str).str.replace("3.0", "0")
rutracker["ukraine_approval"] = rutracker["ukraine_approval"].astype(str).str.replace("4.0", "0")
rutracker["ukraine_approval"] = rutracker["ukraine_approval"].astype(str).str.replace("5.0", "")
rutracker["ukraine_approval"] = rutracker["ukraine_approval"].astype(str).str.replace("nan", "")
rutracker["ukraine_approval"] = pd.to_numeric(rutracker["ukraine_approval"], errors="coerce")
rutracker["survey_date"] = pd.to_datetime(rutracker["survey_date"])

supportbycell = rutracker.groupby(["survey_date", "rgnid", "gender"])["ukraine_approval"].mean()
supportbycell = supportbycell.reset_index()
print(supportbycell)

df = pd.DataFrame(columns = ["date", "from_id", "text"])
for chunk in pd.read_csv("allposts.csv", chunksize = 25000, encoding="utf-8"):
    df = pd.concat([df, chunk])

del df["Unnamed: 0"]
#Filters out text that doesn't contain relevant words. 
df = df[df["text"].astype(str).str.contains("нацист|украин|путин|войн", flags=re.IGNORECASE, regex=True)]

#Taken from aggregating posts code
#
#   Aggregate posts into train and test dfs
#
users = pd.DataFrame(columns=["userid", "city", "sex"])
for chunk in pd.read_csv("users.csv", chunksize = 25000, encoding="utf-8"):
    users = pd.concat([users, chunk])

cities = pd.DataFrame(columns = ["cityname", "region_name", "rgnid"])
for chunk in pd.read_csv("cities.csv", chunksize = 25000, encoding="utf-8"):
    cities = pd.concat([cities, chunk])

del users["Unnamed: 0"]
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

joinedposts = userdata.merge(df, how="inner", left_on=["userid"], right_on=["from_id"])
del joinedposts["userid"]
del joinedposts["from_id"]
del joinedposts["region_name"]
joinedposts["date"] = pd.to_datetime(joinedposts["date"])
joinedposts["date"] = joinedposts["date"].dt.date

df2 = joinedposts.groupby(["date", "rgnid", "sex"], dropna=False)["text"].apply(lambda x: ','.join(x))
aggregatedposts = df2.reset_index()
aggregatedposts["date"] = pd.to_datetime(aggregatedposts["date"])
aggposts = aggregatedposts.merge(supportbycell, left_on=["date","rgnid","sex"], right_on=["survey_date","rgnid","gender"]) #This step only gives five samples, unusable
del aggposts["survey_date"]
del aggposts["gender"]

train_df, test_df = sklearn.model_selection.train_test_split(aggregatedposts, test_size=0.3, train_size=0.7, random_state=None, shuffle=True, stratify=None)
train_df.to_csv(path_or_buf="filteredtrain.csv")
test_df.to_csv(path_or_buf="filteredtrain.csv")

train_df = pd.DataFrame(columns=["text"]) #197
for chunk in pd.read_csv("smalltrain.csv", chunksize = 25000, encoding="utf-8"):
    train_df = pd.concat([train_df, chunk]) 
    
del train_df["Unnamed: 0"]

train = train_df
del train["date"]
del train["sex"]
del train["rgnid"]
del train["text"] #Leaving the number column with the support from the survey
support = pd.DataFrame.to_numpy(train)
support
train_df = pd.DataFrame(columns=["text"]) #197
for chunk in pd.read_csv("smalltrain.csv", chunksize = 25000, encoding="utf-8"):
    train_df = pd.concat([train_df, chunk]) 

train = train_df
del train["date"]
del train["sex"]
del train["rgnid"]
del train["ukraine_approval"] #Leaving the text column
train_arr = pd.DataFrame.to_numpy(train)
train_arr

embedding_model = keras.models.load_model("/home/ec2-user/environment/embedding_model")
del train_df
batch_size = 200

embedding_model.fit(
    x=train_arr,
    y=support,
    batch_size=batch_size,
    epochs=2,
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

embedding_model.save("embedding_model")

#Test
test_df = pd.DataFrame(columns=["text"])
for chunk in pd.read_csv("testf.csv", chunksize = 25000, encoding="utf-8"):
    test_df = pd.concat([test_df, chunk])
    
del test_df["Unnamed: 0"]

test = test_df[:32170] #32171 total; check this before making into arrays first time
del test["date"]
del test["sex"]
del test["rgnid"]
del test["number"]
test_arr = pd.DataFrame.to_numpy(test)
print(test_arr)
test = test_df[:32170]
del test["date"]
del test["sex"]
del test["rgnid"]
del test["text"]
support_arr = pd.DataFrame.to_numpy(test)
print(support_arr)

embedding_model = keras.models.load_model("/home/ec2-user/environment/embedding_model")
batch_size = 200
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

#Predicting
aggposts = pd.DataFrame(columns=["text"])
for chunk in pd.read_csv("/home/ec2-user/environment/aggregatedtext.csv", chunksize = 25000, encoding="utf-8"):
    aggposts = pd.concat([aggposts, chunk])

del aggposts["date"]
del aggposts["sex"]
del aggposts["rgnid"]

agg_arr = pd.DataFrame.to_numpy(aggposts)
embedding_model = keras.models.load_model("/home/ec2-user/environment/embedding_model")
batch_size = 200

predictions = embedding_model.predict(
    agg_arr,
    batch_size=batch_size
    )
