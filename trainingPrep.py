#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:19:26 2022

@author: emilypaulin
"""
#This document is a framework for training.

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

'''import boto3
s3 = boto3.resource('s3')
bucket = s3.Bucket('sm-support-model')
s3_client = boto3.client('s3')
conn = psycopg2.connect() #redacted login
cur = conn.cursor()'''

train_df = pd.DataFrame(columns=["text"])
for chunk in pd.read_csv("trainf.csv", chunksize = 25000, encoding="utf-8"):
    train_df = pd.concat([train_df, chunk]) 
#76630 (-155 rows with no survey data)=76475
    
del train_df["Unnamed: 0"] 
train = train_df[38001:75000]
del train["date"]
del train["sex"]
del train["rgnid"]
del train["text"] #Leaving the number column with the support from the survey
support = pd.DataFrame.to_numpy(train)
print(support)
train = train_df[38001:75000]
del train["date"]
del train["sex"]
del train["rgnid"]
del train["number"] #Leaving the text column
train_arr = pd.DataFrame.to_numpy(train)
print(train_arr)


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

#OR
#agg = aggposts.iloc[:10000]
#agg_arr = pd.DataFrame.to_numpy(agg)
#embedding_model.predict_on_batch(agg)

predictions = embedding_model.predict(
    agg_arr,
    batch_size=batch_size
    )
