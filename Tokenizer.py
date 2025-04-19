#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 15:29:31 2022

@author: emilypaulin
"""

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
#This makes it so I don't see unnecessary messages: 
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

#                               #
#                               #
#       Data preparation        #
#                               #
#                               #
conn = psycopg2.connect() #redacted login
cur = conn.cursor()
cur.execute("SELECT post_text FROM public.start_page_posts limit 20000")
post_text = cur.fetchall()
cur.execute("SELECT link_title FROM public.start_page_posts limit 20000")
link_title = cur.fetchall()
cur.execute("SELECT link_description FROM public.start_page_posts limit 20000")
link_description = cur.fetchall()
cur.execute("SELECT date FROM public.start_page_posts limit 20000")
date = cur.fetchall()

#Original dataframe
d = {
     "date":date,
     "post_text":post_text,
     "link_title":link_title,
     "link_description":link_description
     }
df = pd.DataFrame(d, columns=["date","post_text", "link_title", "link_description"])
print(df)

df.to_csv(path_or_buf = "posts.csv")
df2 = pd.DataFrame(columns=["date", "post_text", "link_title", "link_description"])
for chunk in pd.read_csv("posts.csv", chunksize = 1000, encoding="utf-8"):
    df2 = pd.concat([df2, chunk])
    
    
print(df2)
del df2["Unnamed: 0"]
df2["post_text"] = df2["post_text"].astype(str).str.replace("\(", "")
df2["post_text"] = df2["post_text"].astype(str).str.replace(",\)","")
df2["link_title"] = df2["link_title"].astype(str).str.replace("\(", "")
df2["link_title"] = df2["link_title"].astype(str).str.replace(",\)","")
df2["link_description"] = df2["link_description"].astype(str).str.replace("\(", "")
df2["link_description"] = df2["link_description"].astype(str).str.replace(",\)","")
df2["date"] = df2["date"].astype(str).str.replace("\(", "")
df2["date"] = df2["date"].astype(str).str.replace(",\)","")
#Converts datetime to date
df2["date"] = pd.to_numeric(df2["date"])
df2["date"] = pd.to_datetime(df2["date"], unit = "s")
print(df2)

#Removes empty data and concatenates columns together
df = pd.DataFrame(columns = ["date", "text"])
df["date"] = df2["date"]
df["text"] = df2["post_text"] + ", " + df2["link_title"] + ", " + df2["link_description"]
df = df[df["text"] != "\'\', \'\', \'\'"]
df["text"] = df["text"].astype(str).str.replace("\'\',","")
df["text"] = df["text"].astype(str).str.replace("\'\'","")
df["text"] = df["text"].astype(str).str.replace("\', ","\'")
del df["date"]
print(df)

#Data into one large array and two subset arrays for training and testing
array = pd.DataFrame.to_numpy(df)
print(array)
train_arr = array[1:6000]
test_arr = array[2500:5652]

#Create empty np array for results to go into
support = np.zeros((train_arr.shape), np.int0)
print(support)


#                               #
#                               #
#       Build neural network    #
#                               #
#                               #

# Load the model
embedding_model = keras.models.load_model("embedding_model")
batch_size = 2

embedding_model.fit(
    x=train_arr,
    y=support,
    batch_size=batch_size,
    epochs=1,
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

embedding_model.evaluate(
    x=train_arr,
    y=support,
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

embedding_model.predict(
    array,
    batch_size=batch_size,
    verbose="auto",
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
)
