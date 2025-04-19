#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 10:58:59 2022

@author: emilypaulin
"""
#This code was one of the earlier iterations of code used to prepare data. It's a framework.

import sklearn
import pandas as pd
import numpy as np
import psycopg2
import tensorflow as tf
import os
import tensorflow_hub as hub
from tensorflow import keras
import tensorflow_text as text
from datetime import datetime
os.environ['AUTOGRAPH_VERBOSITY'] = '0'


#                               #
#                               #
#       Data preparation        #
#                               #
#                               #
conn = psycopg2.connect() #redacted login
cur = conn.cursor()
cur.execute("SELECT post_text FROM public.start_page_posts")
post_text = cur.fetchall()
cur.execute("SELECT link_title FROM public.start_page_posts")
link_title = cur.fetchall()
cur.execute("SELECT link_description FROM public.start_page_posts")
link_description = cur.fetchall()
cur.execute("SELECT date FROM public.start_page_posts")
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
for chunk in pd.read_csv("posts.csv", chunksize = 25000, encoding="utf-8"):
    df2 = pd.concat([df2, chunk])

#or df2 = df
    
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
df2.to_csv(path_or_buf = "partiallycleanedposts.csv")

df = pd.DataFrame(columns=["date", "post_text", "link_title", "link_description"])
for chunk in pd.read_csv("partiallycleanedposts.csv", chunksize = 25000, encoding="utf-8"):
    df = pd.concat([df, chunk])
del df["Unnamed: 0"]

df2 = df
#Removes empty data and concatenates columns together
df = pd.DataFrame(columns = ["date", "text"])
df["date"] = df2["date"]
df["text"] = df2["post_text"] + ", " + df2["link_title"] + ", " + df2["link_description"]
df = df[df["text"] != "\'\', \'\', \'\'"]
df["text"] = df["text"].astype(str).str.replace("\'\',","")
df["text"] = df["text"].astype(str).str.replace("\'\'","")
df["text"] = df["text"].astype(str).str.replace("\', ","\'")
del df["date"]
df.to_csv(path_or_buf = "cleanedposts.csv")
print(df)

#Load this dataframe for full data
df = pd.DataFrame(columns=["text"])
for chunk in pd.read_csv("cleanedposts.csv", chunksize = 25000, encoding="utf-8"):
    df = pd.concat([df, chunk])
del df["Unnamed: 0"]
print(df)

#Split data
train_df, test_df = sklearn.model_selection.train_test_split(df, test_size=0.3, train_size=0.7, random_state=None, shuffle=True, stratify=None)
train_df.to_csv(path_or_buf = "train")
test_df.to_csv(path_or_buf = "test")

#Loops to load data from dataframes
train_df = pd.DataFrame(columns=["text"])
for chunk in pd.read_csv("train.csv", chunksize = 25000, encoding="utf-8"):
    train_df = pd.concat([train_df, chunk])
    
test_df = pd.DataFrame(columns=["text"])
for chunk in pd.read_csv("test.csv", chunksize = 25000, encoding="utf-8"):
    test_df = pd.concat([df, chunk])

#Create empty np array for results to go into
train_arr = pd.DataFrame.to_numpy(train_df)
print(train_arr)
test_arr = pd.DataFrame.to_numpy(test_df)
print(test_arr)
support = np.zeros((train_arr.shape), np.int0)
print(support)
