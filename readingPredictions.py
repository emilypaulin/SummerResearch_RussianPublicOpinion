#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 14:11:46 2022

@author: emilypaulin
"""

#This is code for reading the document of predictions. 
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

#For reading out predictions

predictions = pd.DataFrame(columns=[])
for chunk in pd.read_csv("followbpredictions.csv", chunksize = 25000, encoding="utf-8"):
    predictions = pd.concat([predictions, chunk]) 
    
del predictions["Unnamed: 0"]

#Puts posts in with the predictions and support by cell. If not prediction on aggregated posts, use the other data instead.  
aggposts = pd.DataFrame(columns=["text"])
for chunk in pd.read_csv("/home/ec2-user/environment/aggregatedtext.csv", chunksize = 25000, encoding="utf-8"):
    aggposts = pd.concat([aggposts, chunk])

del aggposts["Unnamed: 0"]
aggposts["date"] = pd.to_datetime(aggposts["date"])
aggposts["psupport"] = predictions["0"] 
aggposts["sex"] = aggposts["sex"].astype(str).str.replace("1.0","female")
aggposts["sex"] = aggposts["sex"].astype(str).str.replace("2.0","male")
del predictions

#Again, puts posts with support by cell so it can be compared. Adjust as needed based on inputs. 
supportbycell = pd.DataFrame(columns=["date","rgnid","sex","number"])
for chunk in pd.read_csv("supportbycell.csv", chunksize = 25000, encoding="utf-8"):
    supportbycell = pd.concat([supportbycell, chunk])
    
supportbycell["date"] = pd.to_datetime(supportbycell["date"])
supportbycell["rgnid"] = pd.to_numeric(supportbycell["rgnid"])
supportbycell["support"] = pd.to_numeric(supportbycell["number"])


print(aggposts)
comparesupport = aggposts.merge(supportbycell, how="outer",on=["date", "rgnid", "sex"])
#Adds 155 rows
