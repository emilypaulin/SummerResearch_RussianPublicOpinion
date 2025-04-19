#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 14:29:05 2022

@author: emilypaulin
"""

#This is the newest model, a categorical model with four outputs.
#The idea is based on one of the ways to tweak the model (I forget which one)

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="inputs")
preprocessor = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3", name = "preprocessor")
encoder_input = preprocessor(text_input)
encoder = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4", 
    trainable=True, name = "encoder")
encoder_outputs = encoder(encoder_input)
pooled_output = encoder_outputs["pooled_output"]      # [batch_size, 768].
sequence_output = encoder_outputs["sequence_output"]  # [batch_size, seq_length, 768]. will use pooled_output as an input for the next layer as it is better equipped for text classification tasks.
num_outputs = 4 
classifier = layers.Dense(num_outputs, activation = "sigmoid", name = "classifier") #Labels will be provided in a one-hot representation, and the model will output one-hot representations by default. 
classifier_output = classifier(pooled_output) 

#Create the model
categorical_model = tf.keras.Model(text_input, classifier_output)

#Categorical model
categorical_model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[tf.keras.losses.MeanSquaredError(from_logits=False), 
    ],
    loss_weights=[1.0, 0.2],
)
categorical_model.save("categorical_model")

#To use this model, 4-dimensional output tensors will be needed. Outputs of four are needed, with activations based on the level of support. Ex. the first dimension can be highly support etc. 
#Not sure how training specifically will work. Base it on survey data? Or on manually coded responses?