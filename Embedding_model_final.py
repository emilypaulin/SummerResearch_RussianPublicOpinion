#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:57:59 2022

@author: emilypaulin
"""

#This is the code used to build the neural network. 
import tensorflow as tf
import os
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
#This makes it so I don't see unnecessary messages: 
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

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
num_outputs = 1 
classifier = layers.Dense(num_outputs, activation = "sigmoid", name = "classifier")
classifier_output = classifier(pooled_output)
#bert_classifier = tfm.nlp.models.BertClassifier(network=bert_encoder, num_classes=2)

#Model
embedding_model = tf.keras.Model(text_input, classifier_output)

#How to show the model:
keras.utils.plot_model(embedding_model, "multi_input_and_output_model.png", show_shapes=True) 

#Categorical
embedding_model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[tf.keras.losses.MeanSquaredError(),
        #keras.losses.BinaryCrossentropy(from_logits=True), #changed this setting after failure
        #keras.losses.MeanAbsoluteError(), #commenting this one first, see what performs better
    ],
    loss_weights=[1.0, 0.2],
)
embedding_model.save("embedding_model")

#Binary
embedding_model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[#tf.keras.losses.MeanSquaredError(),
        keras.losses.BinaryCrossentropy(from_logits=False), #changed this setting after failure
        #keras.losses.MeanAbsoluteError(), #commenting this one first, see what performs better
    ],
    loss_weights=[1.0, 0.2],
)

embedding_model.save("embedding_model2")
