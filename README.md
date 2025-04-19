# SummerResearch_RussianPublicOpinion

This repository contains the Python code I utilized to build a neural network to analyze Russian social media data. It was all written in the summer after my freshman year at Princeton and reflects the tinkering of an amateur with a relatively new technology (I had never worked with Python before but was an enthusiastic learner). 

This project's goal was to understand Russian public opinion of the war in Ukraine. This project implemented the model Bidirectional Encoder Representations from Transformers (BERT), an early language model made by Google. I also utilized SQL for database management for this project. 

The files for cities.py, cities.py, joincitiesusers.py, pandasAvgofSupport.py, allDataAvgofSupport.py, and PostsAndUserData.py are utilized to process data from the database into chunks that can be analyzed by groups. 

Embedding_model_final.py builds the language model, and Tokenizer.py processes the data so that it can be read by the model.

trainingPrep.py is a framework for training. ukraineCloselyFollowPrep.py and ukraineEmotionsHope.py process social media data and build predictions of public opinion based on the training data. filteredMulti.py filtered data by keywords and then made categorical classifications. In filteredBinary.py, I attempted to filter for negative Russian terms and then make binary predictions based on those posts. 
