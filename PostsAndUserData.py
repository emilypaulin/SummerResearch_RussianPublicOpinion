#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 11:49:55 2022

@author: emilypaulin
"""

#import necessary packages. 
import psycopg2

#Connect to the database.
conn = psycopg2.connect() #redacted user information

#Open a cursor to perform database operations.
cur = conn.cursor()

#Execute a query
cur.execute("SELECT users.id, cities.rgnid, users.sex, start_page_posts.post_text FROM users INNER JOIN ruregions.cities ON users.city = cities.cityname INNER JOIN start_page_posts ON users.id = start_page_posts.from_id")

#Retrieve query results. They don't show unless printed.
records = cur.fetchall()
print(records)

#Close connection.
cur.close()
conn.close()