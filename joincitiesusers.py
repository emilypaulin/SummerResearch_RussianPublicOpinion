#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 09:31:25 2022

@author: emilypaulin
"""

#import necessary packages. 
import psycopg2

#Connect to the database.
conn = psycopg2.connect() #redacted login

#Open a cursor to perform database operations.
cur = conn.cursor()

#Execute a query
cur.execute("SELECT users.id, users.source_id, users.city, users.city_id, users.sex, users.first_name, users.last_name, cities.rgnid FROM users INNER JOIN ruregions.cities ON users.city = cities.cityname")
records = cur.fetchall()

print(records)

#Close connection.
cur.close()
conn.close()