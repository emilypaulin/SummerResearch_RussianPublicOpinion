# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import necessary packages. 
import psycopg2

#Connect to the database.
conn = psycopg2.connect() #redacted login

#Open a cursor to perform database operations.
cur = conn.cursor()

#Execute a query
cur.execute("select * from ruregions.cities")

#Retrieve query results. They don't show unless printed.
records = cur.fetchall()

#Close connection.
cur.close()
conn.close()