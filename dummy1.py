import pandas as pd
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import sqlalchemy as sa
import datetime as dt
import pyodbc
import urllib
import os
import statsmodels.api as sm

#Use this for windows authentication
params = urllib.parse.quote_plus("DRIVER={SQL Server Native Client 11.0};"
                                 "SERVER=DESKTOP-MAK81E6\SQLEXPRESS;"
                                 "DATABASE=NSEDATA;"
                                 "Trusted_Connection=yes")


#Connection String
engine = sa.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))

# Connect to the required SQL Server
conn=engine.connect()

stocks = ['TATAMOTORS']

for stock in stocks:
    print(stock)
    query = "SELECT * FROM dbo." + stock + " WHERE DATE >='2020-01-01 00:00:00.000'"
    df = pd.read_sql(query, con=conn, parse_dates=True)
    df['Date'] = pd.to_datetime(df['Date'])
    dx = np.diff(range(0,len(df.index)))
    dy = np.diff(df['Close'])
    slope = dy/dx
    slope_rad = (np.rad2deg(np.arctan(slope)))
    df.loc[1:,'Slope'] = slope_rad

print(slope)

print(df['Slope'])