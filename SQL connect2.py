import pandas as pd
import sqlalchemy as sa
import datetime as dt
import pyodbc
import urllib
import quandl
import yfinance as yf
""" Program to load data from any file to SQL table"""
startdate = dt.date(2010,1,1)
enddate = dt.date.today()
split_df = pd.read_csv(r"C:\Users\admin\PycharmProjects\My Projects\MOTHERSUMI.NS1.csv")
# split_df = pd.read_csv(r"C:\Users\admin\PycharmProjects\My Projects\PyTrain\StockSplit_Data.csv")
# split_df1 = pd.read_csv(r"C:\Users\admin\PycharmProjects\My Projects\PyTrain\StockSplit_Data1.csv")
# split_data = split_df.append(split_df1,ignore_index=True).drop_duplicates(keep='first')
# stocks_nxt50 =pd.read_csv(r"C:\Users\admin\Downloads\ind_niftynext50list.csv")
# stocks_50 =pd.read_csv(r"C:\Users\admin\Downloads\ind_nifty50list.csv")
# stocks_mid50 =pd.read_csv(r"C:\Users\admin\Downloads\ind_niftymidcap50list.csv")
#prices=pd.read_csv('Nifty_Midcap_50.csv')
#prices=pd.read_csv('Nifty_Next_50.csv')


#Use this for windows authentication
params = urllib.parse.quote_plus("DRIVER={SQL Server Native Client 11.0};"
                                 "SERVER=DESKTOP-MAK81E6;"
                                 "DATABASE=PYTRAIN;"
                                 "Trusted_Connection=yes")

'''
#Use this for SQL server authentication
params = urllib.parse.quote_plus("DRIVER={SQL Server Native Client 11.0};"
                                 "SERVER=dagger;"
                                 "DATABASE=test;"
                                 "UID=user;"
                                 "PWD=password")
'''

#Connection String
engine = sa.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))

# Connect to the required SQL Server
conn=engine.connect()
stock= 'BAJAJ-AUTO'
get_stocks = yf.Ticker(stock +'.NS')
df_yf = get_stocks.history(start=startdate, end=enddate, interval="1d")
print(df_yf.head())
# split_df.to_sql('DATA_1M_10C',con=conn,index=False,if_exists='append',)
# split_data.to_sql('STOCKSPLIT',con=conn,index=False,if_exists='append')
# stocks_50.to_sql('NIFTY50',con=conn,index=False,if_exists='append')
# stocks_nxt50.to_sql('NIFTYNEXT50',con=conn,index=False,if_exists='append')
# stocks_mid50.to_sql('NIFTYMID50',con=conn,index=False,if_exists='append')

print('Task complete')
