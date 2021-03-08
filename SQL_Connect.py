import pandas as pd
import sqlalchemy as sa
import datetime as dt
import pyodbc
import urllib
import quandl
import yfinance as yf



startdate = dt.date(2010,1,1)
enddate = dt.date.today()

split_df = pd.read_csv(r"C:\Users\admin\PycharmProjects\My Projects\PyTrain\StockSplit_Data1.csv")
prices=pd.read_csv('Stock_Symbol.csv')
#prices=pd.read_csv('Nifty_Midcap_50.csv')
#prices=pd.read_csv('Nifty_Next_50.csv')
stocks = ['EICHERMOT']

split_count_dict = dict(split_df['Stock'].value_counts())
split_stocks_list = split_df['Stock'].unique().tolist()

def adj_close(data,stock):
    df_adj = data.copy()
    no_of_splits = split_count_dict[stock]
    stock_split_info = split_df[split_df['Stock']==stock].copy()
    stock_split_info['Date'] = pd.to_datetime(stock_split_info['Date'])
    for i in range(no_of_splits):
        try:
            to_be_split = (df_adj['Date']<=stock_split_info.iloc[i,0])
        except:
            continue
        df_adj.loc[to_be_split,['Open','High','Low','Close']] = round(df_adj.loc[to_be_split,['Open','High','Low','Close']]/stock_split_info.iloc[i,1],2)
    return df_adj

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

for stock in stocks:
    query = 'SELECT max(DATE) FROM dbo.' + stock
    start_dt = conn.execute(query)
    print(start_dt)
    print("Extracting Data from Quandl for the stock : {}" .format(stock))
    data = quandl.get('NSE/'+stock,start_date=startdate,end_date=enddate,api_key='Rqhpodd29Sx9ixa2G6QQ')
    data.columns = ['Open','High','Low','Last','Close','Volume','Turnover']
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'],format='%Y-%m-%d')
    stock_data = data[['Date','Open','High','Low','Close','Volume']].copy()
    if stock in split_stocks_list:
        stock_data = adj_close(stock_data,stock)

    print("Extracting Data from YF for the stock : {}".format(stock))
    startdate = stock_data['Date'].max() + dt.timedelta(1)
    print(startdate)
    get_stocks = yf.Ticker(stock +'.NS')
    df_yf = get_stocks.history(start=startdate, end=enddate, interval="1d")
    df_yf.reset_index(inplace=True)
    df_yf['Date'] = pd.to_datetime(df_yf['Date'],format='%Y-%m-%d')
    df_yf = df_yf[['Date','Open','High','Low','Close','Volume']]

    stock_data = stock_data.append(df_yf)
    #Read data from .csv file
    # df=pd.read_csv('Results2.csv')
    # df.fillna('',inplace=True)
    print("Start Data Load for the stock : {}".format(stock))
    #Write data read from .csv to SQL Server table
    stock_data.to_sql(name=stock,con=conn,if_exists='replace',index=False)
    print("Data Load done for the stock : {}".format(stock))
#Read data from a SQL server table
# df1=pd.read_sql_table('STOCK_DATA',con=conn)
# print(df1.head())

'''
# Condition to not insert duplicate values into sql server table

for i in range(len(df)):
    try:
        df.iloc[i:i+1].to_sql(name="Table_Name",if_exists='append',con = Engine)
    except IntegrityError:
        pass #or any other action

'''