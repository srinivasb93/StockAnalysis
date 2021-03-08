'''
Created on May 20, 2020
Individual Stock Analysis where there is one entry and multiple exits for entire traded qty
@author: Srinivasulu_B
'''
import pandas as pd
import numpy as np
from math import floor
import datetime as dt
import matplotlib.pyplot as plt
import yfinance as yf
# import mplfinance as mpf
from nsepy import get_history
import os
import statsmodels.api as sm


pd.options.mode.chained_assignment = None
plt.style.use('ggplot')
"""
period: data period to download (Either Use period parameter or use start and end) Valid periods are: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
interval: data interval (intraday data cannot extend last 60 days) Valid intervals are: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
start: If not using period - Download start date string (YYYY-MM-DD) or datetime.
end: If not using period - Download end date string (YYYY-MM-DD) or datetime.
prepost: Include Pre and Post market data in results? (Default is False)
auto_adjust: Adjust all OHLC automatically? (Default is True)
actions: Download stock dividends and stock splits events? (Default is True)
"""

prices=pd.read_csv('Stock_Symbol.csv')
#prices=pd.read_csv('Nifty_Midcap_50.csv')
#prices=pd.read_csv('Nifty_Next_50.csv')
# stocks = prices['Symbol'].tolist()
stocks = ['BRITANNIA']
#stocks = input("Enter the Stock symbol as in NSE : ")
ndays = input("Enter the no. of days for Data Analysis : ")
fast = int(input("Enter the no.of days for Fast MA : "))
slow = int(input("Enter the no.of days for Slow MA : "))
start_date = dt.date(2015,1,1)
end_date = dt.date.today()- dt.timedelta(1)

split_df = pd.read_csv('StockSplit_Data.csv')
split_count_dict = dict(split_df['Stock'].value_counts())
split_stocks_list = split_df['Stock'].unique().tolist()

df4 = pd.DataFrame()
df5 = pd.DataFrame()
df6 = pd.DataFrame()
df7 = pd.DataFrame()
df8 = pd.DataFrame()

def wwma(values, n):
    return values.ewm(alpha=1/n, adjust=False).mean()

# def atr(df, n=14):
#     data = df.copy()
#     high = data['High']
#     low = data['Low']
#     close = data['Close']
#     data['tr0'] = abs(high - low)
#     data['tr1'] = abs(high - close.shift())
#     data['tr2'] = abs(low - close.shift())
#     tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
#     #atr = wwma(tr, n)
#     atr = tr.rolling(n).mean()
#     return atr

def atr(df, n=14):
    data = df.copy()
    high = data['High']
    low = data['Low']
    close = data['Close']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = wwma(tr, n)
    #atr = tr.rolling(n).mean()
    atr_close = close - 2.5*atr
    atr_close = atr_close.shift()
    data['ATR_U'] = atr_close
    atr_diff = atr_close - atr_close.shift()
    atr_close_down = close + 2.5*atr
    atr_close_down = atr_close_down.shift()
    data['ATR_D'] = atr_close_down
    atr_diff_down = atr_close_down - atr_close_down.shift()
    sl_value = 0
    for i in range(1,len(data.index)):
        if data.loc[i,'Close']< sl_value and data.loc[i-1,'Close']> sl_value:
            sl_value = atr_close_down[i]
            data.loc[i,'TSL'] = sl_value
        elif data.loc[i,'Close']> sl_value and data.loc[i-1,'Close']< sl_value:
            sl_value = atr_close[i]
            data.loc[i,'TSL'] = sl_value
        elif data.loc[i,'Close']> sl_value:
            if atr_diff[i]>0 and atr_close[i]>sl_value :
                sl_value = atr_close[i]
                data.loc[i,'TSL'] = sl_value 
            else:
                data.loc[i,'TSL'] = sl_value
        else:
            if atr_diff_down[i]<0 and atr_close_down[i]<sl_value:
                sl_value = atr_close_down[i]
                data.loc[i,'TSL'] = sl_value 
            else:
                data.loc[i,'TSL'] = sl_value
    return data[['ATR_U','ATR_D','TSL']]

def slope(ser,n):
    "function to calculate the slope of regression line for n consecutive points on a plot"
#     ser = (ser - ser.min())/(ser.max() - ser.min())
    x = np.array(range(len(ser)))
#     x = (x - x.min())/(x.max() - x.min())
    slopes = [i*0 for i in range(n-1)]
    reg_prices = [i*0 for i in range(n-1)]
    for i in range(n,len(ser)+1):
        y_scaled = ser[i-n:i]
        x_scaled = x[i-n:i]
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled,x_scaled)
        results = model.fit()
        results1 = model.predict(results.params)
        slopes.append(results.params[-1])
        reg_prices.append(results1[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return slope_angle,reg_prices

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

strategy = 'Reg_Cross4'
os.mkdir(os.path.join(os.getcwd(),strategy))
new_path = os.path.join(os.getcwd(),strategy)

for stock in stocks:
    print(stock)
    #get_stocks = yf.Ticker(stock)
    #df = get_stocks.history(period=ndays, interval="1d")
    df = get_history(symbol=stock, start=start_date, end=end_date)
    #df = get_stocks.history(start=start_date,end=end_date, interval="1d")
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    if stock in split_stocks_list:
        df = adj_close(df,stock)
    df['Close_change'] = df['Close'].pct_change()
    act_ndays = len(df.index)
    
    #high_52_week = df['High'].max()
    #low_52_week = df['Low'].min()
    #latest_close = df['Close'].tail(1).values[0]
    #chg_52_high = ((latest_close - high_52_week) / high_52_week) * 100
    #chg_52_low = ((latest_close - low_52_week) / low_52_week) * 100
    #dfmin = df['Close'].min()
    #dfmax = df['Close'].max()
    #dfmean = round(df['Close'].mean(), 1)

    """ Calculate fast and slow EMA """
    dfmean_fast_C = np.round(df['Close'].ewm(span=fast).mean(), 2)
    dfmean_fast_H = np.round(df['High'].ewm(span=fast).mean(), 2)
    dfmean_fast_L = np.round(df['Low'].ewm(span=fast).mean(), 2)
    dfmean_slow_C = np.round(df['Close'].ewm(span=slow).mean(), 2)
    dfmean_slow_H = np.round(df['High'].ewm(span=slow).mean(), 2)
    dfmean_slow_L = np.round(df['Low'].ewm(span=slow).mean(), 2)
    """ Calculate 11 and 50 EMA , 20 SMA"""
    EMA_3 = np.round(df['Close'].ewm(span=3).mean(), 2)
    EMA_5 = np.round(df['Close'].ewm(span=5).mean(), 2)
    EMA_50 = np.round(df['Close'].ewm(span=50).mean(), 2)
    EMA_11 = np.round(df['Close'].ewm(span=11).mean(), 2)
    SMA_20 = np.round(df['Close'].rolling(20).mean(), 2)
    EMA_20 = np.round(df['Close'].ewm(span=20).mean(), 2)
    """ Calculate Bollinger Bands for 20 day SMA """ 
    BB_Std=df.loc[:,'Close'].rolling(20).std()
    BB2_Upper = SMA_20+(BB_Std*2)
    BB1_Upper = SMA_20+(BB_Std*1)
    BB2_Lower = SMA_20-(BB_Std*2)
    BB1_Lower = SMA_20-(BB_Std*1)
    """Calculate Range & 10 days Average Range """
    df['Range'] = df['High'] - df['Low']
    #df['Range_chg'] = df['Range'].pct_change()
    df['Avg_Range'] = round(df['Range'].rolling(10).mean(), 1)
    df['Close_Low'] = df['Close'] - df['Low']
    df['High_Close'] = df['High'] - df['Close']
    """ Calculate MACD Standard 11,26,9 """
    MACD_mean_fast = np.round(df['Close'].ewm(span=11).mean(), 2) 
    MACD_mean_slow = np.round(df['Close'].ewm(span=26).mean(), 2)
    MACD = MACD_mean_fast - MACD_mean_slow
    MACD_Signal = round(MACD.ewm(span=9).mean(), 2)
    MACD_hist = MACD - MACD_Signal
    """Calculating Support and Resistance zones"""
    s1_1 = df.loc[int(act_ndays*2/3):(act_ndays-10),'Low'].min()
    s1_2 = df.loc[int(act_ndays*2/3):(act_ndays-10),'Close'].min()
    r1_1 = df.loc[int(act_ndays*2/3):(act_ndays-10),'Close'].max()
    r1_2 = df.loc[int(act_ndays*2/3):(act_ndays-10),'High'].max()
    
    """Creating a dictionary with the required data"""
    data1 = {
        'Symbol':stock,
        'Date': df['Date'],
        'Open': df['Open'],
        'High': df['High'],
        'Low': df['Low'],
        'Close': df['Close'],
        'Close_Chg': round(df['Close_change'] * 100, 1),
        'EMA_L_C': dfmean_fast_C - dfmean_fast_L,
        'EMA_H_C': dfmean_fast_H - dfmean_fast_C,
        'EMA_H_L': '',
        '3EMA_H_L': '',
        'Slow_EMA_L_C': dfmean_slow_C - dfmean_slow_L,
        'Slow_EMA_H_C': dfmean_slow_H - dfmean_slow_C,
        'Slow_EMA_H_L': '',
        'Slow_3EMA_H_L': '',
        'Range': df['Range'],
        'Close_Low':df['Close_Low'],
        'High_Close':df['High_Close'],
        'Avg_Range': df['Avg_Range'],
        'Slow_Mean_Chg': '',
        'EMA_fast': dfmean_fast_C,
        'EMA_slow': dfmean_slow_C,
        'EMA_3': EMA_3,
        'EMA_5': EMA_5,
        'EMA_50': EMA_50,
        'SMA_20':SMA_20,
        'EMA_20':EMA_20,
        'EMA_11':EMA_11,
        'BB_Std':BB_Std,
        'BB1_Upper':BB1_Upper,
        'BB2_Upper': BB2_Upper,
        'BB1_Lower': BB1_Lower,
        'BB2_Lower': BB2_Lower,
        'Volume': df['Volume'],
        'BB_width':'',
        'MACD': MACD,
        'MACD_Signal': MACD_Signal,
        'MACD_Hist': MACD_hist,
        'Signal_B': 0,
        'Signal_SL': 0,
        'Signal_SP': 0,
        'PnL': 0,
        'Present_Value': 0
    }

    """Create a DataFrame with dictionary data"""

    df3 = pd.DataFrame(data1)
    df3['MACD_Chg'] = round(df3['MACD'].pct_change() * 100, 1)
    df3['MACD_Sig_Chg'] = round(df3['MACD_Signal'].pct_change() * 100, 1)
    df3['Slow_Mean_Chg'] = round(df3['EMA_slow'].pct_change() * 100, 1)
    df3['Fast_Mean_Chg'] = round(df3['EMA_fast'].pct_change() * 100, 1)
    df3['Mean_50_Chg'] = round(df3['EMA_50'].pct_change() * 100, 1)
    df3['Volume_10'] = round(df3['Volume'].rolling(10, min_periods=1).mean(), 1)
    df3['Volume_10_max'] = df3['Volume'].rolling(10, min_periods=1).max()
    df3['Volume_10_min'] = df3['Volume'].rolling(10, min_periods=1).min()
    df3['Range_10_max'] = df3['Range'].rolling(10, min_periods=1).max()
    df3['Range_10_min'] = df3['Range'].rolling(10, min_periods=1).min()
#     df3['Slow_Mean_Chg'] = pd.to_numeric(df3['Slow_Mean_Chg'])
    df3['EMA_H_L'] = df3['EMA_H_C'] - df3['EMA_L_C']
    df3['3EMA_H_L'] = round(df3['EMA_H_L'].ewm(span=3).mean(), 2)
    df3['Slow_EMA_H_L'] = df3['Slow_EMA_H_C'] - df3['Slow_EMA_L_C']
    df3['Slow_3EMA_H_L'] = round(df3['Slow_EMA_H_L'].ewm(span=3).mean(), 2)

    """ High/Low Change to identify HH,HL,LH,LL and 20 day High/Low calculation """
    df3['High_Chg'] = round(df3['High'].pct_change() * 100, 1)
    df3['Low_Chg'] = round(df3['Low'].pct_change() * 100, 1)
    df3['High_20day'] = df3['High'].rolling(20).max()
    df3['Low_20day'] = df3['Low'].rolling(20).min()
    df3['C_High_20day'] = df3['Close'].rolling(20).max()
    df3['C_Low_20day'] = df3['Close'].rolling(20).min()
    df3[['ATR_UP','ATR_Down','ATR_TSL']] = atr(df3[['High','Low','Close']])
    df3['Slope_5'],df3['Reg_5'] = slope(df3["Close"],5)
    df3['Slope_10'],df3['Reg_10'] = slope(df3["Close"],10)
    df3['Reg_5_Pct'] = df3['Reg_5'].pct_change()
    df3['Reg_10_Pct'] = df3['Reg_10'].pct_change()
    
    """ Bollinger Band width calculation """
    df3['BB1_width'] = df3['BB1_Upper'] - df3['BB1_Lower']
    df3['BB2_width'] = df3['BB2_Upper'] - df3['BB2_Lower']
    df3['BB1_30days'] = df3['BB1_width'].rolling(30).min()
    df3['BB2_30days'] = df3['BB2_width'].rolling(30).min()
    
    """ Variables required to track the cash flow of the strategy """
    len_di = len(df3.index)
    cnt_trades = 0
    profit_trades = 0
    loss_trades= 0
    position = False 
    start_value = 100000
    available_cash = 0
    latest_value = 0
    risk = .02
    trail = .02
#     print('Starting Portfolio Value : {}'.format(start_value))
    for row_index, row_data in df3.iterrows():
        k = row_index + 5
        if (k < len_di):
            df3.loc[k,'Signal_SL'] = np.NaN
            df3.loc[k,'Signal_SP'] = np.NaN
            df3.loc[k,'Signal_B'] = np.NaN
            if not position:
                """ Calculate the buy qty for available cash if strategy condition is satisfied """
                buy_qty = floor(start_value / df3.loc[k, 'Close'])
#                 Signal2_3EMA_H_L = (0 < df3.loc[k - 3, '3EMA_H_L'] < df3.loc[k - 2, '3EMA_H_L']) and (df3.loc[k - 2, '3EMA_H_L'] > df3.loc[k - 1, '3EMA_H_L'] > df3.loc[k, '3EMA_H_L'] > 0) and df3.loc[k, 'Close_Chg'] <= 0
#                 BB1_Width = (df3.loc[k - 3, 'BB1_width'] < df3.loc[k - 2, 'BB1_width'] < df3.loc[k - 1, 'BB1_width'] < df3.loc[k, 'BB1_width']) and df3.loc[k, 'Close_Chg'] < 0 and (df3.loc[k - 3, 'Close_Chg'] > 0.5 or df3.loc[k - 2, 'Close_Chg'] > 0.5)
#                 SMA_20_Signal1 = df3.loc[k - 2, 'Close'] > df3.loc[k - 2, 'SMA_20'] and df3.loc[k - 1, 'Close'] > df3.loc[k - 1, 'SMA_20'] and df3.loc[k, 'Low'] < df3.loc[k, 'SMA_20'] and df3.loc[k, 'Close'] > df3.loc[k, 'SMA_20'] and df3.loc[k, 'Close'] >= df3.loc[k, 'EMA_50']
#                 EMA_Cross_MACD_MACDP_11_26 = df3.loc[k, 'EMA_fast'] > df3.loc[k, 'EMA_slow'] and df3.loc[k - 1, 'EMA_fast'] < df3.loc[k - 1, 'EMA_slow'] and df3.loc[k, 'MACD'] >= df3.loc[k - 1, 'MACD'] and np.abs(df3.loc[k, 'Close_Chg'])<=3
#                 if SMA_20_Signal1:
#                 # SMA_20 Signal2
#                 if ((df3.loc[k - 2, 'Close'] < df3.loc[k - 2, 'SMA_20'] and df3.loc[k - 1, 'Close'] < df3.loc[k - 1, 'SMA_20']) and 
#                         (df3.loc[k, 'Low'] < df3.loc[k, 'SMA_20']) and (df3.loc[k, 'Close'] > df3.loc[k, 'SMA_20'])): 
#                 # EMA_50_Signal
#                 if ((df3.loc[k - 2, 'Close'] > df3.loc[k - 2, 'EMA_50'] and df3.loc[k - 1, 'Close'] > df3.loc[k - 1, 'EMA_50']) and 
#                         (df3.loc[k, 'Low'] < df3.loc[k, 'EMA_50']) and (df3.loc[k, 'Close'] > df3.loc[k, 'EMA_50'])):
#                 # EMA_Cross_MACD_MACDP_11_26
#                 if ((df3.loc[k, 'EMA_fast'] > df3.loc[k, 'EMA_slow']) and (df3.loc[k - 1, 'EMA_fast'] < df3.loc[k - 1, 'EMA_slow']) and 
#                             (df3.loc[k, 'MACD'] >= df3.loc[k - 1, 'MACD'])):
#                 # EMA_Cross_MACD_Signal_11_26
#                 if ((df3.loc[k, 'EMA_fast'] > df3.loc[k, 'EMA_slow']) and (df3.loc[k - 1, 'EMA_fast'] < df3.loc[k - 1, 'EMA_slow']) and 
#                             df3.loc[k, 'MACD_Signal'] >= df3.loc[k - 1, 'MACD_Signal']):
#                 #EMA_fast_11_Signal1
#                 if ((df3.loc[k - 2, 'Close'] > df3.loc[k - 2, 'EMA_fast'] and df3.loc[k - 1, 'Close'] > df3.loc[k - 1, 'EMA_fast']) and 
#                             (df3.loc[k, 'Low'] < df3.loc[k, 'EMA_fast']) and (df3.loc[k, 'Close'] > df3.loc[k, 'EMA_fast'])):
#                 #EMA_fast_11_Signal2
#                 if ((df3.loc[k - 2, 'Close'] < df3.loc[k - 2, 'EMA_fast'] and df3.loc[k - 1, 'Close'] < df3.loc[k - 1, 'EMA_fast']) and 
#                         (df3.loc[k, 'Low'] < df3.loc[k, 'EMA_fast']) and (df3.loc[k, 'Close'] > df3.loc[k, 'EMA_fast'])):
#                 #EMA_slow_26_Signal1
#                 if ((df3.loc[k - 2, 'Close'] > df3.loc[k - 2, 'EMA_slow'] and df3.loc[k - 1, 'Close'] > df3.loc[k - 1, 'EMA_slow']) and 
#                         (df3.loc[k, 'Low'] < df3.loc[k, 'EMA_slow']) and (df3.loc[k, 'Close'] > df3.loc[k, 'EMA_slow'])):
#                 #EMA_slow_26_Signal2
#                 if ((df3.loc[k - 2, 'Close'] < df3.loc[k - 2, 'EMA_slow'] and df3.loc[k - 1, 'Close'] < df3.loc[k - 1, 'EMA_slow']) and 
#                         (df3.loc[k, 'Low'] < df3.loc[k, 'EMA_slow']) and (df3.loc[k, 'Close'] > df3.loc[k, 'EMA_slow'])):
#                 # Infy1_EMA_H_L
#                 if (0 < (df3.loc[k - 3, 'EMA_H_L'] < df3.loc[k - 2, 'EMA_H_L']) and (df3.loc[k - 2, 'EMA_H_L'] > df3.loc[k - 1, 'EMA_H_L'] > 0) and 
#                         (df3.loc[k - 1, 'EMA_H_L'] > df3.loc[k, 'EMA_H_L'] > 0) and (0 <= df3.loc[k, 'Close_Chg'] <= 3.5)):
#                 # Narrow range for last 10 days
#                 if (df3.loc[k, 'Range'] < df3.loc[k, 'Avg_Range'] and df3.loc[k, 'Range'] == df3.loc[k, 'Range_10_min']
#                           and (df3.loc[k, 'Close'] < df3.loc[k, 'EMA_fast'] or df3.loc[k, 'MACD'] < df3.loc[k, 'MACD_Signal'])):
#                 # EMA_Cross_MACD_MACDP_3_5
#                 if ((df3.loc[k, 'EMA_3'] > df3.loc[k, 'EMA_5']) and (df3.loc[k - 1, 'EMA_3'] < df3.loc[k - 1, 'EMA_5']) and 
#                             (df3.loc[k, 'MACD'] >= df3.loc[k - 1, 'MACD'])):
#                 # EMA_Cross_MACD_Signal_3_5
#                 if ((df3.loc[k, 'EMA_3'] > df3.loc[k, 'EMA_5']) and (df3.loc[k - 1, 'EMA_3'] < df3.loc[k - 1, 'EMA_5']) and df3.loc[k, 'MACD_Signal'] >= df3.loc[k - 1, 'MACD_Signal']):
                Reg_Cross = df3.loc[k, 'Reg_5'] > df3.loc[k, 'Reg_10'] and df3.loc[k - 1, 'Reg_5'] <= df3.loc[k - 1, 'Reg_10'] and df3.loc[k, 'Close'] > df3.loc[k, 'EMA_20']
                Reg_5_cond = df3.loc[k-1,'Reg_5_Pct']<0 or df3.loc[k-2,'Reg_5_Pct']<0 or df3.loc[k-3,'Reg_5_Pct']<0

                if Reg_Cross and Reg_5_cond:
#                 """ Specify the strategy condition to be verified """
#                 if ((df3.loc[k - 3, 'BB1_width'] < df3.loc[k - 2, 'BB1_width'] < df3.loc[k - 1, 'BB1_width'] < df3.loc[k, 'BB1_width']) and df3.loc[k, 'Close_Chg'] < 0 and
#                     (df3.loc[k - 3, 'Close_Chg'] > 0.5 or df3.loc[k - 2, 'Close_Chg'] > 0.5)):
#                     df3.loc[k, 'BB_width'] = 'Long'
#                 if ((df3.loc[k - 1, 'BB_width'] == 'Long') and df3.loc[k, 'Close_Chg'] < 0 and (df3.loc[k - 1, 'BB1_width'] < df3.loc[k, 'BB1_width'])):
                    """ Calculate the parameters if a Buy is executed """
                    cnt_trades += 1
                    buy_price = df3.loc[k, 'Close']
                    stop_price = buy_price - risk * buy_price
                    trail_price = buy_price + trail*buy_price
                    risk_amt = (buy_price - stop_price) * buy_qty
                    buy_cost = buy_qty * df3.loc[k, 'Close']
                    available_cash = start_value - buy_cost
                    position = True
                    trail_start = False
                    df3.loc[k,'Signal_B'] = buy_price
#                     print('Date: {0[0]}, Open: {0[1]} ,High: {0[2]}, Low: {0[3]}, Close: {0[4]}'.format(df3.iloc[k, 1:6].tolist()))
#                     print('BUY EXECUTED , Qty: %d, Total Buy Cost :%.2f, Available Cash: %.2f' % (buy_qty, buy_cost, available_cash))
#                     print('Buy Price: {0:.2f}, Stop Price: {1:.2f}'.format(buy_price, stop_price))
#                     print('=====================================================================')
            else:
                
                """ Sell below the STOP LOSS in case if GAP DOWN in Open Price """
                if (df3.loc[k, 'Open'] < stop_price ) and position and not trail_start:
                    stop_price = df3.loc[k, 'Open']
#                     print('Date: {0[0]}, Open: {0[1]} ,High: {0[2]}, Low: {0[3]}, Close: {0[4]}'.format(df3.iloc[k, 1:6].tolist()))
#                     print('SELL AT GAP DOWN STOP PRICE, %.2f, Sell Qty, %d' % (stop_price,buy_qty))
                    risk_amt = (buy_price - stop_price) * buy_qty
                    start_value = start_value - risk_amt - (.00206*buy_cost) # Brokerage Cost is considered too
                    loss_trades += 1
                    position = False
                    df3.loc[k,'Signal_SL'] = stop_price
                    df3.loc[k,'PnL'] = -risk_amt
                    df3.loc[k,'Present_Value'] = start_value
#                     print('LOSS: - %.2f , Latest Position Value: %.2f' % (risk_amt, start_value))
#                     print('*******************************************************************')
                    continue
                                   
                    """ Sell at Stop Loss Limit Price """
                if (df3.loc[k, 'Low'] < stop_price < df3.loc[k, 'High'] and position and not trail_start):
#                     print('Date: {0[0]}, Open: {0[1]} ,High: {0[2]}, Low: {0[3]}, Close: {0[4]}'.format(df3.iloc[k, 1:6].tolist()))
#                     print('SELL AT STOP PRICE, %.2f, Sell Qty, %d' % (stop_price,buy_qty))
                    risk_amt = (buy_price - stop_price) * buy_qty
                    start_value = start_value - risk_amt - (.00206*buy_cost) # Brokerage Cost is considered too
                    position = False
                    loss_trades += 1
                    df3.loc[k,'Signal_SL'] = stop_price
                    df3.loc[k,'PnL'] = -risk_amt
                    df3.loc[k,'Present_Value'] = start_value
#                     print('LOSS: - %.2f , Latest Position Value: %.2f' % (risk_amt, start_value))
#                     print('*******************************************************************')
                    continue
                    
                    """ Target Price based on ATR Trailing Stop Loss """
                if (df3.loc[k, 'Close'] < trail_price and stop_price < df3.loc[k, 'Close'] and position and not trail_start):
#                     print('cond1',df3.loc[k, 'Close'], df3.loc[k-1, 'Close'],df3.loc[k, 'Date'],trail_price)
                    continue
                
                if (df3.loc[k, 'Low'] > trail_price and df3.loc[k, 'Close'] < df3.loc[k-1, 'Close'] and position and trail_start):
#                     print('cond2',df3.loc[k, 'Close'], df3.loc[k-1, 'Close'],df3.loc[k, 'Date'],trail_price)
                    continue
                                       
                if (df3.loc[k, 'Open'] < trail_price and position and trail_start):
                    target_price = df3.loc[k, 'Open']
#                     print('Date: {0[0]}, Open: {0[1]} ,High: {0[2]}, Low: {0[3]}, Close: {0[4]}'.format(df3.iloc[k, 1:6].tolist()))
#                     print('SELL AT GAPDOWN TRAILING TARGET, %.2f, Sell Qty, %d' % (target_price,buy_qty))
                    tgt_amt = ((target_price-buy_price) * buy_qty)-(.00212*buy_cost)
                    start_value = start_value + tgt_amt
                    position = False
                    if tgt_amt >=0:
                        profit_trades += 1
                    else:
                        loss_trades += 1
                    df3.loc[k,'Signal_SP'] = target_price
                    df3.loc[k,'PnL'] = tgt_amt
                    df3.loc[k,'Present_Value'] = start_value
#                     print('PROFIT: + %.2f , Latest Position Value: %.2f' % (tgt_amt, start_value))
#                     print('*******************************************************************')
                    
                if (df3.loc[k, 'High'] > trail_price > df3.loc[k, 'Low'] and position and trail_start):
                    target_price = trail_price
#                     print('Date: {0[0]}, Open: {0[1]} ,High: {0[2]}, Low: {0[3]}, Close: {0[4]}'.format(df3.iloc[k, 1:6].tolist()))
#                     print('SELL AT TRAILING TARGET, %.2f, Sell Qty, %d' % (target_price,buy_qty))
                    tgt_amt = ((target_price-buy_price) * buy_qty)-(.00212*buy_cost)
                    start_value = start_value + tgt_amt
                    position = False
                    if tgt_amt >=0:
                        profit_trades += 1
                    else:
                        loss_trades += 1
                    df3.loc[k,'Signal_SP'] = target_price
                    df3.loc[k,'PnL'] = tgt_amt
                    df3.loc[k,'Present_Value'] = start_value
#                     print('PROFIT: + %.2f , Latest Position Value: %.2f' % (tgt_amt, start_value))
#                     print('*******************************************************************')
                    
                if (df3.loc[k, 'Close'] > trail_price and position):
                    if not trail_start:
                        trail_start = True
                        trail_price = df3.loc[k, 'Close'] - trail_price*.02
#                         print('Trail Started at {0}'.format(trail_price))
#                         print( df3.loc[k, 'Close'], df3.loc[k-1, 'Close'],df3.loc[k, 'Date'])
                    elif df3.loc[k, 'Low'] > trail_price and df3.loc[k, 'Close'] > df3.loc[k-1, 'Close'] :
#                         print( df3.loc[k, 'Close'], df3.loc[k-1, 'Close'],df3.loc[k, 'Date'])
                        new_trail = df3.loc[k, 'Close'] - trail_price*.02
                        if trail_price<new_trail:
                            trail_price = new_trail
#                         print('Trail Set to {0}'.format(trail_price))
        df3.loc[k,'Trade_Count'] = cnt_trades
        df3.loc[k,'Profit_Trades'] = profit_trades
        df3.loc[k,'Loss_Trades'] = loss_trades
        df3.loc[k,'Portfolio_Value'] = round(start_value,2)  
        df7 = df3[['Date', 'Open', 'High', 'Low', 'Close','Volume','Close_Chg','Reg_5','Reg_10','EMA_20','Signal_B','Signal_SL','Signal_SP','PnL','Present_Value']]
        #df4 = df3[['Date', 'Open', 'High', 'Low', 'Close','Volume','Close_Chg','EMA_fast','EMA_slow','ATR_TSL','ATR_UP','ATR_Down','Signal_B','Signal_SL','Signal_SP','PnL','Present_Value']]
        #df4 = df3[['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Close_Chg','My_Signal']][df3['My_Signal']!=""]
    df7.to_csv(os.path.join(new_path,"Results_"+stock+".csv"),index=False)
    
    df4 = df3[['Date','Symbol', 'Trade_Count', 'Profit_Trades', 'Loss_Trades', 'Portfolio_Value']][(df3['Date'] >= '2014-1-1') & (df3['Date'] <= '2014-12-31')]
    year_start = 100000
    try:
        year_end = df4['Portfolio_Value'][df4['Date']==df4['Date'].max()].values[0]
        profit = year_end - year_start
        profit_perc = (year_end - year_start) / year_start * 100
        df4['Year'] = pd.DatetimeIndex(df4['Date']).year
        df4.loc[df4.index[-1],'Start'] = round(year_start,2)
        df4.loc[df4.index[-1],'End'] = round(year_end,2)
        df4.loc[df4.index[-1],'Profit'] = round(profit,2)
        df4.loc[df4.index[-1],'PL_Perc'] = str(round(profit_perc,2)) +'%'
        df5 = df5.append(df4.iloc[:,1:11].tail(1))
    except:
        pass

    df4 = df3[['Date','Symbol', 'Trade_Count', 'Profit_Trades', 'Loss_Trades', 'Portfolio_Value']][(df3['Date'] >= '2015-1-1') & (df3['Date'] <= '2015-12-31')]
    year_start = df4['Portfolio_Value'][df4['Date']==df4['Date'].min()].values[0]
    year_end = df4['Portfolio_Value'][df4['Date']==df4['Date'].max()].values[0]
    profit = year_end - year_start
    profit_perc = (year_end - year_start) / year_start * 100
    df4['Year'] = pd.DatetimeIndex(df4['Date']).year
    df4.loc[df4.index[-1],'Start'] = round(year_start,2)
    df4.loc[df4.index[-1],'End'] = round(year_end,2)
    df4.loc[df4.index[-1],'Profit'] = round(profit,2)
    df4.loc[df4.index[-1],'PL_Perc'] = str(round(profit_perc,2)) +'%'
    df5 = df5.append(df4.iloc[:,1:11].tail(1))

    df4 = df3[['Date','Symbol', 'Trade_Count', 'Profit_Trades', 'Loss_Trades', 'Portfolio_Value']][(df3['Date'] >= '2016-1-1') & (df3['Date'] <= '2016-12-31')]
    year_start = df4['Portfolio_Value'][df4['Date']==df4['Date'].min()].values[0]
    year_end = df4['Portfolio_Value'][df4['Date']==df4['Date'].max()].values[0]
    profit = year_end - year_start
    profit_perc = (year_end - year_start) / year_start * 100
    df4['Year'] = pd.DatetimeIndex(df4['Date']).year
    df4.loc[df4.index[-1],'Start'] = round(year_start,2)
    df4.loc[df4.index[-1],'End'] = round(year_end,2)
    df4.loc[df4.index[-1],'Profit'] = round(profit,2)
    df4.loc[df4.index[-1],'PL_Perc'] = str(round(profit_perc,2)) +'%'
    df5 = df5.append(df4.iloc[:,1:11].tail(1))

    df4 = df3[['Date','Symbol', 'Trade_Count', 'Profit_Trades', 'Loss_Trades', 'Portfolio_Value']][(df3['Date'] >= '2017-1-1') & (df3['Date'] <= '2017-12-31')]
    year_start = df4['Portfolio_Value'][df4['Date']==df4['Date'].min()].values[0]
    year_end = df4['Portfolio_Value'][df4['Date']==df4['Date'].max()].values[0]
    profit = year_end - year_start
    profit_perc = (year_end - year_start) / year_start * 100
    df4['Year'] = pd.DatetimeIndex(df4['Date']).year
    df4.loc[df4.index[-1],'Start'] = round(year_start,2)
    df4.loc[df4.index[-1],'End'] = round(year_end,2)
    df4.loc[df4.index[-1],'Profit'] = round(profit,2)
    df4.loc[df4.index[-1],'PL_Perc'] = str(round(profit_perc,2)) +'%'
    df5 = df5.append(df4.iloc[:,1:11].tail(1))

    df4 = df3[['Date','Symbol', 'Trade_Count', 'Profit_Trades', 'Loss_Trades', 'Portfolio_Value']][(df3['Date'] >= '2018-1-1') & (df3['Date'] <= '2018-12-31')]
    year_start = df4['Portfolio_Value'][df4['Date']==df4['Date'].min()].values[0]
    year_end = df4['Portfolio_Value'][df4['Date']==df4['Date'].max()].values[0]
    profit = year_end - year_start
    profit_perc = (year_end - year_start) / year_start * 100
    df4['Year'] = pd.DatetimeIndex(df4['Date']).year
    df4.loc[df4.index[-1],'Start'] = round(year_start,2)
    df4.loc[df4.index[-1],'End'] = round(year_end,2)
    df4.loc[df4.index[-1],'Profit'] = round(profit,2)
    df4.loc[df4.index[-1],'PL_Perc'] = str(round(profit_perc,2)) +'%'
    df5 = df5.append(df4.iloc[:,1:11].tail(1))

    df4 = df3[['Date','Symbol', 'Trade_Count', 'Profit_Trades', 'Loss_Trades', 'Portfolio_Value']][(df3['Date'] >= '2019-1-1') & (df3['Date'] <= '2019-12-31')]
    year_start = df4['Portfolio_Value'][df4['Date']==df4['Date'].min()].values[0]
    year_end = df4['Portfolio_Value'][df4['Date']==df4['Date'].max()].values[0]
    profit = year_end - year_start
    profit_perc = (year_end - year_start) / year_start * 100
    df4['Year'] = pd.DatetimeIndex(df4['Date']).year
    df4.loc[df4.index[-1],'Start'] = round(year_start,2)
    df4.loc[df4.index[-1],'End'] = round(year_end,2)
    df4.loc[df4.index[-1],'Profit'] = round(profit,2)
    df4.loc[df4.index[-1],'PL_Perc'] = str(round(profit_perc,2)) +'%'
    df5 = df5.append(df4.iloc[:,1:11].tail(1))
    
    df4 = df3[['Date','Symbol', 'Trade_Count', 'Profit_Trades', 'Loss_Trades', 'Portfolio_Value']][(df3['Date'] >= '2020-1-1') & (df3['Date'] <= '2020-7-9')]
    year_start = df4['Portfolio_Value'][df4['Date']==df4['Date'].min()].values[0]
    year_end = df4['Portfolio_Value'][df4['Date']==df4['Date'].max()].values[0]
    profit = year_end - year_start
    profit_perc = (year_end - year_start) / year_start * 100
    df4['Year'] = pd.DatetimeIndex(df4['Date']).year
    df4.loc[df4.index[-1],'Start'] = round(year_start,2)
    df4.loc[df4.index[-1],'End'] = round(year_end,2)
    df4.loc[df4.index[-1],'Profit'] = round(profit,2)
    df4.loc[df4.index[-1],'PL_Perc'] = str(round(profit_perc,2)) +'%'
    df5 = df5.append(df4.iloc[:,1:11].tail(1))
    
    

    df8 = df3[['Symbol', 'Trade_Count', 'Profit_Trades', 'Loss_Trades', 'Portfolio_Value']].tail(1)
    df8['Start_Portfolio'] = 100000
    no_of_trades = df8['Trade_Count'].values[0] if df8['Trade_Count'].values[0]>0 else 1000
    df8['Win_Loss_Perc'] = round(df8['Profit_Trades'].values[0]/no_of_trades,2)*100
    df8['P/L_Perc'] = round((df8['Portfolio_Value'].values[0]-100000)/100000,2)*100
    df6 = df6.append(df8)
    #

df5.to_csv(os.path.join(new_path,"Results_Yearwise_"+strategy+".csv"),index=False)
df6.sort_values(by='Symbol').to_csv(os.path.join(new_path,"Results_Overall_"+strategy+".csv"),index=False)
