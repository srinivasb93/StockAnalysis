#Fetch all table names from the database
fetch_all_stocks = "SELECT name FROM sys.tables"

fetch_Next50_stocks = "SELECT Symbol FROM dbo.NIFTYNEXT50"
fetch_Mid50_stocks = "SELECT Symbol FROM dbo.NIFTYMID50"
all_stocks_list,nifty_50_list,nifty_next50_list,nifty_mid_list = []
all_stocks = pd.read_sql(fetch_all_stocks,con=conn,parse_dates=True)
all_stocks = all_stocks.fetchall()
for data in all_stocks:
    all_stocks_list.append(data[0])

fetch_N50_stocks = "SELECT Symbol FROM dbo.NIFTY50"
n50_stocks = pd.read_sql(fetch_N50_stocks,con=conn,parse_dates=True)
n50_stocks = all_stocks.fetchall()
for data in n50_stocks:
    nifty_50_list.append(data[0])

fetch_Next50_stocks = "SELECT Symbol FROM dbo.NIFTYNEXT50"
next50_list = pd.read_sql(fetch_Next50_stocks,con=conn,parse_dates=True)
next50_list = all_stocks.fetchall()
for data in next50_list:
    nifty_next50_list.append(data[0])