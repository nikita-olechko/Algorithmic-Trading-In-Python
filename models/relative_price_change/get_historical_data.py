from utilities.generalUtilities import get_months_of_historical_data, initialize_ib_connection, \
    get_days_of_historical_data

ib = initialize_ib_connection()
# data = get_months_of_historical_data(ib, 'XOM', months=12, directory_offset=2)

for ticker in ["NVDA", "AAPL", "TSLA", "AMZN"]:
    get_months_of_historical_data(ib, ticker, months=12, barsize="1 min", directory_offset=2)

# for ticker in ["NVDA", "AAPL", "TSLA", "AMZN"]:
#     get_days_of_historical_data(ib, ticker, days=30, barsize="5 secs", directory_offset=2)