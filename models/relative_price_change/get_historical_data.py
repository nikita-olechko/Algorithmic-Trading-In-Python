from utilities.generalUtilities import get_months_of_historical_data, initialize_ib_connection

ib = initialize_ib_connection()
# data = get_months_of_historical_data(ib, 'XOM', months=12, directory_offset=2)

for ticker in ["NVDA", "AAPL", "TSLA", "AMZN", "GOOG", "FB", "MSFT", "NFLX", "INTC", "AMD"]:
    get_months_of_historical_data(ib, ticker, months=12, barsize="1 Min", directory_offset=2)