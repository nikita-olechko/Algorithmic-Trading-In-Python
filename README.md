# Live-Algorithmic-Trading-In-Python

This project contains a live trading bot that executes trades on the Interactive Brokers platform. The bot is designed to be highly customizable, allowing you to define your own trading strategies. 

As an example, the bot is set to use a testing strategy based on the 60-period Simple Moving Average (SMA). 

***Please note that this strategy is for testing purposes only and not recommended for live trading.***

***See Full Disclaimer Below.***

# Features

- Connects to Interactive Brokers API for live trading
- Supports customizable strategies
- Built with a modular design for adding more order types and quantities
- Supports limit and stop-loss orders as well as market orders
- Supports customized backtesting on historical data
- Can manage multiple trading bots running different strategies simultaneously (up to 30 trades per second per bot)
- Sample testing strategy based on 60-period SMA

# Requirements

- Python 3.6 or higher
- Interactive Brokers account
- IBKR Trader Workstation installed, open, and running on port 4000
- Install necessary Python packages: see requirements.txt for complete list

# Setup and Usage

- Clone the repository to your local machine.
- Install the necessary Python packages using pip:
    pip install pandas numpy matplotlib seaborn ibapi scikit-learn ta pytz APScheduler ib_insync

Update the Bot instantiation at the bottom of LiveTradingBot.py to reflect your desired trading symbol, quantity, and strategy.
Run the Python script:

    python LiveTradingBot.py

# Customizing Strategies

You can customize the trading bot by modifying the strategy functions buySellConditionFunc and generateNewDataFunc in the Bot class. These functions determine the buy/sell decisions and data generation for each trading bot, respectively.

For example, in the testing strategy greaterthan60barsma.py, there are two functions:

- generate60PeriodSMA(barDataFrame): 
    This function generates new data based on existing bar data. 
    In this case, it calculates the 60-period SMA.

- sampleSMABuySellStrategy(barDataFrame): 
    This function defines the strategy's decision-making process. 
    It returns "BUY", "SELL", or "" (no action), based on whether the average price is above or below the 60-period SMA.

At the moment, the bot can only trade a fixed quantity of the asset. For example, if you specify 100, and the ticker "AAPL", the bot will place orders for 100 shares of AAPL according to your strategy. In this example, it will BUY 100 shares if above the 60-period SMA, and SELL if below the 60-period SMA. 

Orders do NOT stack. Meaning if your strategy places a BUY order, it will not place another BUY order unti a SELL order has been placed, even if the BUY condition has been met. This corresponds with the fixed quantity capacity of the current model.

You can create a new strategy by writing similar functions that align with your trading approach. Of course, any data generated in the first function can and should be used in the second. Future updates will provide more functionality related to the type and quantity of orders placed.

# Simulatenously Running Strategies

To run several strategies simulatenously, you can instantiate new instances of the Bot class, which will run the strategy. I would recommend running them in a seaparate console if you wish to see live updates, otherwise it gets very confusing very fast. 

If running multiple strategies, OR restarting strategies multiple times in one day:

- the Bot class argument twsConnectionID must be a unique integer for each running instance of the bot
- the Bot class argument orderStarterID should be passed the function get_starter_order_id(n) where n is an integer not passed to this function today (order ids reset daily)

# Backtesting

Backtesting is now available in Python! Here are the steps to backtest a strategy:

- Step 0: Create your strategy condition functions in the same way as you would for live trading. See the section above on Customizing Strategies for more details.
- Step 1: Assuming you are generating new data, you MUST modify your data generation function to operate on the entire dataframe instead of only the last row. This is because the backtesting framework requires the entire dataframe to be generated at once. For example, if you are calculating the 60-period SMA, you must calculate the 60-period SMA for the entire dataframe, not just the last row. See the sample strategy greaterthan60barsma.py for an example of how to do this.
- Step 2: Make sure IBKR TWS is open and running on port 4000.
- Step 3: Open backtesting.py and import your strategy functions like so:
    
        ```python
        from mystrategy import myStrategyBuySellConditionFunction, myStrategyGenerateAdditionalDataFunction
        ```

- Step 4: Change the variables strategy_name, strategy_buy_or_sell_condition_function, generate_additional_data_function to reflect your strategy. For example, if your strategy is called "myStrategy", you would change the variables to:

        ```python
        strategy_name = "myStrategy"
        strategy_buy_or_sell_condition_function = myStrategyBuySellConditionFunction
        generate_additional_data_function = myStrategyGenerateAdditionalDataFunction
        ```

- Step 4b: Change the variables barsize and duration to reflect your desired backtesting timeframe. A full list of possible values can be found here: https://interactivebrokers.github.io/tws-api/historical_bars.html#hd_duration. For example, if you want to backtest on 1-minute bars for the last 3 months, you would change the variables to:

        ```python
        barsize = "1 min"
        duration = "3 M"
        ```

- Step 5: Run backtesting.py. The results will be saved in a CSV file in the directory data/Strategy Results, with the name of the file as your strategy name + barsize + duration (e.g. myStrategy1min3M). If you would like to customize the results from backtesting, you can modify the function create_summary_data in backtestingUtilities/simulationUtilities.py. This function is called after the backtesting is complete, and generates the results saved to the CSV file.

Note: There is an equally functional backtesting framework in R, which can be found in the complementary repository "Backtesting Trading Strategies". However, I am no longer directly supporting the R version, and I recommend using the Python version instead.

# Modelling

Far be it from me to tell you how to model your data. However, a simple linear model framework using sci-kit learn is available in models/relative_price_change.py. You can see how parameters are added to base data, and how those parameters are used to model the next price change using a linear model. 

More importantly, in liveTest_relative_price_change you can see how to directly implement your model into operating on live market data. This can then be directly translated into a trading strategy. Note that it is designed for separate models for individual tickers, I.E. "relative_price_change_XOM.pkl" is unique to the ticker "XOM".

# Contact Me

If you have any questions, suggestions, or want to discuss ideas about improving or extending the program, feel free to reach out to me:

- Email: nikita.olechko@gmail.com
- LinkedIn: https://www.linkedin.com/in/nikitaolechko/
- GitHub: https://github.com/nikita-olechko

# Disclaimer

Use this bot at your own risk. This bot is a proof-of-concept and is not meant to be used for live trading without careful review and understanding of its risks and limitations. Financial trading involves substantial risk and should not be done without careful analysis and observation. Any financial numbers referenced here, or on any of my sites and projects, are simply estimates or projections and should not be considered exact, actual, or any form of guarantee of potential earnings. I am NOT an accredited financial advisor. This is a PERSONAL project and is NOT meant to be used for commercial purposes. I am NOT responsible for any losses you may incur by using this bot and offer no guarantees of any kind, including but not limited to strategy profitability, strategy accuracy, software functionality, and software security.
