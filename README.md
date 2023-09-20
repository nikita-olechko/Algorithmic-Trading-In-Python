# Algorithmic Trading In Python

This project allows users to build, test, and trade custom algorithmic trading strategies based on any parameters. It features a Live Trading bot with real-time data connection, a system to backtest your strategy on historical market data, and a basic framework for building and testing predictive models on market data.

As a demonstrative example, the bot is set to use a sample strategy based on the 10-period Simple Moving Average (SMA). 

***Please note that this strategy is for testing purposes only and not recommended for live trading.***

# Who Is This Program For?

This program is tailored for users who meet the following criteria:

Basic Python Proficiency: Users should have a fundamental understanding of Python programming, including variables, functions, and data structures.

Interest in Algorithmic Trading: Users should have an interest in algorithmic trading and a basic understanding of financial markets. This program is not intended for absolute beginners in either Python or trading. 

Willingness to Learn: While I provide instructions, users are encouraged to explore the program, understand how it works, and see how to expand to meet their trading criteria. This requires a willingness to learn and experiment. 

Risk Awareness: Users must understand the risks associated with algorithmic trading and should not use this program for live trading without thorough analysis.

***See Full Disclaimer Below.***

# Features

- Connects to Interactive Brokers API for live trading
- Supports customizable strategies
- Built with a modular design for adding more order types and quantities
- Supports limit and stop-loss orders as well as market orders
- Supports customized backtesting on historical data
- Can manage multiple trading bots running different strategies simultaneously
- Sample testing strategy based on 10-period SMA
- Easy integration of Machine Learning / Deep Learning models to live trading strategies

# Demo

Watch a brief demo of backtesting and live trading in action here:

https://youtu.be/3PZ054bn8Jw

# Requirements

- Python 3.6 or higher
- Interactive Brokers (IBKR) account
- IBKR Trader Workstation (TWS) installed, open, and running on port 4000 (this is not the default)
- A subscription to live market data with IBKR (only for live trading / live testing, not required for backtesting)
- Optional (but recommended): A simulated trading account. IBKR allows you to open a simulated account with the same data privileges as your real account. This is ideal for testing strategies on live data risk-free.

# Setup and Usage

- Clone the repository to your local machine.
- Install the necessary Python packages using pip:

      pip install pandas==2.0.3 numpy==1.25.0 matplotlib==3.7.2 seaborn~=0.12.2 ibapi~=9.81.1.post1 scikit-learn==1.3.0 ta~=0.10.2 pytz~=2023.3 APScheduler==3.7.0 ib_insync==0.9.69 statsmodels~=0.14.0 django-basic-stats~=0.2.0 patsy~=0.5.3 scipy~=1.10.1 joblib~=1.2.0

- No additional parameters (or keys or passwords) are required for setup.

Update the Bot instantiation in of RunTradingBot.py to reflect your desired trading symbol, quantity, and strategy.
Navigate into the liveTrading directory and run the Python script:

    python liveTrading\\RunTradingBot.py

There is also a batch file template to run the bot automatically on a daily basis.

# Customizing Strategies

You can customize the trading bot by modifying the strategy functions buySellConditionFunc and generateNewDataFunc in the Bot instantiation. These functions respectively determine the buy/sell decisions and new data generation for each trading bot. **Make sure your customized functions accept the same parameters as below**

For example, in the testing strategy greaterthan10barsma.py, there are two functions:

- generate10PeriodSMA(barDataFrame): 
    This function generates new data based on existing bar data. 
    In this case, it calculates the 10-period SMA.

- sampleSMABuySellStrategy(barDataFrame, last_order_index=0, ticker="", current_index=0): 
    This function defines the strategy's decision-making process. 
    It returns 1 (BUY), -1 (SELL), 2 (HOLD), or 0 (no action), based on whether the average price is above or below the 10-period SMA.
    No action occurs at 2 or 0, but it is helpful for analysis purposes to differentiate the two.

The bot can only trade a fixed quantity of the asset. For example, if you specify 100, and the ticker "AAPL", the bot will place orders for 100 shares of AAPL according to your strategy. In this example, it will BUY 100 shares if above the 10-period SMA, and SELL if below the 10-period SMA. If you wish to customize orders, you will need to add custom order functions to the customOrders.py file in the liveTrading directory, and add your order condition to the place_orders_if_needed() method in LiveTradingBot.py.

Orders in the sample strategy do NOT stack. Meaning if your strategy places a BUY order, it will not place another BUY order until a SELL order has been placed, even if the BUY condition has been met. This corresponds with the fixed quantity capacity of the current model. However, this is built into the strategy, not the bot class itself, as it is designed to be flexible (e.g., maybe you want to double up under specific circumstances). You can check the last order using barDataFrame["Orders"][last_order_index] and build accordingly.

You can create a new strategy by writing similar functions that align with your trading approach. Of course, any data generated in the first function can and should be used in the second. 

# Simulatenously Running Strategies

To run several strategies simulatenously, you can simply instantiate multiple instances of the Bot class, which will run the strategies simulatenously without interference. I would recommend running them in a seaparate console if you wish to see live updates, otherwise updates get confusing. 

# Backtesting

Backtesting is now available in Python! Here are the steps to backtest a strategy:

- Step 0: Create your strategy condition functions in the same way as you would for live trading. See the section above on Customizing Strategies for more details.
- Step 1: Assuming you are generating new data, you MUST modify your data generation function to operate on the entire dataframe instead of only the last row. This is because the backtesting framework requires the entire dataframe to be generated at once. For example, if you are calculating the 10-period SMA, you must calculate the 10-period SMA for the entire dataframe, not just the last row. See the sample strategy greaterthan10barsma.py for an example of how to do this.
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

- Step 4b: Change the variables barsize and duration to reflect your desired backtesting timeframe. A full list of possible values can be found here: https://interactivebrokers.github.io/tws-api/historical_bars.html#hd_duration. For example, if you want to backtest on 1-minute bars for the last 2 months, you would change the variables to:

        ```python
        barsize = "1 min"
        duration = "2 M"
        ```

- Step 5: Run backtesting.py. The results will be saved in a CSV file in the directory data/Strategy Results, with the name of the file as your strategy name + barsize + duration (e.g. myStrategy1min2M). If you would like to customize the results from backtesting, you can modify the function create_summary_data in backtestingUtilities/simulationUtilities.py. This function is called after the backtesting is complete, and generates the results saved to the CSV file.

Note: There is an equally functional backtesting framework in R, which can be found in the complementary repository "Backtesting Trading Strategies". However, I am no longer supporting the R version, and I recommend using the Python version instead.

# Modelling

Far be it from me to tell you how to model your data. However, a simple modeling framework using sci-kit learn is available in models/relative_price_change.py. You can see how parameters are added to base data, and how those parameters are used to model the next price change using a Linear Regression, a Random Forest, and a Multi Layer Perceptron (MLP). 

In liveTest_relative_price_change you can see how to directly implement your model into operating on live market data. This can then be directly translated into a trading strategy. Note that it is designed for separate models for individual tickers, I.E. "relative_price_change_linear_model_XOM.pkl" is a linear model that predicts relative change in price unique to the ticker "XOM".

# Retrieving Historical Data

This section is intended for people who need to retrieve gargantuan amounts of data to build Machine Learning / Deep Learning models. Retrieving moderate amounts of historical data is done automatically for backtesting strategies.

To retrieve extremely large quantities of historical data that IBKR does not directly support, view models/relative_price_change/get_historical_data.py. Data can be mined using get_months_of_historical_data (for intervals >= 1 Minute) or get_days_of_historical_data (for intervals < 1 Minute). This process will make repeated calls to IBKR and combine the retrieved data into one giant file.

# Contact Me

If you have any questions, suggestions, or want to discuss ideas about improving or extending the program, feel free to reach out to me:

- Email: nikita.olechko@gmail.com
- LinkedIn: https://www.linkedin.com/in/nikitaolechko/
- GitHub: https://github.com/nikita-olechko

# Disclaimer

Use this bot at your own risk. This bot is a proof-of-concept and is not meant to be used for live trading without careful review and understanding of its risks and limitations. Financial trading involves substantial risk and should not be done without careful analysis and observation. Any financial numbers referenced here, or on any of my sites and projects, are simply estimates or projections and should not be considered exact, actual, or any form of guarantee of potential earnings. I am NOT an accredited financial advisor. This is a PERSONAL project and is NOT meant to be used for commercial purposes. I am NOT responsible for any losses you may incur by using this bot and offer no guarantees of any kind, including but not limited to strategy profitability, strategy accuracy, software functionality, and software security.
