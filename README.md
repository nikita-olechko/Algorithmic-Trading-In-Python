# Live-Algorithmic-Trading-In-Python

This project contains a live trading bot that executes trades on the Interactive Brokers platform. The bot is designed to be highly customizable, allowing you to define your own trading strategies. As an example, the bot is currently set to use a testing strategy based on the 60-period Simple Moving Average (SMA). 

***Please note that this strategy is for testing purposes only and not recommended for live trading. See Full Disclaimer Below.***

# Features

- Connects to Interactive Brokers API for live trading
- Supports customizable strategies
- Built with a modular design for adding more order types and quantities
- Supports limit and stop-loss orders as well as market orders
- Can manage multiple trading bots running different strategies simultaneously (up to 30 trades per second per bot)
- Sample testing strategy based on 60-period SMA

# Requirements

- Python 3.6 or higher
- Interactive Brokers account
- IBKR Trader Workstation installed, open, and running on port 4000
- Install necessary Python packages: ib_insync, pandas, pytz

# Setup and Usage

- Clone the repository to your local machine.
- Install the necessary Python packages using pip:
    pip install ib_insync pandas pytz

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

Orders do NOT stack. Meaning if your strategy places a BUY order, it will not place another BUY order unti a SELL order has been placed. This corresponds with the fixed quantity capacity of the current model.

You can create a new strategy by writing similar functions that align with your trading approach. Of course, any data generated in the first function can and should be used in the second. Future updates will provide more functionality related to the type and quantity of orders placed.

# Simulatenously Running Strategies

To run several strategies simulatenously, you can instantiate new instances of the Bot class, which will run the strategy. I would recommend running them in a seaparate console if you wish to see live updates, otherwise it gets very confusing very fast. 

If running multiple strategies, OR restarting strategies multiple times in one day:

- the Bot class argument twsConnectionID must be a unique integer for each running instance of the bot
- the Bot class argument orderStarterID should be passed the function get_starter_order_id(n) where n is an integer not passed to this function today

# Backtesting

To backtest strategies, use the complementary repository "Backtesting Trading Strategies". Note that although the two repos are directly complementary, "Backtesting Trading Strategies" is written in R, and has some minor tweaks in strategy function requirements.

I am looking into rebuilding the backtesting framework in Python. 

# Contact Me

If you have any questions, suggestions, or want to discuss ideas about improving or extending the program, feel free to reach out to me:

- Email: nikita.olechko@gmail.com
- LinkedIn: https://www.linkedin.com/in/nikitaolechko/
- GitHub: https://github.com/nikita-olechko

# Disclaimer

This bot is a proof-of-concept and is not meant to be used for live trading without careful review and enhancements. Financial trading involves substantial risk, and there is always the potential for loss. Your results may vary and depend on many factors, including but not limited to your background, experience, and work ethic. Any financial numbers referenced here, or on any of my sites and projects, are simply estimates or projections and should not be considered exact, actual, or a promise of potential earnings. Use this bot at your own risk. I am NOT an accredited financial advisor.
