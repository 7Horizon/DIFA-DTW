import yfinance as yf
import backtrader as bt
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# ETF-Daten abrufen und vorbereiten
def fetch_etf_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"Keine Daten für {ticker} im angegebenen Zeitraum gefunden.")
    # Backtrader benötigt spezifische Spaltennamen
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    data.columns = ['open', 'high', 'low', 'close', 'volume']
    data['openinterest'] = 0  # Backtrader benötigt diese Spalte
    return data


# Strategien mit Signalzählern
#Standard Moving Average
class SmaStrategy(bt.Strategy):
    params = (("short_sma", 10), ("long_sma", 50))

    def __init__(self):
        self.short_sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.short_sma)
        self.long_sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.long_sma)
        self.buy_signals = 0
        self.sell_signals = 0

    def next(self):
        if self.short_sma > self.long_sma and not self.position:
            self.buy()
            self.buy_signals += 1
        elif self.short_sma < self.long_sma and self.position:
            self.sell()
            self.sell_signals += 1

# Linear Weighted Moving Average
class LwmaStrategy(bt.Strategy):
    params = (("short_lwma", 10), ("long_lwma", 50))

    def __init__(self):
        self.short_lwma = bt.indicators.WeightedMovingAverage(self.data.close, period=self.params.short_lwma)
        self.long_lwma = bt.indicators.WeightedMovingAverage(self.data.close, period=self.params.long_lwma)
        self.buy_signals = 0
        self.sell_signals = 0

    def next(self):
        if self.short_lwma > self.long_lwma and not self.position:
            self.buy()
            self.buy_signals += 1
        elif self.short_lwma < self.long_lwma and self.position:
            self.sell()
            self.sell_signals += 1

# Exponential Weighted Moving Average
class EwmaStrategy(bt.Strategy):
    params = (("short_ewma", 10), ("long_ewma", 50))

    def __init__(self):
        self.short_ewma = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.short_ewma)
        self.long_ewma = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.long_ewma)
        self.buy_signals = 0
        self.sell_signals = 0

    def next(self):
        if self.short_ewma > self.long_ewma and not self.position:
            self.buy()
            self.buy_signals += 1
        elif self.short_ewma < self.long_ewma and self.position:
            self.sell()
            self.sell_signals += 1

# Moving Average Crossover Divergence
class MacdStrategy(bt.Strategy):
    params = (("fast_period", 12), ("slow_period", 26), ("signal_period", 9))

    def __init__(self):
        self.macd = bt.indicators.MACD(self.data.close, period_me1=self.params.fast_period, period_me2=self.params.slow_period, period_signal=self.params.signal_period)
        self.buy_signals = 0
        self.sell_signals = 0

    def next(self):
        if self.macd.macd > self.macd.signal and not self.position:
            self.buy()
            self.buy_signals += 1
        elif self.macd.macd < self.macd.signal and self.position:
            self.sell()
            self.sell_signals += 1

# Relative Strength Index (RSI)
class RsiStrategy(bt.Strategy):
    params = (("period", 14), ("overbought", 70), ("oversold", 30))

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.period)
        self.buy_signals = 0
        self.sell_signals = 0

    def next(self):
        if self.rsi < self.params.oversold and not self.position:
            self.buy()
            self.buy_signals += 1
        elif self.rsi > self.params.overbought and self.position:
            self.sell()
            self.sell_signals += 1


# Bollinger Bands
class BollingerBandsStrategy(bt.Strategy):
    params = (("period", 20), ("devfactor", 2.0))

    def __init__(self):
        self.bb = bt.indicators.BollingerBands(self.data.close, period=self.params.period, devfactor=self.params.devfactor)
        self.buy_signals = 0
        self.sell_signals = 0

    def next(self):
        if self.data.close < self.bb.bot and not self.position:
            self.buy()
            self.buy_signals += 1
        elif self.data.close > self.bb.top and self.position:
            self.sell()
            self.sell_signals += 1

# Momentum
class MomentumStrategy(bt.Strategy):
    params = (("period", 12),)

    def __init__(self):
        self.momentum = bt.indicators.Momentum(self.data.close, period=self.params.period)
        self.buy_signals = 0
        self.sell_signals = 0

    def next(self):
        if self.momentum > 1.0 and not self.position:
            self.buy()
            self.buy_signals += 1
        elif self.momentum < 1.0 and self.position:
            self.sell()
            self.sell_signals += 1

# Price Rate-of-Change (ROC)
class RoCStrategy(bt.Strategy):
    params = (("period", 12),)

    def __init__(self):
        self.roc = bt.indicators.RateOfChange(self.data.close, period=self.params.period)
        self.buy_signals = 0
        self.sell_signals = 0

    def next(self):
        if self.roc > 0 and not self.position:
            self.buy()
            self.buy_signals += 1
        elif self.roc < 0 and self.position:
            self.sell()
            self.sell_signals += 1

# Backtest-Funktion mit Signalzählung
def backtest_and_collect(data, strategies, seed_capital, commission):
    earnings = []
    for strategy, strategy_name in strategies:
        data_feed = bt.feeds.PandasData(dataname=data)
        cerebro = bt.Cerebro()
        cerebro.adddata(data_feed)
        cerebro.addstrategy(strategy)
        cerebro.broker.setcash(seed_capital)
        cerebro.broker.setcommission(commission)
        strategies_run = cerebro.run()
        final_value = cerebro.broker.getvalue()

        # Signale sammeln
        strat_instance = strategies_run[0]
        buy_signals = getattr(strat_instance, "buy_signals", 0)
        sell_signals = getattr(strat_instance, "sell_signals", 0)

        earnings.append({
            "Strategy": strategy_name,
            "Earnings": final_value,
            "Seed Capital": seed_capital,
            "Profit": final_value - seed_capital,
            "Profit_Percent": (final_value / seed_capital * 100) - 100,
            "Buy Signals": buy_signals,
            "Sell Signals": sell_signals,
        })
    return earnings

# Parameter
etfs = {
    "SMI ETF": "SMI",
    "DAX ETF": "EXS1.DE",
}
start_date = "2015-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")
seed_capital = 100000
commission = 0.001

# Strategien definieren
strategies = [
    (SmaStrategy, "SMA"),
    (LwmaStrategy, "LWMA"),
    (EwmaStrategy, "EWMA"),
    (MacdStrategy, "MACD"),
    (RsiStrategy, "RSI"),
    (BollingerBandsStrategy, "Bollinger Bands"),
    (MomentumStrategy, "Momentum"),
    (RoCStrategy, "Rate of Change"),
]



# Ergebnisse anzeigen
earnings = []
for name, ticker in etfs.items():
    data = fetch_etf_data(ticker, start_date, end_date)
    data.index = data.index.tz_localize(None)
    etf_earnings = backtest_and_collect(data, strategies, seed_capital, commission)
    for result in etf_earnings:
        result["ETF"] = name
    earnings.extend(etf_earnings)

# DataFrame erstellen
earnings_df = pd.DataFrame(earnings)

# Spalten sortieren
earnings_df = earnings_df[["ETF", "Strategy", "Earnings", "Seed Capital", "Profit", "Profit_Percent", "Buy Signals", "Sell Signals"]]

# Zahlen runden
earnings_df[["Earnings", "Profit", "Profit_Percent"]] = earnings_df[["Earnings", "Profit", "Profit_Percent"]].round(2)

# Nach Strategie sortieren
earnings_df = earnings_df.sort_values(by=["Strategy", "ETF"]).reset_index(drop=True)

# DataFrame anzeigen
print(earnings_df)


# Tabelle mit Matplotlib darstellen
def display_table_with_matplotlib(df):
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.5))  # Dynamische Höhe basierend auf Zeilenanzahl
    ax.axis('tight')
    ax.axis('off')

    # Tabelle hinzufügen
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')

    # Stil anpassen
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # Optional: Farben anpassen
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # Header-Zeilen
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4CAF50')  # Grüner Header
        elif key[0] % 2 == 0:  # Alternierende Zeilen
            cell.set_facecolor('#f9f9f9')  # Hellgrau
        else:
            cell.set_facecolor('#ffffff')  # Weiß

    plt.show()


# Tabelle darstellen
display_table_with_matplotlib(earnings_df)
