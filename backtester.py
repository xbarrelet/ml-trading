import time
from datetime import datetime
from pprint import pprint
import pandas_ta as ta

from backtesting import Backtest
from pandas import DataFrame
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from strategies.SmaCross import SmaCross
from strategies.SupportResRSIStrat import SupportResRSIStrat

connection_url = 'postgresql://postgres:postgres123@localhost:5429/quotes'
db_pool = ConnectionPool(connection_url, min_size=4, max_size=20, kwargs={"row_factory": dict_row, "autocommit": True})




def load_quotes(symbol, interval):
    with db_pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
            select close, high, low, open, timestamp from quotes 
            where symbol = %s and interval = %s order by timestamp asc
            """, (symbol, interval))
            return cursor.fetchall()


def prepare_quotes(quotes_df):
    quotes_df.sort_values(by="timestamp", inplace=True)

    quotes_df['datetime'] = quotes_df['timestamp'].apply(lambda x: datetime.fromtimestamp(x))
    quotes_df = quotes_df.set_index('datetime')
    quotes_df.drop(columns=["timestamp"], inplace=True)
    quotes_df.dropna(inplace=True)

    for column in quotes_df.columns:
        quotes_df[column] = quotes_df[column].astype(float)

    quotes_df = quotes_df.rename(columns={"close": "Close", "high": "High", "low": "Low", "open": "Open"})
    quotes_df['RSI'] = ta.rsi(quotes_df['Close'], length=12)

    return quotes_df


if __name__ == '__main__':
    # TODO: Not really useful... Wait for the Deep Learning model to be trained to see what is worth doing
    for interval in [
        60,
        # 30,
        # 15,
        # 5,
        # 1
        ]:
        print(f"Starting backtesting with interval:{interval}.\n", )

        start_time = time.time()

        quotes_df = DataFrame(load_quotes("ADA", interval))
        quotes_df = prepare_quotes(quotes_df)

        bt = Backtest(quotes_df, SupportResRSIStrat, cash=10_000, commission=.002)
        stats = bt.optimize(n1=range(1, 10, 5),
                            n2=range(5, 15, 5),
                            maximize='Equity Final [$]',
                            constraint=lambda param: param.n1 < param.n2)
        print(f"\nInterval:{interval}, processing time:{time.time() - start_time}s, results:\n")
        pprint(stats, width=1000)
