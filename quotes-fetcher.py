from datetime import datetime
from pprint import pprint

import requests
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


BINANCE_URL = "https://api1.binance.com/api/v3/klines?symbol=$SYMBOLUSDT&interval=$INTERVALm&limit=1000&startTime=$START_TIMESTAMP"


connection_url = 'postgresql://postgres:postgres123@localhost:5429/quotes'
db_pool = ConnectionPool(connection_url, min_size=4, max_size=20, kwargs={"row_factory": dict_row, "autocommit": True})


def save_quotes(quotes: list[tuple]) -> None:
    with db_pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO quotes("close", high, "interval", low, "open", "timestamp", symbol) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)                
                ON CONFLICT DO NOTHING
                """, quotes)


if __name__ == '__main__':
    intervals = [15]
    symbols = ["ADA"]
    from_timestamp = 0

    for symbol in symbols:
        for interval in intervals:
            has_finished = False
            while not has_finished:
                filled_url = (BINANCE_URL.replace("$SYMBOL", symbol)
                              .replace("$INTERVAL", str(interval))
                              .replace("$START_TIMESTAMP", str(from_timestamp)))

                binance_quotes = requests.get(filled_url).json()

                if len(binance_quotes) < 2:
                    has_finished = True
                    continue
                else:
                    print(f"Fetched {len(binance_quotes)} quotes from:{datetime.fromtimestamp(from_timestamp / 1000)}")

                quotes = [(quote[4], quote[2], interval, quote[3], quote[1], quote[0] / 1000, symbol)
                          for quote in binance_quotes]
                save_quotes(quotes)

                from_timestamp = binance_quotes[-1][0]
