import os
from datetime import datetime

from pandas import DataFrame
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

connection_url = 'postgresql://postgres:postgres123@localhost:5429/quotes'
db_pool = ConnectionPool(connection_url, min_size=4, max_size=20, kwargs={"row_factory": dict_row, "autocommit": True})

RESULTS_PATH = f"results/{datetime.now().strftime("%x-%X")}"


def load_quotes(symbol, interval):
    with db_pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
            select close, high, low, open, timestamp from quotes 
            where symbol = %s and interval = %s order by timestamp asc
            """, (symbol, interval))
            return cursor.fetchall()


def prepare_df(symbol, interval):
    quotes = load_quotes(symbol, interval)
    df = DataFrame(quotes)
    # df = DataFrame(quotes)

    return df.assign(next_close=df.close.shift(-1)).dropna(axis=0)


if __name__ == '__main__':
    print("Starting training script.\n")
    # os.mkdir(RESULTS_PATH)

    df = prepare_df(symbol="ADA", interval=30)

    df = df.head(1000)

    set_without_target_column_values = df.drop("next_close", axis=1)
    target_column_values = df["next_close"]

    x_train, x_test, y_train, y_test = train_test_split(set_without_target_column_values, target_column_values,
                                                        test_size=0.2, random_state=42)
    print(f"training set size:{len(x_train)}, test set size:{len(x_test)}\n")

    rf_hyperparameters = {'max_depth': range(2, 8), 'max_features': range(2, 10)}

    grid_search_cv = GridSearchCV(RandomForestRegressor(n_estimators=300), rf_hyperparameters,
                                  cv=KFold(2, shuffle=True), scoring='neg_root_mean_squared_error',
                                  n_jobs=-1, return_train_score=True)

    print("Starting grid search now.\n")
    grid_search_cv.fit(x_train, y_train)

    best_parameters = grid_search_cv.best_params_
    print(f"Best mean squared score:{grid_search_cv.best_score_} with params:{best_parameters}\n")
