import os
from datetime import datetime
from pprint import pprint

from matplotlib import pyplot as plt
from pandas import DataFrame
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler

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

    interval = 30

    df = prepare_df(symbol="ADA", interval=interval)
    df.sort_values(by="timestamp", inplace=True)

    # df = df.head(10000)

    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df)

    last_3_months = int(float(60 / interval * 24 * 90))
    validation_df = df.tail(last_3_months)

    df = df.head(len(df) - last_3_months)

    set_without_target_column_values = df.drop("next_close", axis=1)
    target_column_values = df["next_close"]

    x_train, x_test, y_train, y_test = train_test_split(set_without_target_column_values, target_column_values,
                                                        test_size=0.2, random_state=42)
    print(f"training set size:{len(x_train)}, test set size:{len(x_test)}\n")

    rf_hyperparameters = {'max_depth': range(2, 8), 'max_features': range(2, 10)}

    # Try xgboost.XGBRegressor
    grid_search_cv = GridSearchCV(RandomForestRegressor(n_estimators=100),
                                  rf_hyperparameters,
                                  cv=KFold(5, shuffle=True),
                                  scoring='neg_root_mean_squared_error',
                                  n_jobs=-1,
                                  return_train_score=True,
                                  verbose=2
                                  )

    print("Starting grid search now.\n")
    grid_search_cv.fit(x_train, y_train)

    best_parameters = grid_search_cv.best_params_
    print(f"Best mean squared score:{grid_search_cv.best_score_} with params:{best_parameters}\n")

    validation_df = validation_df.drop("next_close", axis=1)
    validation_df.reset_index(drop=True, inplace=True)

    predicted_prices = grid_search_cv.predict(validation_df)

    # Hard to see the details in this graph, seaborn could provide something better
    plt.plot(validation_df['close'], color='red', label='Real Prices')
    plt.plot(predicted_prices, color='blue', label='Predicted Prices')
    plt.title('Prices Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
