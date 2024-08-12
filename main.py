import os
import shutil

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

connection_url = 'postgresql://postgres:postgres123@localhost:5429/quotes'
db_pool = ConnectionPool(connection_url, min_size=4, max_size=20, kwargs={"row_factory": dict_row, "autocommit": True})

RESULTS_PATH = f"results"


# RESULTS_PATH = f"results/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"


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


def generate_nrmse_plot(nrmse_results):
    nrmse_results = pd.DataFrame(nrmse_results.items(), columns=['interval', 'nrmse_score'])
    nrmse_results.sort_values("nrmse_score", ascending=False, inplace=True)

    nrmse_results_plot = (nrmse_results[["nrmse_score", "interval"]]
                          .plot(kind="bar", x="interval", figsize=(15, 5), rot=0,
                                title="NRMSE score per interval[min]"))

    nrmse_results_plot.title.set_size(20)
    nrmse_results_plot.legend(["NRMSE score"])
    nrmse_results_plot.set(xlabel=None)

    nrmse_results_plot.get_figure().savefig(f"{RESULTS_PATH}/nrmse_scores_per_interval.png")
    plt.close()


def generate_ari_scores_plot(ari_results):
    ari_results = pd.DataFrame(ari_results.items(), columns=['interval', 'ari_score'])
    ari_results.sort_values("ari_score", ascending=False, inplace=True)

    ari_scores_plot = (ari_results[["ari_score", "interval"]]
                       .plot(kind="bar", x="interval", figsize=(15, 5), rot=0,
                             title="ARI score per interval[min]"))

    ari_scores_plot.title.set_size(20)
    ari_scores_plot.legend(["ARI score"])
    ari_scores_plot.set(xlabel=None)

    ari_scores_plot.get_figure().savefig(f"{RESULTS_PATH}/ari_scores_per_interval.png")
    plt.close()


def create_feature_importance_plots(model, keys, interval):
    features = []

    if hasattr(model, 'feature_importances_'):
        sns.set_theme(rc={'figure.figsize': (17, 12)})
        feature_importances = model.feature_importances_

        for result in sorted(zip(feature_importances, keys), reverse=True):
            score = result[0]
            if score > 0.01:
                features.append({"name": result[1], "score": score})

    elif hasattr(model, 'coef_'):
        sns.set_theme(rc={'figure.figsize': (30, 20)})
        importance = model.coef_

        for i, v in enumerate(importance):
            if v != 0:
                features.append({"name": keys[i], "score": v})
    else:
        return

    barplot = sns.barplot(DataFrame(features), x="score", y="name", hue="name", legend=False)
    barplot.set(xlabel=None)
    barplot.set(ylabel=None)
    barplot.get_figure().savefig(f"{RESULTS_PATH}/feature_importance_{interval}.png")
    plt.close()


def generate_predictions_vs_prices_plot(results, interval):
    plt.figure(figsize=(12, 12), dpi=600)

    plot = sns.lineplot(data=results[['next_close', 'predicted_prices']], palette=['red', 'blue'])

    plot.set_title('Prices Prediction')
    plot.legend(['Real Prices', 'Predicted Prices'])

    plot.get_figure().savefig(f"{RESULTS_PATH}/prices_prediction_{interval}.png")
    plt.close()


if __name__ == '__main__':
    print("Starting training script.\n")
    shutil.rmtree(RESULTS_PATH)
    os.makedirs(RESULTS_PATH)

    ari_results = {}
    nrmse_results = {}
    for interval in [
        # 5,
        # 15,
        # 30,
        60
    ]:
        print(f"Starting training for interval:{interval}.\n")
        df = prepare_df(symbol="ADA", interval=interval)

        df.sort_values(by="timestamp", inplace=True)
        df.drop("timestamp", axis=1, inplace=True)

        # output = talib.SMA(close)

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

        # rf_hyperparameters = {'max_depth': [7], 'max_features': [4]}
        rf_hyperparameters = {'max_depth': range(2, 8), 'max_features': range(2, 10)}
        # xgb_hyperparameters = {'max_depth': range(2, 15)}

        # https://github.com/ta-lib/ta-lib-python?tab=readme-ov-file#function-api

        # Try xgboost.XGBRegressor
        # grid_search_cv = GridSearchCV(XGBRegressor(tree_method="hist", n_estimators=100), xgb_hyperparameters,
        grid_search_cv = GridSearchCV(RandomForestRegressor(n_estimators=100), rf_hyperparameters,
                                      cv=2,
                                      scoring='neg_root_mean_squared_error',
                                      n_jobs=-1,
                                      return_train_score=True,
                                      # verbose=2
                                      )

        print("Starting grid search now.\n")
        grid_search_cv.fit(x_train, y_train)

        best_parameters = grid_search_cv.best_params_
        print(f"Best mean squared score:{grid_search_cv.best_score_} with params:{best_parameters}\n")
        nrmse_results[interval] = grid_search_cv.best_score_

        results = DataFrame()

        validation_df.reset_index(drop=True, inplace=True)
        results['next_close'] = validation_df['next_close']

        validation_df = validation_df.drop("next_close", axis=1)

        results['predicted_prices'] = grid_search_cv.predict(validation_df)

        ari_score = adjusted_rand_score(results['next_close'], results['predicted_prices'])
        ari_results[interval] = ari_score
        print(f"ari_score:{ari_score}")

        generate_predictions_vs_prices_plot(results, interval)

        create_feature_importance_plots(grid_search_cv.best_estimator_, x_train.keys(), interval)

        print(f"Finished training for interval:{interval}.\n")

    generate_ari_scores_plot(ari_results)
    generate_nrmse_plot(nrmse_results)
