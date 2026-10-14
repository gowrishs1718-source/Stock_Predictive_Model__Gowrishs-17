import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tf_keras.models import load_model
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

plt.style.use("fivethirtyeight")

app = Flask(__name__)

os.makedirs("static", exist_ok=True)

model = load_model("stock_dl_model.h5")

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        stock = request.form.get("stock")

        if not stock:
            stock = "POWERGRID.NS"

        stock = stock.upper().strip()

        start = dt.datetime(2000, 1, 1)
        end = dt.datetime.today()

        try:
            df = yf.download(stock, start=start, end=end)

            if df.empty:
                return render_template(
                    "index.html",
                    error="Invalid Stock Symbol! Please enter a valid Yahoo Finance ticker."
                )

            # -----------------------------
            # Basic Statistics
            # -----------------------------
            latest_price = round(float(df["Close"].iloc[-1]), 2)
            highest_price = round(float(df["High"].max()), 2)
            lowest_price = round(float(df["Low"].min()), 2)
            average_price = round(float(df["Close"].mean()), 2)

            data_desc = df.describe()

            # -----------------------------
            # EMA
            # -----------------------------
            ema20 = df.Close.ewm(span=20, adjust=False).mean()
            ema50 = df.Close.ewm(span=50, adjust=False).mean()
            ema100 = df.Close.ewm(span=100, adjust=False).mean()
            ema200 = df.Close.ewm(span=200, adjust=False).mean()

            # -----------------------------
            # Data Split
            # -----------------------------
            data_training = pd.DataFrame(df["Close"][: int(len(df) * 0.70)])
            data_testing = pd.DataFrame(df["Close"][int(len(df) * 0.70):])

            scaler = MinMaxScaler(feature_range=(0, 1))

            data_training_array = scaler.fit_transform(data_training)

            past_100_days = data_training.tail(100)

            final_df = pd.concat(
                [past_100_days, data_testing],
                ignore_index=True
            )

            input_data = scaler.fit_transform(final_df)

            x_test = []
            y_test = []

            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i - 100:i])
                y_test.append(input_data[i, 0])

            x_test = np.array(x_test)
            y_test = np.array(y_test)

            # -----------------------------
            # Prediction
            # -----------------------------
            y_predicted = model.predict(x_test, verbose=0)

            scale_factor = 1 / scaler.scale_[0]

            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor

            # Latest predicted price
            predicted_price = round(float(y_predicted[-1][0]), 2)

            # Difference between predicted and current price
            prediction_difference = round(predicted_price - latest_price, 2)
            # -----------------------------
            # Graph 1
            # -----------------------------
            fig1, ax = plt.subplots(figsize=(12, 6))

            ax.plot(df.Close, label="Closing Price")
            ax.plot(ema20, label="EMA 20")
            ax.plot(ema50, label="EMA 50")

            ax.set_title("Closing Price vs EMA (20 & 50)")
            ax.legend()

            ema_20_50 = "static/ema_20_50.png"

            fig1.savefig(ema_20_50, bbox_inches="tight")
            plt.close(fig1)

            # -----------------------------
            # Graph 2
            # -----------------------------
            fig2, ax = plt.subplots(figsize=(12, 6))

            ax.plot(df.Close, label="Closing Price")
            ax.plot(ema100, label="EMA 100")
            ax.plot(ema200, label="EMA 200")

            ax.set_title("Closing Price vs EMA (100 & 200)")
            ax.legend()

            ema_100_200 = "static/ema_100_200.png"

            fig2.savefig(ema_100_200, bbox_inches="tight")
            plt.close(fig2)

            # -----------------------------
            # Graph 3
            # -----------------------------
            fig3, ax = plt.subplots(figsize=(12, 6))

            ax.plot(y_test, label="Actual Price")
            ax.plot(y_predicted, label="Predicted Price")

            ax.set_title("Prediction vs Actual")

            ax.legend()

            prediction_graph = "static/stock_prediction.png"

            fig3.savefig(prediction_graph, bbox_inches="tight")
            plt.close(fig3)

            # -----------------------------
            # CSV
            # -----------------------------
            csv_file = f"static/{stock}_dataset.csv"

            df.to_csv(csv_file)

            return render_template(
                "index.html",
                stock=stock,
                latest_price=latest_price,
                predicted_price=predicted_price,
                prediction_difference=prediction_difference,
                highest_price=highest_price,
                lowest_price=lowest_price,
                average_price=average_price,
                data_desc=data_desc.to_html(
                    classes="table table-striped table-bordered",
                    border=0
                ),

                plot_path_ema_20_50=ema_20_50,

                plot_path_ema_100_200=ema_100_200,

                plot_path_prediction=prediction_graph,

                dataset_link=csv_file
            )

        except Exception as e:

            return render_template(
                "index.html",
                error=str(e)
            )

    return render_template("index.html")


@app.route("/download/<filename>")
def download_file(filename):

    return send_file(
        f"static/{filename}",
        as_attachment=True
    )


if __name__ == "__main__":
    app.run(debug=True)