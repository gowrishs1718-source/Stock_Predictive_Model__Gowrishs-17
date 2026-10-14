# Stock Price Prediction Using Deep Learning

A student-friendly machine-learning project that analyses historical stock-market data and displays stock statistics, moving averages, and model predictions through a Flask web application.

> **Project type:** Academic / BCA project  
> **Primary language:** Python  
> **Repository:** https://github.com/gowrishs1718-source/Stock_Predictive_Model__Gowrishs-17

---

## Abstract

The stock market is influenced by many factors, making future stock prices difficult to predict accurately. This project demonstrates how historical stock-price data and a deep-learning model can be used to estimate stock prices and visualise market trends.

The application is built with Python and Flask. It retrieves historical market data through Yahoo Finance, processes closing-price data, prepares sequential input windows, and passes them to a pre-trained deep-learning model stored in `stock_dl_model.h5`. The application displays the latest available closing price, a model-generated predicted value, basic price statistics, Exponential Moving Average (EMA) charts, and a comparison of actual and predicted values. It can also export the retrieved dataset as a CSV file.

This project is intended for learning and demonstration. Stock prices are uncertain, and the displayed prediction must not be treated as a guarantee or as financial advice.

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Problem Statement](#2-problem-statement)
- [3. Objectives](#3-objectives)
- [4. Scope](#4-scope)
- [5. Technologies Used](#5-technologies-used)
- [6. System Requirements](#6-system-requirements)
- [7. System Architecture](#7-system-architecture)
- [8. Working Methodology](#8-working-methodology)
- [9. Repository Structure](#9-repository-structure)
- [10. Installation and Setup](#10-installation-and-setup)
- [11. How to Run the Project](#11-how-to-run-the-project)
- [12. How to Use the Application](#12-how-to-use-the-application)
- [13. Main Modules Explained](#13-main-modules-explained)
- [14. Output and Testing](#14-output-and-testing)
- [15. Limitations](#15-limitations)
- [16. Future Enhancements](#16-future-enhancements)
- [17. Troubleshooting](#17-troubleshooting)
- [18. Conclusion](#18-conclusion)
- [19. Disclaimer](#19-disclaimer)

---

## 1. Introduction

Stock-price prediction is a time-series analysis problem. A time series is a collection of observations recorded over time, such as a stock's closing price on each trading day.

This project combines historical stock data, data preprocessing, a pre-trained deep-learning model, and a web interface. A user enters a stock ticker, and the application attempts to retrieve its historical data and show charts and a prediction.

The current implementation uses Flask for the web application, `yfinance` to retrieve market data, Pandas and NumPy for data processing, scikit-learn's `MinMaxScaler` for scaling, Matplotlib for charts, and a saved Keras-compatible model for predictions.

## 2. Problem Statement

Understanding historical price movements manually can be time-consuming. The project aims to provide a simple interface that summarises historical stock prices, visualises selected moving averages, and demonstrates model-based price estimation.

## 3. Objectives

1. Retrieve historical stock-market data using a stock ticker.
2. display basic statistics such as latest, highest, lowest, and average closing prices.
3. Calculate and plot 20-, 50-, 100-, and 200-period Exponential Moving Averages.
4. Prepare historical closing prices for a sequence-based deep-learning model.
5. Display predicted and actual values for visual comparison.
6. Allow the user to download retrieved stock data in CSV format.
7. Demonstrate the integration of Python, machine learning, data analysis, and web development.

## 4. Scope

The project is an educational prototype. It can be used to learn about:

- Python web development with Flask.
- Retrieving historical market data.
- Data preparation and normalisation.
- Time-series prediction using a pre-trained deep-learning model.
- Moving averages and data visualisation.
- Exporting tabular data as CSV.

The repository includes a sample Power Grid dataset and a saved model. Although the interface accepts ticker symbols, a successful prediction depends on valid downloaded data and compatibility between the saved model and the input data. Do not assume that the model has been trained for every possible stock unless this has been verified.

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Flask | Web server, routes, form handling, and page rendering |
| HTML templates | User interface rendered by Flask |
| TensorFlow / `tf_keras` | Loading and running the saved deep-learning model |
| NumPy | Numerical operations and sequence-array preparation |
| Pandas | Historical data processing and descriptive statistics |
| yfinance | Retrieving historical market data from Yahoo Finance |
| scikit-learn | Min-max scaling of price values |
| Matplotlib | Generating price and prediction charts |
| Jupyter Notebook | Exploring or demonstrating the prediction workflow |
| CSV | Downloadable historical data format |

**Note:** Check the notebook and installed package versions before documenting a specific model architecture, training configuration, or measured accuracy. The existence of a saved `.h5` file alone does not establish those details.

## 6. System Requirements

### Hardware

- A computer capable of running Python.
- Internet access to retrieve stock data.
- Enough memory to load the Python libraries and model.

### Software

- Windows, macOS, or Linux.
- Python version compatible with the installed TensorFlow/Keras dependencies.
- Git (optional, for cloning the repository).
- A terminal or Visual Studio Code.
- A modern web browser.

**Compatibility tip:** TensorFlow and Keras versions must be compatible with the Python version and with the model file. If model-loading errors occur, use the project's known working dependency versions rather than upgrading every package at once.

## 7. System Architecture

```text
User
 |
 | Enters a stock ticker in the browser
 v
Flask Web Application (app.py)
 |
 +----> Yahoo Finance data retrieval (yfinance)
 |
 +----> Historical data and basic statistics (Pandas)
 |
 +----> EMA calculations and charts (Pandas + Matplotlib)
 |
 +----> Scale prices and create sequential input windows
 |             |
 |             v
 |      Saved deep-learning model
 |      (stock_dl_model.h5)
 |             |
 |             v
 |      Predicted values
 |
 +----> Save charts and CSV under static/
 |
 v
HTML template renders statistics, charts, errors, and download link
 |
 v
User views the results in the browser
```

## 8. Working Methodology

### Step 1: Input

The user enters a ticker symbol in the web interface. The application converts the symbol to uppercase and removes surrounding whitespace. If no symbol is supplied, the code uses `POWERGRID.NS` as the default.

### Step 2: Data collection

The application requests historical data from Yahoo Finance, starting from 1 January 2000 and ending at the current date. Downloaded data commonly includes Open, High, Low, Close, and Volume columns.

A network connection and a valid supported ticker are required. If the returned dataset is empty, the application displays an invalid-symbol error.

### Step 3: Exploratory statistics

The program calculates:

- Latest available closing price.
- Highest recorded high price in the downloaded dataset.
- Lowest recorded low price in the downloaded dataset.
- Average closing price.
- Descriptive statistics for the downloaded data.

### Step 4: Exponential Moving Averages

The application calculates EMA values for spans of 20, 50, 100, and 200 observations.

An EMA gives more weight to recent prices than to older prices. The general recurrence is:

\[
EMA_t=\alpha P_t+(1-\alpha)EMA_{t-1}
\]

where:

- \(P_t\) is the price at time \(t\).
- \(\alpha=\frac{2}{N+1}\).
- \(N\) is the selected EMA span.

These lines help visualise historical trends. They do not guarantee future price direction.

### Step 5: Prepare the model input

The current code separates the closing-price series into an initial 70% training portion and a remaining portion for evaluation. It uses the last 100 training observations along with the later observations to construct input sequences.

Each sequence contains 100 consecutive observations and is supplied to the model to generate a prediction. The model's expected input shape and scaling procedure must match the saved model's training process.

**Implementation note:** The current application fits a scaler to the training data and then fits it again on the combined sequence data. Re-fitting a scaler changes its parameters and may make inverse scaling or evaluation inconsistent with the model's original training procedure. For a rigorous evaluation, fit the scaler only on training data, transform later data with that same scaler, and inverse-transform predictions correctly.

### Step 6: Generate predictions

The saved model is loaded from `stock_dl_model.h5` when the Flask application starts. The app passes the prepared sequences to the model and derives a displayed predicted value from the final prediction output.

The precise forecasting horizon and model architecture should be confirmed from the notebook and the original training code before making stronger claims about them.

### Step 7: Visualisation

The application generates three charts:

1. Closing price with EMA 20 and EMA 50.
2. Closing price with EMA 100 and EMA 200.
3. Actual and model-predicted values.

The charts are saved under the `static/` directory so the HTML page can display them.

### Step 8: Dataset export

The downloaded historical dataset is saved as a CSV file in `static/`. The interface provides a link to download the file.

## 9. Repository Structure

The repository currently contains the following main files and directories:

```text
Stock_Predictive_Model__Gowrishs-17/
├── app.py                         # Flask application and prediction workflow
├── Stock Price Prediction .ipynb  # Notebook for project exploration/workflow
├── powergrid.csv                  # Sample stock dataset
├── stock_dl_model.h5              # Saved deep-learning model
├── templates/                     # HTML templates
├── static/                        # Static assets and generated charts/CSV
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

The `static/` directory may also receive generated charts and CSV files when the application runs. Exact files inside `templates/` and `static/` may vary as the project develops.

## 10. Installation and Setup

### Step 1: Install Python

Install a Python version supported by the TensorFlow/Keras dependencies used by this project. Verify the installation:

```bash
python --version
```

On some Windows installations, this command may work instead:

```bash
py --version
```

### Step 2: Clone the repository

Open Command Prompt or the VS Code terminal and run:

```bash
git clone https://github.com/gowrishs1718-source/Stock_Predictive_Model__Gowrishs-17.git
cd Stock_Predictive_Model__Gowrishs-17
```

Alternatively, download the repository as a ZIP file from GitHub and extract it.

### Step 3: Create a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

A virtual environment keeps this project's Python packages separate from other projects.

### Step 4: Install dependencies

The repository does not currently list a `requirements.txt` in its root. Install the dependencies in your virtual environment, using versions compatible with the saved model:

```bash
python -m pip install --upgrade pip
python -m pip install flask numpy pandas matplotlib yfinance scikit-learn tensorflow tf-keras
```

If installation fails or the model cannot be loaded, check your Python version and TensorFlow/Keras compatibility. The application sets `TF_USE_LEGACY_KERAS=1` and imports `load_model` from `tf_keras`, so a compatible `tf-keras` installation is particularly important.

For reproducible setup, test the working environment and save its exact versions:

```bash
python -m pip freeze > requirements.txt
```

Commit the resulting `requirements.txt` only after testing it in a clean environment.

## 11. How to Run the Project

1. Open the project folder in VS Code.
2. Activate the virtual environment.
3. Confirm that `app.py`, `templates/`, and `stock_dl_model.h5` are present.
4. Start the application:

   ```bash
   python app.py
   ```

5. Wait for Flask to print the local server address in the terminal. The code uses Flask's default development server, normally at:

   ```text
   http://127.0.0.1:5000
   ```

6. Open that address in your browser.
7. Enter a supported stock ticker and submit the form.
8. Review the returned statistics and charts.

Stop the server with `Ctrl+C` in the terminal.

**Important:** `app.py` loads the model at startup. If the model file is missing or incompatible, the server may fail before the page opens.

## 12. How to Use the Application

1. Open the application in your browser.
2. Enter a stock ticker supported by Yahoo Finance. For example, `POWERGRID.NS` is the default ticker in the current code.
3. Submit the form and wait for data retrieval and chart generation.
4. Review the latest price and the other displayed statistics.
5. Examine the EMA charts and the actual-versus-predicted chart.
6. Use the dataset link to download the CSV, if available.
7. If an error appears, verify the ticker, internet connection, data availability, and model compatibility.

Ticker symbols and market suffixes differ by exchange. Confirm the correct symbol on Yahoo Finance before using another stock.

## 13. Main Modules Explained

### `app.py`

The main Flask application. It:

- Defines the home route (`/`) and download route.
- Receives the stock ticker from the form.
- Downloads historical prices.
- Calculates statistics and EMA series.
- Prepares the data and calls the saved model.
- Saves charts and a CSV file.
- Passes results or error messages to the HTML template.

### `templates/`

Contains the HTML template(s) used by Flask to render the user interface. The main route renders `index.html`, so confirm that `templates/index.html` exists.

### `static/`

Stores static files and generated output, including the chart images and downloaded CSV files.

### `Stock Price Prediction .ipynb`

A Jupyter Notebook associated with the project. Open it in Jupyter Notebook or VS Code with the Jupyter extension. Read the cells from top to bottom and run them in order. If the notebook trains or evaluates a model, its outputs can help verify the model methodology and reported metrics.

### `powergrid.csv`

A sample stock dataset committed with the repository. Inspect its column names and date range before using it in any separate training or evaluation workflow.

### `stock_dl_model.h5`

A saved model file loaded by the Flask application. It is a binary model artifact, not a Python source file. The model's architecture, training history, input shape, and accuracy should be documented only after checking the original training code and evaluation results.

## 14. Output and Testing

### Expected output

When data retrieval and model inference succeed, the application is designed to display:

- The selected ticker.
- Latest closing price.
- Highest high, lowest low, and average close.
- Descriptive statistics.
- EMA charts.
- Actual-versus-predicted chart.
- A link to download the historical dataset as CSV.

The exact results depend on the ticker, the data returned by Yahoo Finance, and the saved model.

### Suggested test cases

| Test | Input / condition | Expected behaviour |
|---|---|---|
| Default ticker | Submit without a ticker | Uses `POWERGRID.NS` as fallback |
| Valid ticker | Enter a supported ticker | Attempts to retrieve data and show results |
| Invalid ticker | Enter an unknown ticker | Displays an invalid-symbol message if data is empty |
| Network failure | Disconnect from the internet | Shows an error or data-fetch failure |
| Missing model | Temporarily remove the model file in a test copy | Application fails during model loading |
| Dataset download | Submit a valid ticker and use the CSV link | Downloads the generated CSV if it exists |
| Chart generation | Submit a ticker with sufficient data | Attempts to create and display the three charts |

Record actual test results and screenshots when preparing a college submission. Do not claim that a test passed unless you have run it.

### Evaluation metrics

For a stronger academic report, evaluate predictions against a held-out test set. Useful regression metrics include:

- **MAE (Mean Absolute Error):** average absolute difference between actual and predicted values.
- **RMSE (Root Mean Squared Error):** square root of the average squared prediction error.
- **MAPE (Mean Absolute Percentage Error):** average absolute percentage error; it can be unsuitable when actual values are zero or very close to zero.

The current web application displays a prediction comparison chart but does not, by itself, establish a validated accuracy score. Calculate metrics on a properly held-out dataset before reporting them.

## 15. Limitations

1. Stock prices are affected by news, market sentiment, economic events, and many other variables not represented by historical closing prices alone.
2. Predictions depend on the data provider, ticker validity, network availability, and the saved model.
3. A pre-trained model may not generalise to every stock symbol accepted by the form.
4. The exact model architecture and training procedure should be verified from the notebook or original training script.
5. The scaler is re-fitted on combined data in the current implementation, which can lead to inconsistent evaluation and scaling.
6. The Flask development server is not intended for production deployment.
7. The current code enables `debug=True`; this should be disabled before any public deployment.
8. A predicted value is an estimate, not a guarantee of future market price.

## 16. Future Enhancements

- Add a tested `requirements.txt` with pinned package versions.
- Document the model architecture, training dataset, and training parameters.
- Use a consistent scaler fitted only on training data.
- Add MAE, RMSE, and other appropriate evaluation metrics.
- Separate data loading, preprocessing, inference, and chart generation into modules.
- Add unit tests and clearer logging.
- Improve input validation and user-friendly error handling.
- Add date-range selection and more informative interactive charts.
- Add a reproducible training pipeline and model-version information.
- Deploy with a production-ready WSGI server and disable debug mode.
- Add a clear notice explaining that predictions are educational and not financial advice.

## 17. Troubleshooting

### `ModuleNotFoundError`

Activate the virtual environment and install the missing package:

```bash
python -m pip install <package-name>
```

Replace `<package-name>` with the package named in the error.

### Model-loading or Keras compatibility error

- Check that `stock_dl_model.h5` is in the repository root.
- Check the Python, TensorFlow, and `tf-keras` versions.
- Use the same compatible Keras environment that was used to save the model.
- Avoid randomly upgrading or downgrading packages without testing.

### `TemplateNotFound: index.html`

Confirm that the file exists at:

```text
templates/index.html
```

Flask expects templates inside the `templates` directory by default.

### Invalid stock symbol or empty data

- Verify the ticker on Yahoo Finance.
- Check the internet connection.
- Try a known supported symbol such as `POWERGRID.NS`.
- Remember that market holidays, data-provider changes, or unavailable history can affect results.

### Port already in use

Another process may be using Flask's default port. Stop that process or configure a different port in the development environment.

### Chart images do not appear

Confirm that the `static/` directory exists, that the application can write files there, and that the template uses the correct paths for the generated images.

## 18. Conclusion

The Stock Price Prediction project demonstrates a practical combination of Python programming, data analysis, deep learning, and web development. It retrieves historical market data, calculates descriptive statistics and moving averages, generates predictions using a saved model, and presents the results through a browser-based interface.

As a student project, it provides a foundation for learning a complete machine-learning application workflow. The next step toward a more reliable and portfolio-ready project is to document the training process, validate the model with appropriate test data, correct the scaling workflow, add reproducible dependencies, and include verified screenshots and evaluation metrics.

## 19. Disclaimer

This project is for educational and demonstration purposes only. It is not financial advice and should not be used as the sole basis for buying, selling, or holding any security. Historical data and model predictions do not guarantee future performance.

---

## Academic Project Details

Complete these details before submitting the README as part of a college report:

- **Student name:** Gowrish
- **Course:** Bachelor of Computer Applications (BCA)
- **Project title:** Stock Price Prediction Using Deep Learning
- **College / department:** Add your college and department
- **Academic year:** Add your academic year
- **Project guide:** Add your guide's name, if applicable
