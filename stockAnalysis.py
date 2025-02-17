import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Function to fetch stock data
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="10y")  # Use 5 years of data
        if history.empty:
            messagebox.showerror("Error", "No data found for this ticker!")
            return None
        return history
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return None

# Function to prepare data for LSTM
def prepare_data(data, lookback=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']].values)

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM
    return X, y, scaler

# Function to build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to analyze stock using LSTM
def analyze_stock():
    ticker = entry_ticker.get()
    history = fetch_stock_data(ticker)
    if history is None:
        return

    # Prepare data
    lookback = 60  # Use 60 days of historical data
    X, y, scaler = prepare_data(history, lookback)

    # Split data into training and testing sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build and train LSTM model
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    # Predict future prices
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate recommendation
    last_actual_price = actual_prices[-1][0]
    last_predicted_price = predicted_prices[-1][0]
    if last_predicted_price > last_actual_price:
        recommendation = "Buy"
    elif last_predicted_price < last_actual_price:
        recommendation = "Sell"
    else:
        recommendation = "Hold"

    # Display results
    result_text = (
        f"Ticker: {ticker}\n"
        f"Last Actual Price: ${last_actual_price:.2f}\n"
        f"Last Predicted Price: ${last_predicted_price:.2f}\n"
        f"Recommendation: {recommendation}"
    )
    label_result.config(text=result_text)

    # Plot actual vs predicted prices
    plt.figure(figsize=(10, 6))
    plt.plot(actual_prices, label="Actual Prices")
    plt.plot(predicted_prices, label="Predicted Prices")
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

# Create GUI
app = Tk()
app.title("Stock Analysis App with LSTM")
app.geometry("400x300")

# Ticker input
label_ticker = Label(app, text="Enter Stock Ticker:")
label_ticker.pack(pady=10)
entry_ticker = Entry(app)
entry_ticker.pack(pady=5)

# Analyze button
button_analyze = Button(app, text="Analyze Stock", command=analyze_stock)
button_analyze.pack(pady=10)

# Result label
label_result = Label(app, text="", justify=LEFT)
label_result.pack(pady=20)

# Run the app
app.mainloop()