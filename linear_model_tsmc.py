import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf
import os
import pandas as pd

model_dir = "linear_model_tsmc"

# 1. 從 Yahoo Finance 獲取台積電的股價數據
def fetch_stock_data(ticker='2330.TW', start='2020-01-01', end=pd.Timestamp.today().strftime('%Y-%m-%d')):
    stock_data = yf.download(ticker, start=start, end=end)
    stock_data = stock_data[['Close']]
    stock_data = stock_data.dropna()
    return stock_data

tsmc_data = fetch_stock_data()
X = np.arange(len(tsmc_data)).reshape(-1, 1)
y = tsmc_data['Close'].values

# 2. 建立 TensorFlow 線性模型
class LinearModel(tf.Module):
    def __init__(self):
        self.W = tf.Variable(tf.random.normal([1]), name='weight')
        self.b = tf.Variable(tf.random.normal([1]), name='bias')

    def __call__(self, X):
        return self.W * X + self.b

# 3. 定義損失函數和訓練過程
@tf.function
def loss_fn(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

@tf.function
def train_step(model, X, y, learning_rate=0.01):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
    gradients = tape.gradient(loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * gradients[0])
    model.b.assign_sub(learning_rate * gradients[1])
    return loss

# 4. 訓練模型
epochs = 5000
learning_rate = 0.0001
linear_model = LinearModel()

X_normalized = (X - np.mean(X)) / np.std(X)
X_tf = tf.constant(X_normalized, dtype=tf.float32)
y_normalized = (y - np.mean(y)) / np.std(y)
y_tf = tf.constant(y_normalized, dtype=tf.float32)

loss_history = []
for epoch in range(epochs):
    loss = train_step(linear_model, X_tf, y_tf, learning_rate)
    loss_history.append(loss)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss: {loss:.4f}")

# 5. 預測與視覺化結果
future_periods = {'5_years': 365 * 5, '3_years': 365 * 3, '1_year': 365}
for label, future_days in future_periods.items():
    plt.figure(figsize=(14, 8))
    plt.plot(tsmc_data.index, tsmc_data['Close'], label='Actual Stock Price', color='blue')
    y_pred = linear_model(X_tf).numpy()
    y_pred_rescaled = y_pred * np.std(y) + np.mean(y)
    plt.plot(tsmc_data.index, y_pred_rescaled, color='red', label='Fitted Line')

    # 預測未來股價
    future_X = np.arange(len(tsmc_data), len(tsmc_data) + future_days).reshape(-1, 1)
    future_X_normalized = (future_X - np.mean(X)) / np.std(X)
    future_X_tf = tf.constant(future_X_normalized, dtype=tf.float32)
    future_y_pred = linear_model(future_X_tf)
    future_y_pred_rescaled = future_y_pred.numpy() * np.std(y) + np.mean(y)

    future_dates = pd.date_range(start=tsmc_data.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='B')
    plt.plot(future_dates, future_y_pred_rescaled, color='green', linestyle='dashed', label='Predicted Future Prices')

    # 標記最高和最低點
    plt.grid(True)
    min_price_idx = np.argmin(tsmc_data['Close'].values)
    max_price_idx = np.argmax(tsmc_data['Close'].values)
    plt.text(tsmc_data.index[min_price_idx], tsmc_data['Close'].iloc[min_price_idx].item(), f"{tsmc_data['Close'].iloc[min_price_idx].item():.2f}", fontsize=12, color='black')
    plt.text(tsmc_data.index[max_price_idx], tsmc_data['Close'].iloc[max_price_idx].item(), f"{tsmc_data['Close'].iloc[max_price_idx].item():.2f}", fontsize=12, color='black')

    min_future_price_idx = np.argmin(future_y_pred_rescaled)
    max_future_price_idx = np.argmax(future_y_pred_rescaled)
    plt.text(future_dates[min_future_price_idx], future_y_pred_rescaled[min_future_price_idx].item(), f"{future_y_pred_rescaled[min_future_price_idx].item():.2f}", fontsize=12, color='black')
    plt.text(future_dates[max_future_price_idx], future_y_pred_rescaled[max_future_price_idx].item(), f"{future_y_pred_rescaled[max_future_price_idx].item():.2f}", fontsize=12, color='black')

    # 標記最後的價格和日期
    last_price = tsmc_data['Close'].iloc[-1].item()
    last_date = tsmc_data.index[-1]
    plt.text(last_date, last_price, f"{last_date.strftime('%Y-%m-%d')}: {last_price:.2f}", fontsize=12, color='black', ha='left', va='top')

    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'TSMC Stock Price Trend and {label.replace("_", " ").title()} Prediction')
    plt.legend()
    plt.xticks(rotation=45)
    os.makedirs(model_dir, exist_ok=True)
    plt.savefig(os.path.join(model_dir, f'tsmc_{label}_stock_price_prediction.png'))

# 6. 輸出模型參數
print(f"Trained Weight: {linear_model.W.numpy()[0]:.4f}")
print(f"Trained Bias: {linear_model.b.numpy()[0]:.4f}")

# 7. 存儲模型參數
os.makedirs(model_dir, exist_ok=True)
np.save(os.path.join(model_dir, "weight.npy"), linear_model.W.numpy())
np.save(os.path.join(model_dir, "bias.npy"), linear_model.b.numpy())
print(f"Model saved in directory: {model_dir}")
