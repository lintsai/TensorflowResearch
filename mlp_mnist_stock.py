import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, Input, PReLU, ReLU, ELU
from tensorflow.keras.optimizers import Adam, AdamW, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import yfinance as yf
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import logging
from tensorflow.keras import backend as K

# 設定模型保存目錄
model_dir = "mlp_mnist_stock"
os.makedirs(model_dir, exist_ok=True)

# 設定日誌記錄
logging.basicConfig(filename=os.path.join(model_dir, 'training_log.log'), level=logging.INFO, format='%(asctime)s - %(message)s')

# 1. 從 Yahoo Finance 獲取台積電和其他前十大企業的股價數據
def fetch_stock_data(ticker='2330.TW', start='2020-01-01', end=pd.Timestamp.today().strftime('%Y-%m-%d')):
    stock_data = yf.download(ticker, start=start, end=end)
    stock_data = stock_data[['Close']]
    stock_data = stock_data.dropna()
    logging.info(f"Fetched data for {ticker} from {start} to {end}. Number of records: {len(stock_data)}")
    return stock_data

# 台積電股價數據
tsmc_data = fetch_stock_data()
# 其他前十大企業股價數據
foxconn_data = fetch_stock_data('2317.TW')  # 鴻海
cathay_data = fetch_stock_data('2882.TW')   # 國泰金
fubon_data = fetch_stock_data('2881.TW')    # 富邦金
mega_data = fetch_stock_data('2886.TW')     # 兆豐金
china_trust_data = fetch_stock_data('2891.TW')  # 中信金
united_micro_data = fetch_stock_data('2303.TW') # 聯電
asus_data = fetch_stock_data('2357.TW')     # 華碩
mediatek_data = fetch_stock_data('2454.TW') # 聯發科
pegatron_data = fetch_stock_data('4938.TW') # 和碩

# 合併台積電和其他前十大企業的數據
combined_data = pd.concat([tsmc_data, foxconn_data, cathay_data, fubon_data, mega_data, china_trust_data, united_micro_data, asus_data, mediatek_data, pegatron_data], axis=1).ffill()
combined_data.columns = ['TSMC_Close', 'Foxconn_Close', 'Cathay_Close', 'Fubon_Close', 'Mega_Close', 'ChinaTrust_Close', 'UnitedMicro_Close', 'Asus_Close', 'MediaTek_Close', 'Pegatron_Close']

X = np.arange(len(combined_data)).reshape(-1, 1)
y_tsmc = combined_data['TSMC_Close'].values
y_foxconn = combined_data['Foxconn_Close'].values

# 數據預處理
scaler_X = MinMaxScaler()
scaler_y_tsmc = MinMaxScaler()
scaler_y_foxconn = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)
y_tsmc_normalized = scaler_y_tsmc.fit_transform(y_tsmc.reshape(-1, 1)).flatten()
y_foxconn_normalized = scaler_y_foxconn.fit_transform(y_foxconn.reshape(-1, 1)).flatten()

# 建立 MLP 模型的函數
def build_mlp_model(activation_function='PReLU', dropout_rates=[0.5, 0.5, 0.4, 0.3, 0.2], regularization_strength=0.0001):
    model = Sequential()
    model.add(Input(shape=(1,)))
    model.add(Dense(1024, kernel_regularizer=tf.keras.regularizers.l1_l2(regularization_strength)))
    if activation_function == 'PReLU':
        model.add(PReLU())
    elif activation_function == 'LeakyReLU':
        model.add(LeakyReLU())
    elif activation_function == 'ReLU':
        model.add(ReLU())
    elif activation_function == 'ELU':
        model.add(ELU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rates[0]))
    model.add(Dense(512, kernel_regularizer=tf.keras.regularizers.l1_l2(regularization_strength)))
    if activation_function == 'PReLU':
        model.add(PReLU())
    elif activation_function == 'LeakyReLU':
        model.add(LeakyReLU())
    elif activation_function == 'ReLU':
        model.add(ReLU())
    elif activation_function == 'ELU':
        model.add(ELU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rates[1]))
    model.add(Dense(256, kernel_regularizer=tf.keras.regularizers.l1_l2(regularization_strength)))
    if activation_function == 'PReLU':
        model.add(PReLU())
    elif activation_function == 'LeakyReLU':
        model.add(LeakyReLU())
    elif activation_function == 'ReLU':
        model.add(ReLU())
    elif activation_function == 'ELU':
        model.add(ELU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rates[2]))
    model.add(Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(regularization_strength)))
    if activation_function == 'PReLU':
        model.add(PReLU())
    elif activation_function == 'LeakyReLU':
        model.add(LeakyReLU())
    elif activation_function == 'ReLU':
        model.add(ReLU())
    elif activation_function == 'ELU':
        model.add(ELU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rates[3]))
    model.add(Dense(64, kernel_regularizer=tf.keras.regularizers.l1_l2(regularization_strength)))
    if activation_function == 'PReLU':
        model.add(PReLU())
    elif activation_function == 'LeakyReLU':
        model.add(LeakyReLU())
    elif activation_function == 'ReLU':
        model.add(ReLU())
    elif activation_function == 'ELU':
        model.add(ELU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rates[4]))
    model.add(Dense(1))
    return model

# 優化超參數
initial_learning_rate = 0.0005
learning_rate_decay_factor = 0.5
loss_function = tf.keras.losses.MeanSquaredError()

# 訓練模型的參數
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=learning_rate_decay_factor, patience=10, min_lr=1e-6, verbose=1)

# 進行交叉驗證
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
all_histories = []
fold = 1

best_combination = None
best_val_loss = float('inf')

dropout_rates_list = [[0.5, 0.5, 0.4, 0.3, 0.2], [0.4, 0.4, 0.3, 0.2, 0.1], [0.3, 0.3, 0.2, 0.1, 0.1]]
regularization_strengths = [0.0001, 0.0005, 0.001]
learning_rates = [0.0005, 0.001, 0.005]
activation_functions = ['PReLU', 'LeakyReLU', 'ReLU', 'ELU']
optimizers = [Adam, SGD, RMSprop, AdamW]

for activation_function in activation_functions:
    for optimizer_class in optimizers:
        for dropout_rates in dropout_rates_list:
            for regularization_strength in regularization_strengths:
                for learning_rate in learning_rates:
                    for train_idx, val_idx in kfold.split(X_normalized):
                        logging.info(f"Training fold {fold} with activation function {activation_function}, optimizer {optimizer_class.__name__}, dropout rates {dropout_rates}, regularization strength {regularization_strength}, and learning rate {learning_rate}...")
                        X_train, X_val = X_normalized[train_idx], X_normalized[val_idx]
                        y_train, y_val = y_tsmc_normalized[train_idx], y_tsmc_normalized[val_idx]

                        model = build_mlp_model(activation_function=activation_function, dropout_rates=dropout_rates, regularization_strength=regularization_strength)
                        optimizer = optimizer_class(learning_rate=learning_rate)  # 每次重新建構模型時創建新的優化器
                        model.compile(optimizer=optimizer, loss=loss_function, metrics=['mae'])

                        history = model.fit(X_train, y_train,
                                            validation_data=(X_val, y_val),
                                            epochs=1000,
                                            batch_size=32,
                                            callbacks=[early_stopping, reduce_lr],
                                            verbose=0)
                        all_histories.append(history)

                        # 記錄每輪訓練的總結
                        val_loss = min(history.history['val_loss'])
                        logging.info(f"Fold {fold}, Activation {activation_function}, Optimizer {optimizer_class.__name__}, Dropout Rates {dropout_rates}, Regularization Strength {regularization_strength}, Learning Rate {learning_rate}, Best Val Loss: {val_loss:.4f}")

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_combination = {
                                'activation_function': activation_function,
                                'optimizer': optimizer_class.__name__,
                                'dropout_rates': dropout_rates,
                                'regularization_strength': regularization_strength,
                                'learning_rate': learning_rate
                            }

                        fold += 1

# 儲存最終模型
model.save(os.path.join(model_dir, "mlp_model_final.keras"))

# 評估模型
test_loss, test_mae = model.evaluate(X_normalized, y_tsmc_normalized, verbose=2)
logging.info(f'Test MAE: {test_mae:.4f}')
print(f'Test MAE: {test_mae:.4f}')

# 紀錄最終選擇的超參數組合和原因
logging.info(f'Best Combination: {best_combination}, with Best Val Loss: {best_val_loss:.4f}')
print(f'Best Combination: {best_combination}, with Best Val Loss: {best_val_loss:.4f}')

# 繪製實際股價與預測股價
y_pred = model.predict(X_normalized)
y_pred_rescaled = scaler_y_tsmc.inverse_transform(y_pred)

plt.figure(figsize=(14, 8))
plt.plot(combined_data.index, y_tsmc, label='Actual Stock Price', color='blue')
plt.plot(combined_data.index, y_pred_rescaled, color='red', label='Fitted Line')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('TSMC Stock Price and Fitted Line using MLP')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(model_dir, 'fitted_stock_price.png'), bbox_inches='tight')
plt.close()

# 預測未來股價
future_periods = {'1_year': 365, '3_years': 365 * 3, '5_years': 365 * 5}
for label, future_days in future_periods.items():
    future_X = np.arange(len(combined_data), len(combined_data) + future_days).reshape(-1, 1)
    future_X_normalized = scaler_X.transform(future_X)
    future_y_pred = model.predict(future_X_normalized)
    future_y_pred_rescaled = scaler_y_tsmc.inverse_transform(future_y_pred)

    # 繪製未來預測結果
    plt.figure(figsize=(14, 8))
    plt.plot(combined_data.index, y_tsmc, label='Actual Stock Price', color='blue')
    plt.plot(combined_data.index, y_pred_rescaled, color='red', label='Fitted Line')
    future_dates = pd.date_range(start=combined_data.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='B')
    plt.plot(future_dates, future_y_pred_rescaled[:future_days], color='green', linestyle='dashed', label='Predicted Future Prices')

    # 標記最高和最低點
    plt.grid(True)
    min_price_idx = np.argmin(y_tsmc)
    max_price_idx = np.argmax(y_tsmc)
    plt.text(combined_data.index[min_price_idx], y_tsmc[min_price_idx], f"{y_tsmc[min_price_idx]:.2f}", fontsize=12, color='black')
    plt.text(combined_data.index[max_price_idx], y_tsmc[max_price_idx], f"{y_tsmc[max_price_idx]:.2f}", fontsize=12, color='black')

    min_future_price_idx = np.argmin(future_y_pred_rescaled[:future_days])
    max_future_price_idx = np.argmax(future_y_pred_rescaled[:future_days])
    plt.text(future_dates[min_future_price_idx], future_y_pred_rescaled[min_future_price_idx][0], f"{future_y_pred_rescaled[min_future_price_idx][0]:.2f}", fontsize=12, color='black')
    plt.text(future_dates[max_future_price_idx], future_y_pred_rescaled[max_future_price_idx][0], f"{future_y_pred_rescaled[max_future_price_idx][0]:.2f}", fontsize=12, color='black')

    # 標記最後的價格和日期
    last_price = y_tsmc[-1]
    last_date = combined_data.index[-1]
    plt.text(last_date, last_price, f"{last_date.strftime('%Y-%m-%d')}: {last_price:.2f}", fontsize=12, color='black', ha='left', va='top')

    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'TSMC Stock Price Trend and {label.replace("_", " ").title()} Prediction')
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(model_dir, f'tsmc_{label}_stock_price_prediction.png'), bbox_inches='tight')
    plt.close()

# 重新讀取模型進行鴻海股價預測
loaded_model = load_model(os.path.join(model_dir, "mlp_model_final.keras"))
y_foxconn_pred = loaded_model.predict(X_normalized)
y_foxconn_pred_rescaled = scaler_y_foxconn.inverse_transform(y_foxconn_pred)

# 繪製實際鴻海股價與預測股價
plt.figure(figsize=(14, 8))
plt.plot(combined_data.index, y_foxconn, label='Actual Foxconn Stock Price', color='blue')
plt.plot(combined_data.index, y_foxconn_pred_rescaled, color='red', label='Fitted Line (Foxconn)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Foxconn Stock Price and Fitted Line using MLP (Loaded Model)')
plt.legend()
plt.grid(True)

# 標記最高和最低點
min_price_idx_foxconn = np.argmin(y_foxconn)
max_price_idx_foxconn = np.argmax(y_foxconn)
plt.text(combined_data.index[min_price_idx_foxconn], y_foxconn[min_price_idx_foxconn], f"{y_foxconn[min_price_idx_foxconn]:.2f}", fontsize=12, color='black')
plt.text(combined_data.index[max_price_idx_foxconn], y_foxconn[max_price_idx_foxconn], f"{y_foxconn[max_price_idx_foxconn]:.2f}", fontsize=12, color='black')

plt.savefig(os.path.join(model_dir, 'foxconn_fitted_stock_price.png'), bbox_inches='tight')
plt.close()

# 預測未來鴻海股價
for label, future_days in future_periods.items():
    future_y_foxconn_pred = loaded_model.predict(future_X_normalized[:future_days])
    future_y_foxconn_pred_rescaled = scaler_y_foxconn.inverse_transform(future_y_foxconn_pred)

    # 繪製未來預測結果
    plt.figure(figsize=(14, 8))
    plt.plot(combined_data.index, y_foxconn, label='Actual Foxconn Stock Price', color='blue')
    plt.plot(combined_data.index, y_foxconn_pred_rescaled, color='red', label='Fitted Line (Foxconn)')
    future_dates = pd.date_range(start=combined_data.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='B')
    plt.plot(future_dates, future_y_foxconn_pred_rescaled, color='green', linestyle='dashed', label='Predicted Future Prices (Foxconn)')

    # 標記最高和最低點
    min_price_idx_foxconn_future = np.argmin(future_y_foxconn_pred_rescaled)
    max_price_idx_foxconn_future = np.argmax(future_y_foxconn_pred_rescaled)
    plt.text(future_dates[min_price_idx_foxconn_future], future_y_foxconn_pred_rescaled[min_price_idx_foxconn_future][0], f"{future_y_foxconn_pred_rescaled[min_price_idx_foxconn_future][0]:.2f}", fontsize=12, color='black')
    plt.text(future_dates[max_price_idx_foxconn_future], future_y_foxconn_pred_rescaled[max_price_idx_foxconn_future][0], f"{future_y_foxconn_pred_rescaled[max_price_idx_foxconn_future][0]:.2f}", fontsize=12, color='black')

    # 標記最後的價格和日期
    last_price_foxconn = y_foxconn[-1]
    last_date_foxconn = combined_data.index[-1]
    plt.text(last_date_foxconn, last_price_foxconn, f"{last_date_foxconn.strftime('%Y-%m-%d')}: {last_price_foxconn:.2f}", fontsize=12, color='black', ha='left', va='top')

    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'Foxconn Stock Price Trend and {label.replace("_", " ").title()} Prediction')
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(model_dir, f'foxconn_{label}_stock_price_prediction.png'), bbox_inches='tight')
    plt.close()
