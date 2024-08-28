import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 加載MNIST數據集
print("正在加載MNIST數據集...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 數據預處理
print("正在進行數據預處理...")
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# 定義模型
print("正在定義模型...")
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 編譯模型
print("正在編譯模型...")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
print("開始訓練模型...")
history = model.fit(x_train, y_train, epochs=5, validation_split=0.2, verbose=1)

# 評估模型
print("正在評估模型...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n測試準確率: {test_acc}")

# 進行預測
print("正在進行預測...")
predictions = model.predict(x_test[:5])

# 顯示一些預測結果
for i in range(5):
    plt.figure()
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"預測結果: {np.argmax(predictions[i])}")
    plt.axis('off')
    plt.show()

print("程序執行完畢!")