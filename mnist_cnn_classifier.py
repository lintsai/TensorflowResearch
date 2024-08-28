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

# 改變輸入形狀以適應CNN
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 定義CNN模型
print("正在定義CNN模型...")
model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 編譯模型
print("正在編譯模型...")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
print("開始訓練模型...")
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, verbose=1)

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
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"預測結果: {np.argmax(predictions[i])}")
    plt.axis('off')
    plt.show()

print("程序執行完畢!")