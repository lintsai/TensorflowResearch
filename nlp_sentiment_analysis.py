import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. 加載數據
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=20000)

# 2. 數據預處理
max_len = 200
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# 3. 構建改進的模型
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(20000, 32, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. 訓練模型
history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2, verbose=1)

# 5. 評估模型
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.3f}")

# 6. 使用模型進行預測
word_index = tf.keras.datasets.imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

def encode_review(text):
    encoded = [1]
    for word in text.split():
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()] + 3)
        else:
            encoded.append(2)
    return encoded

def predict_sentiment(text):
    encoded_review = encode_review(text)
    padded = pad_sequences([encoded_review], maxlen=max_len)
    prediction = model.predict(padded)[0][0]
    print(f"Prediction score: {prediction:.3f}")
    return "Positive" if prediction > 0.5 else "Negative"

# 正向測試
positive_review = "This movie was fantastic! I really enjoyed it and would recommend it to anyone."
print(f"Sentiment: {predict_sentiment(positive_review)}")

# 負向測試
negative_review = "This movie was terrible. I hated every minute of it."
print(f"Sentiment for negative review: {predict_sentiment(negative_review)}")

# 額外的測試
extra_review = "I enjoyed the movie but it might too terrible and bloody to you if you don't like thriller film."
print(f"Sentiment for Extra: {predict_sentiment(extra_review)}")