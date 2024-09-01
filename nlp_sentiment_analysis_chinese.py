import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import requests
from bs4 import BeautifulSoup
import jieba
import re
import time
import random
import concurrent.futures
import queue
import matplotlib.pyplot as plt
import pickle
import os
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

# 全局變量
reviews_queue = queue.Queue()
total_reviews = 0

# PTT 電影版數據收集函數
def collect_ptt_reviews(max_reviews=20000, min_delay=0.1, max_delay=0.5):
    global total_reviews
    base_url = "https://www.ptt.cc"
    current_page_url = f"{base_url}/bbs/movie/index.html"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    def process_article(article_url, title_text):
        global total_reviews
        try:
            article_response = requests.get(article_url, headers=headers)
            article_response.raise_for_status()
            article_soup = BeautifulSoup(article_response.text, 'html.parser')
            
            content = article_soup.select_one('#main-content')
            if content:
                for meta in content.select('div.article-metaline'):
                    meta.decompose()
                for meta in content.select('div.article-metaline-right'):
                    meta.decompose()
                
                text = content.text.strip()
                text = re.sub(r': .*', '', text)
                text = re.sub(r'※ 引述.*', '', text)
                
                label = 1 if '[好雷]' in title_text or '[普雷]' in title_text else 0
                reviews_queue.put((text, label))
                total_reviews += 1
                print(f"\r已收集 {total_reviews} 條評論", end="", flush=True)
        except Exception as e:
            print(f"\n處理文章時出錯: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        while total_reviews < max_reviews:
            try:
                response = requests.get(current_page_url, headers=headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for article in soup.select('div.r-ent'):
                    if total_reviews >= max_reviews:
                        break
                    
                    title = article.select_one('div.title a')
                    if title and ('[好雷]' in title.text or '[普雷]' in title.text or '[負雷]' in title.text):
                        article_url = base_url + title['href']
                        executor.submit(process_article, article_url, title.text)
                        time.sleep(random.uniform(min_delay, max_delay))
                
                next_page_link = soup.select_one('a.btn.wide:contains("‹ 上頁")')
                if next_page_link:
                    current_page_url = base_url + next_page_link['href']
                else:
                    print("\n已到達最舊的頁面，停止爬取。")
                    break

            except requests.RequestException as e:
                print(f"\n請求錯誤: {e}")
                time.sleep(10)  # 如果出錯，等待10秒後重試
            except Exception as e:
                print(f"\n未預期的錯誤: {e}")
                break

    print(f"\n總共收集了 {total_reviews} 條評論")
    reviews = []
    labels = []
    while not reviews_queue.empty():
        review, label = reviews_queue.get()
        reviews.append(review)
        labels.append(label)
    return reviews, labels

# 數據預處理函數
def preprocess_text(text):
    # 停用詞列表
    stopwords = set(['的', '是', '在', '和', '了', '與', '就', '都', '而', '及', '著', '或', '一個', '沒有', '我們', '你們', '他們', 
                     '了', '的', '是', '之', '於', '以', '及', '和', '或', '這', '那', '要', '就', '但', '與', '因為', '所以', '可以', '如果',
                     '啊', '呀', '哎', '哦', '唉', '嗯', '嘛', '吧', '呢', '了', '喔', '喲', '嗨', '欸'])
    words = jieba.cut(text)
    return ' '.join([word for word in words if word not in stopwords and word.strip() and not re.match(r'[^\w\s]', word)])

# 模型構建函數 - 使用 CNN
def build_model(vocab_size, embedding_dim, max_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

# 保存模型和相關數據
def save_model_and_data(model, tokenizer, max_len):
    folder_name = "nlp_sentiment_analysis_chinese"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    model.save(os.path.join(folder_name, 'sentiment_model.keras'))
    with open(os.path.join(folder_name, 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(folder_name, 'max_len.txt'), 'w') as f:
        f.write(str(max_len))
    print("模型和相關數據已保存。")

# 加載模型和相關數據
def load_model_and_data():
    folder_name = "nlp_sentiment_analysis_chinese"
    model_path = os.path.join(folder_name, 'sentiment_model.keras')
    tokenizer_path = os.path.join(folder_name, 'tokenizer.pickle')
    max_len_path = os.path.join(folder_name, 'max_len.txt')
    
    print(f"正在檢查以下路徑：")
    print(f"模型路徑: {os.path.abspath(model_path)}")
    print(f"tokenizer路徑: {os.path.abspath(tokenizer_path)}")
    print(f"最大長度路徑: {os.path.abspath(max_len_path)}")
    
    if os.path.exists(model_path) and os.path.exists(tokenizer_path) and os.path.exists(max_len_path):
        try:
            model = tf.keras.models.load_model(model_path)
            with open(tokenizer_path, 'rb') as handle:
                tokenizer = pickle.load(handle)
            with open(max_len_path, 'r') as f:
                max_len = int(f.read())
            print("已成功加載保存的模型和相關數據。")
            return model, tokenizer, max_len
        except Exception as e:
            print(f"加載過程中出錯：{e}")
            return None, None, None
    else:
        print("未找到一個或多個必要的文件。")
        if not os.path.exists(model_path):
            print(f"缺少模型文件: {model_path}")
        if not os.path.exists(tokenizer_path):
            print(f"缺少tokenizer文件: {tokenizer_path}")
        if not os.path.exists(max_len_path):
            print(f"缺少最大長度文件: {max_len_path}")
        return None, None, None

# 主程序
if __name__ == "__main__":
    print(f"當前工作目錄: {os.getcwd()}")
    # 檢查 nlp_sentiment_analysis_chinese 文件夾是否存在
    folder_name = "nlp_sentiment_analysis_chinese"
    if os.path.exists(folder_name):
        print(f"{folder_name} 文件夾存在")
        # 列出文件夾中的文件
        print("文件夾中的文件:")
        for file in os.listdir(folder_name):
            print(f" - {file}")
    else:
        print(f"{folder_name} 文件夾不存在")

    # 嘗試加載已保存的模型和數據
    model, tokenizer, max_len = load_model_and_data()

    if model is None:
        print("正在收集 PTT 電影版評論...")
        reviews, labels = collect_ptt_reviews(max_reviews=20000)  # 保持數據量

        print("正在預處理評論...")
        processed_reviews = [preprocess_text(review) for review in reviews]

        # 使用 Tokenizer 進行文本轉換
        tokenizer = Tokenizer(num_words=15000, oov_token="<OOV>")
        tokenizer.fit_on_texts(processed_reviews)
        
        sequences = tokenizer.texts_to_sequences(processed_reviews)
        max_len = max([len(x) for x in sequences])
        X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
        y = np.array(labels)

        # 數據分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # 使用 SMOTE 處理不平衡數據
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # 計算類別權重
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(class_weights))

        # 使用 StratifiedKFold 交叉驗證
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_no = 1
        acc_per_fold = []
        loss_per_fold = []
        auc_per_fold = []

        for train_index, val_index in skf.split(X_train_resampled, y_train_resampled):
            X_train_fold, X_val = X_train_resampled[train_index], X_train_resampled[val_index]
            y_train_fold, y_val = y_train_resampled[train_index], y_train_resampled[val_index]

            print(f'訓練折疊 {fold_no} ...')

            model = build_model(len(tokenizer.word_index) + 1, 100, max_len)

            # 早停機制和學習率衰減
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3)

            history = model.fit(X_train_fold, y_train_fold, 
                                epochs=50,  # 減少 epochs
                                batch_size=32,  # 減小 batch size
                                validation_data=(X_val, y_val), 
                                class_weight=class_weights,
                                callbacks=[early_stopping, lr_reducer],
                                verbose=1)

            scores = model.evaluate(X_val, y_val, verbose=0)
            print(f'折疊 {fold_no} 分數 - 損失: {scores[0]:.4f} - 準確率: {scores[1]:.4f} - AUC: {scores[2]:.4f}')

            acc_per_fold.append(scores[1])
            loss_per_fold.append(scores[0])
            auc_per_fold.append(scores[2])

            fold_no += 1

        # 打印每個折疊的結果
        print('每個折疊的分數:')
        for i in range(len(acc_per_fold)):
            print(f'> 折疊 {i+1} - 損失: {loss_per_fold[i]:.4f} - 準確率: {acc_per_fold[i]:.4f} - AUC: {auc_per_fold[i]:.4f}')

        # 打印平均分數
        print('平均分數:')
        print(f'> 準確率: {np.mean(acc_per_fold):.4f} (+/- {np.std(acc_per_fold):.4f})')
        print(f'> AUC: {np.mean(auc_per_fold):.4f} (+/- {np.std(auc_per_fold):.4f})')
        print(f'> 損失: {np.mean(loss_per_fold):.4f}')

        # 在完全獨立的測試集上進行最終評估
        final_test_loss, final_test_accuracy, final_test_auc = model.evaluate(X_test, y_test)
        print(f"最終測試準確率：{final_test_accuracy:.4f}")
        print(f"最終測試 AUC：{final_test_auc:.4f}")

        # 在測試集上進行預測並輸出分類報告
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)  # 使用0.5作為閾值
        print("\n分類報告:")
        print(classification_report(y_test, y_pred_binary))
        print("\n混淆矩陣:")
        print(confusion_matrix(y_test, y_pred_binary))

        # 保存模型和相關數據
        save_model_and_data(model, tokenizer, max_len)

    # 預測新評論
    def predict_sentiment(text):
        processed = preprocess_text(text)
        sequence = tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
        prediction = model.predict(padded)[0][0]
        return "正面" if prediction > 0.5 else "負面", prediction

    # 測試預測
    new_reviews = [
        "這部電影真的很精彩，情節緊湊，演員演技出色。",
        "劇情很無聊，演員表演也很差勁，不推薦。",
        "這電影還可以，有些地方很有趣，但整體來說不算特別出色。",
        "我覺得這部電影非常糟糕，完全浪費了我的時間和金錢。"
    ]

    for review in new_reviews:
        sentiment, score = predict_sentiment(review)
        print(f"評論：{review}")
        print(f"情感：{sentiment}")
        print(f"預測分數：{score:.4f}")
        print()

    # 繪製訓練過程圖表（如果有歷史數據）
    if 'history' in locals():
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(history.history['auc'], label='Training AUC')
        plt.plot(history.history['val_auc'], label='Validation AUC')
        plt.title('Model AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()

        plt.tight_layout()
        plt.show()