import tensorflow as tf
import numpy as np
import os
import pickle
import traceback
import tensorflow_model_optimization as tfmot
from sklearn.metrics import f1_score

# 設置目錄
BASE_DIR = 'nlp_sentiment_analysis_chinese'
IMPROVE_DIR = 'nlp_sentiment_analysis_chinese_improve'
os.makedirs(IMPROVE_DIR, exist_ok=True)

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

def load_model_and_data():
    model_path = os.path.join(BASE_DIR, 'sentiment_model.keras')
    tokenizer_path = os.path.join(BASE_DIR, 'tokenizer.pickle')
    max_len_path = os.path.join(BASE_DIR, 'max_len.txt')

    print(f"模型檔案路徑: {os.path.abspath(model_path)}")
    print(f"模型檔案大小: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")

    try:
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open(max_len_path, 'r') as f:
            max_len = int(f.read().strip())

        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            print("成功加載完整模型")
        except Exception as e:
            print(f"無法加載完整模型: {e}")
            print("嘗試重建模型結構並加載權重...")
            
            vocab_size = len(tokenizer.word_index) + 1
            embedding_dim = 100
            model = build_model(vocab_size, embedding_dim, max_len)
            
            try:
                model.load_weights(model_path)
                print("成功加載模型權重")
            except Exception as e:
                print(f"無法加載模型權重: {e}")
                print("將使用初始化的權重")

        print("模型和數據加載成功")
        return model, tokenizer, max_len
    except Exception as e:
        print(f"加載過程中出錯：{e}")
        print(f"錯誤詳情:\n{traceback.format_exc()}")
        return None, None, None

def load_test_data(tokenizer, max_len):
    # 這裡應該使用實際的測試數據
    # 以下僅為示例
    x_test = np.random.randint(0, len(tokenizer.word_index) + 1, size=(1000, max_len))
    y_test = np.random.randint(0, 2, size=(1000,))
    return x_test, y_test

def quantize_model(model, x_test):
    def representative_dataset():
        for i in range(100):
            yield [x_test[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32
    tflite_quant_model = converter.convert()
    
    quant_model_path = os.path.join(IMPROVE_DIR, 'sentiment_model_quantized.tflite')
    with open(quant_model_path, 'wb') as f:
        f.write(tflite_quant_model)
    
    print(f"量化後模型大小: {len(tflite_quant_model) / 1024 / 1024:.2f} MB")
    return tflite_quant_model

def prune_model(model, x_test, y_test):
    pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
          initial_sparsity=0.40,
          final_sparsity=0.75,
          begin_step=0,
          end_step=1500)
    }
    
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    model_for_pruning.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=IMPROVE_DIR),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]

    model_for_pruning.fit(x_test, y_test, 
                          epochs=10,
                          batch_size=32, 
                          validation_split=0.2,
                          callbacks=callbacks)
    
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    model_for_export.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    pruned_model_path = os.path.join(IMPROVE_DIR, 'sentiment_model_pruned.keras')
    model_for_export.save(pruned_model_path)
    print(f"剪枝後模型大小: {os.path.getsize(pruned_model_path) / 1024 / 1024:.2f} MB")
    return model_for_export

def distill_knowledge(teacher_model, x_train, y_train):
    input_shape = teacher_model.input_shape[1:]
    student_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=teacher_model.layers[0].input_dim, 
                                  output_dim=64,
                                  input_length=input_shape[0]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    class DistillationModel(tf.keras.Model):
        def __init__(self, student, teacher):
            super(DistillationModel, self).__init__()
            self.student = student
            self.teacher = teacher
            self.distillation_loss_fn = tf.keras.losses.BinaryCrossentropy()

        def call(self, inputs, training=False):
            return self.student(inputs, training=training)

        def train_step(self, data):
            x, y = data

            teacher_predictions = self.teacher(x, training=False)

            with tf.GradientTape() as tape:
                student_predictions = self.student(x, training=True)
                distillation_loss = self.distillation_loss_fn(teacher_predictions, student_predictions)

            trainable_vars = self.student.trainable_variables
            gradients = tape.gradient(distillation_loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            self.compiled_metrics.update_state(y, student_predictions)
            results = {m.name: m.result() for m in self.metrics}
            results.update({"distillation_loss": distillation_loss})
            return results

        def test_step(self, data):
            x, y = data
            student_predictions = self.student(x, training=False)
            self.compiled_metrics.update_state(y, student_predictions)
            return {m.name: m.result() for m in self.metrics}

    distillation_model = DistillationModel(student_model, teacher_model)
    distillation_model.compile(optimizer='adam', metrics=['accuracy'])

    validation_split = 0.2
    split_index = int(len(x_train) * (1 - validation_split))
    
    x_train_split, x_val = x_train[:split_index], x_train[split_index:]
    y_train_split, y_val = y_train[:split_index], y_train[split_index:]

    batch_size = 32
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_split, y_train_split)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

    history = distillation_model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    )

    student_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    distilled_model_path = os.path.join(IMPROVE_DIR, 'sentiment_model_distilled.keras')
    student_model.save(distilled_model_path)
    print(f"蒸餾後模型大小: {os.path.getsize(distilled_model_path) / 1024 / 1024:.2f} MB")
    return student_model

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred_classes = (y_pred > 0.5).astype(int)
    accuracy = np.mean(y_pred_classes == y_test)
    f1 = f1_score(y_test, y_pred_classes)
    return accuracy, f1

def quantize_pruned_model(model, x_test):
    def representative_dataset():
        for i in range(100):
            yield [x_test[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32
    tflite_quant_model = converter.convert()
    
    quant_pruned_model_path = os.path.join(IMPROVE_DIR, 'sentiment_model_pruned_quantized.tflite')
    with open(quant_pruned_model_path, 'wb') as f:
        f.write(tflite_quant_model)
    
    print(f"剪枝後量化模型大小: {len(tflite_quant_model) / 1024 / 1024:.2f} MB")
    return tflite_quant_model

if __name__ == "__main__":
    print("正在加載原始模型和數據...")
    original_model, tokenizer, max_len = load_model_and_data()

    if original_model is None:
        print("模型加載失敗，無法繼續優化過程。")
    else:
        batch_size = 32
        x_test, y_test = load_test_data(tokenizer, max_len)
        num_samples = (len(x_test) // batch_size) * batch_size
        x_test = x_test[:num_samples]
        y_test = y_test[:num_samples]

        print(f"原始模型大小: {os.path.getsize(os.path.join(BASE_DIR, 'sentiment_model.keras')) / 1024 / 1024:.2f} MB")
        accuracy, f1 = evaluate_model(original_model, x_test, y_test)
        print(f"原始模型準確率: {accuracy:.4f}, F1 分數: {f1:.4f}")

        print("\n執行模型量化...")
        quantized_model = quantize_model(original_model, x_test)

        print("\n執行模型剪枝...")
        pruned_model = prune_model(original_model, x_test, y_test)
        accuracy, f1 = evaluate_model(pruned_model, x_test, y_test)
        print(f"剪枝後模型準確率: {accuracy:.4f}, F1 分數: {f1:.4f}")

        print("\n對剪枝後的模型進行量化...")
        quantized_pruned_model = quantize_pruned_model(pruned_model, x_test)

        print("\n執行知識蒸餾...")
        distilled_model = distill_knowledge(original_model, x_test, y_test)
        accuracy, f1 = evaluate_model(distilled_model, x_test, y_test)
        print(f"蒸餾後模型準確率: {accuracy:.4f}, F1 分數: {f1:.4f}")

        print("\n優化完成。優化後的模型保存在", IMPROVE_DIR)

    print("\nTensorFlow 版本:", tf.__version__)