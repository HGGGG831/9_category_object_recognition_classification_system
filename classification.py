import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 创建保存图像的目录
os.makedirs('results', exist_ok=True)

# 1. 数据加载和预处理
def load_and_preprocess_data(csv_file, test_size=0.2):
    data = pd.read_csv(csv_file)

    features1 = data[['功率密度_mW/cm2', '光照电流_nA']].values
    features2 = data[['压强_KPa', '压力电流_nA']].values
    labels = data['类别'].values

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
        features1, features2, labels, test_size=test_size, random_state=42)

    scaler1 = StandardScaler()
    scaler2 = StandardScaler()

    X1_train = scaler1.fit_transform(X1_train)
    X1_test = scaler1.transform(X1_test)

    X2_train = scaler2.fit_transform(X2_train)
    X2_test = scaler2.transform(X2_test)

    return (X1_train, X2_train, y_train), (X1_test, X2_test, y_test), label_encoder


# 2. 构建多模态模型
def build_multimodal_model(num_classes):
    input1 = layers.Input(shape=(2,), name='modal1_input')
    input2 = layers.Input(shape=(2,), name='modal2_input')

    x1 = layers.Dense(64, activation='relu')(input1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dense(128, activation='relu')(x1)

    x2 = layers.Dense(64, activation='relu')(input2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dense(128, activation='relu')(x2)

    merged = layers.concatenate([x1, x2])
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.Dropout(0.3)(x)

    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=[input1, input2], outputs=output)
    return model


# 3. 保存训练曲线
def plot_training_history(history, save_path=None):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] 训练曲线图已保存至: {save_path}")
    plt.close()


# 4. 保存混淆矩阵图
def plot_normalized_confusion_matrix(y_true, y_pred, classes, title='Normalized Confusion Matrix', save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                vmin=0, vmax=1)

    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] 混淆矩阵图已保存至: {save_path}")
    plt.close()


# 5. 训练与评估
def train_and_evaluate(csv_file, epochs=100, batch_size=16):
    (X1_train, X2_train, y_train), (X1_test, X2_test, y_test), label_encoder = load_and_preprocess_data(csv_file)
    num_classes = len(label_encoder.classes_)

    model = build_multimodal_model(num_classes)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.003),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

    history = model.fit(
        [X1_train, X2_train], y_train,
        validation_data=([X1_test, X2_test], y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[lr_scheduler],
        verbose=1
    )

    test_loss, test_acc = model.evaluate([X1_test, X2_test], y_test, verbose=0)
    print(f'\nTest Accuracy: {test_acc:.4f}')

    y_pred = model.predict([X1_test, X2_test])
    y_pred_classes = np.argmax(y_pred, axis=1)

    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred_classes)

    print('\nClassification Report:')
    print(classification_report(y_test_labels, y_pred_labels))

    # 保存图像
    plot_training_history(history, save_path='results/accuracy_loss_curve.png')
    plot_normalized_confusion_matrix(y_test_labels, y_pred_labels,
                                     classes=label_encoder.classes_,
                                     save_path='results/confusion_matrix.png')

    return model, history, label_encoder


# 主程序
if __name__ == "__main__":
    CSV_FILE = 'uniform_9class_dataset.csv'  # 替换为你的CSV路径
    model, history, label_encoder = train_and_evaluate(CSV_FILE, epochs=100)

    model.save('multimodal_classifier.h5')
    np.save('label_encoder_classes.npy', label_encoder.classes_)
    print("[INFO] 模型与标签编码器已保存")
