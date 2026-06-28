# train.py
# Loads preprocessed numpy arrays and trains KNN, SVM, LSTM, 1D CNN

import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ────────────────────────────────────────────
WINDOW_SIZE = 80
N_CHANNELS  = 11
EPOCHS      = 100
BATCH_SIZE  = 32

# ── Load arrays ───────────────────────────────────────
X_train_seq  = np.load('X_train_seq.npy')
X_val_seq    = np.load('X_val_seq.npy')
X_test_seq   = np.load('X_test_seq.npy')
X_train_feat = np.load('X_train_feat.npy')
X_val_feat   = np.load('X_val_feat.npy')
X_test_feat  = np.load('X_test_feat.npy')
y_train      = np.load('y_train.npy')
y_val        = np.load('y_val.npy')
y_test       = np.load('y_test.npy')
classes      = np.load('label_classes.npy', allow_pickle=True)
n_classes    = len(classes)

print(f"Train: {len(X_train_seq)} | Val: {len(X_val_seq)} | Test: {len(X_test_seq)} windows")
print(f"Classes ({n_classes}): {list(classes)}")

# ── Scale features for KNN/SVM ────────────────────────
scaler = StandardScaler()
X_train_feat = scaler.fit_transform(X_train_feat)
X_val_feat   = scaler.transform(X_val_feat)
X_test_feat  = scaler.transform(X_test_feat)

# ── Confusion matrix ──────────────────────────────────
def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def evaluate(name, y_pred_val, y_pred_test):
    acc_val  = accuracy_score(y_val,  y_pred_val)
    acc_test = accuracy_score(y_test, y_pred_test)
    print(f"\n── {name} ──")
    print(f"  Within-subject (val):  {acc_val*100:.1f}%")
    print(f"  Cross-subject  (test): {acc_test*100:.1f}%")
    print(classification_report(y_test, y_pred_test, target_names=classes))
    plot_confusion(y_test, y_pred_test, f"{name} Cross-Subject")
    return acc_val, acc_test

results = {}

# ── KNN ───────────────────────────────────────────────
print("\nTraining KNN...")
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', n_jobs=-1)
knn.fit(X_train_feat, y_train)
results['KNN'] = evaluate('KNN', knn.predict(X_val_feat), knn.predict(X_test_feat))
with open("knn_model.pkl", "wb") as f:
    pickle.dump({'knn': knn, 'scaler': scaler, 'classes': classes}, f)

# ── SVM ───────────────────────────────────────────────
print("\nTraining SVM...")
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']}
svm = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=5, n_jobs=-1, verbose=1)
svm.fit(X_train_feat, y_train)
print(f"  Best params: {svm.best_params_}")
results['SVM'] = evaluate('SVM', svm.predict(X_val_feat), svm.predict(X_test_feat))
with open("svm_model.pkl", "wb") as f:
    pickle.dump({'svm': svm, 'scaler': scaler, 'classes': classes}, f)

# ── Categorical labels ────────────────────────────────
y_train_cat = tf.keras.utils.to_categorical(y_train, n_classes)
y_val_cat   = tf.keras.utils.to_categorical(y_val,   n_classes)

cb = [
    callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
]

# ── LSTM ──────────────────────────────────────────────
print("\nTraining LSTM...")
lstm = models.Sequential([
    layers.Input(shape=(WINDOW_SIZE, N_CHANNELS)),
    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(64),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(n_classes, activation='softmax')
])
lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lstm.fit(X_train_seq, y_train_cat, validation_data=(X_val_seq, y_val_cat),
         epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=1)
results['LSTM'] = evaluate('LSTM',
    np.argmax(lstm.predict(X_val_seq),  axis=1),
    np.argmax(lstm.predict(X_test_seq), axis=1))
lstm.save("lstm_model.keras")

# ── 1D CNN ────────────────────────────────────────────
print("\nTraining 1D CNN...")
cnn = models.Sequential([
    layers.Input(shape=(WINDOW_SIZE, N_CHANNELS)),
    layers.Conv1D(32,  kernel_size=5, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv1D(64,  kernel_size=5, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling1D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(n_classes, activation='softmax')
])
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(X_train_seq, y_train_cat, validation_data=(X_val_seq, y_val_cat),
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=1)
results['1D CNN'] = evaluate('1D CNN',
    np.argmax(cnn.predict(X_val_seq),  axis=1),
    np.argmax(cnn.predict(X_test_seq), axis=1))
cnn.save("cnn_model.keras")

# ── Summary ───────────────────────────────────────────
print("\n── Final Results ──")
print(f"  {'Model':<10} {'Within-Subject':>16} {'Cross-Subject':>14}")
print(f"  {'─'*44}")
for name, (acc_val, acc_test) in sorted(results.items(), key=lambda x: x[1][1], reverse=True):
    print(f"  {name:<10} {acc_val*100:>14.1f}%  {acc_test*100:>12.1f}%")