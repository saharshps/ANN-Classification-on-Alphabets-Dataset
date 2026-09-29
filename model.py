import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow import keras
from tensorflow.keras import layers

df = pd.read_csv("Alphabets_data.csv")

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

le = LabelEncoder()
y = le.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

n_features = X_train.shape[1]
n_classes = len(np.unique(y))

baseline = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(n_features,)),
    layers.Dense(n_classes, activation="softmax")
])

baseline.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

baseline.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

y_pred1 = np.argmax(baseline.predict(X_test), axis=1)

print("Baseline Accuracy:", accuracy_score(y_test, y_pred1))
print(classification_report(y_test, y_pred1))

tuned = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(n_features,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(n_classes, activation="softmax")
])

tuned.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = tuned.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.1
)

y_pred2 = np.argmax(tuned.predict(X_test), axis=1)

print("Tuned Accuracy:", accuracy_score(y_test, y_pred2))
print(classification_report(y_test, y_pred2))

cm = confusion_matrix(y_test, y_pred2)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

neurons = [32, 64, 128]

for n in neurons:
    model = keras.Sequential([
        layers.Dense(n, activation='relu', input_shape=(16,)),
        layers.Dense(26, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, epochs=20, verbose=0)

    acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print("Neurons:", n, "Accuracy:", acc)

    
from sklearn.metrics import precision_score, recall_score, f1_score

print("Precision:", precision_score(y_test, y_pred2, average='weighted'))
print("Recall:", recall_score(y_test, y_pred2, average='weighted'))
print("F1:", f1_score(y_test, y_pred2, average='weighted'))


print(baseline.summary())
