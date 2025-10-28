# train_and_save.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

# --------------- konfigurasi ---------------
DATA_PATH = "Online Gaming Prediction using 7 scale Dataset.xlsx"
OUT_MODEL = "model_bundle.joblib"   # file yang akan disimpan
RANDOM_STATE = 42

# --------------- load data ---------------
df = pd.read_excel(DATA_PATH)
df.columns = df.columns.str.strip()

# optional: tampilkan info singkat
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# --------------- pra-pemrosesan ---------------
# Simpan encoders per kolom teks agar konsisten saat inferensi
encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = {cls: int(lbl) for lbl, cls in enumerate(le.classes_)}  # mapping (optional readable)
    # also store the LabelEncoder object for inverse/consistent transform:
    encoders[col + "_leobj"] = le

# Hapus baris yang memiliki target kosong (jika ada)
df = df.dropna(subset=['Addiction'])

# pisahkan X dan y
X = df.drop('Addiction', axis=1)
y = df['Addiction'].astype(int)

# --------------- split ---------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y if len(y.unique())>1 else None
)

# --------------- Model A (overfit, tidak dipakai) ---------------
model_A = DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_STATE) 
model_A.fit(X_train, y_train)

# --------------- Model B (pruned) ---------------
# Pilih Model B sebagai model final — kontrol kompleksitas dengan max_depth
model_B = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=RANDOM_STATE)
model_B.fit(X_train, y_train)

# evaluasi singkat
y_pred = model_B.predict(X_test)
print("Accuracy (Model B):", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# --------------- simpan model + encoders + metadata ---------------
bundle = {
    "model_A": model_A,
    "model_B": model_B,
    "final_model": model_B,
    "encoders": {k: (v if isinstance(v, dict) else v) for k, v in encoders.items()},
    "feature_columns": X.columns.tolist(),
    "target_name": "Addiction"
}

joblib.dump(bundle, OUT_MODEL)
print(f"Model dan metadata tersimpan ke: {OUT_MODEL}")
