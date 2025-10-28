# ============================================================
# TA-05 | Decision Tree Classifier
# Kasus: Prediksi Kecanduan Game Online
# Dataset: Online Gaming Prediction using 7 Scale Dataset
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Membaca dataset dari file Excel
df = pd.read_excel("Online Gaming Prediction using 7 scale Dataset.xlsx")

# Membersihkan nama kolom dari spasi berlebih
df.columns = df.columns.str.strip()

# Menampilkan 5 data teratas
print("=== 5 Data Teratas ===")
print(df.head())

# Menampilkan informasi dataset
print("\n=== Informasi Dataset ===")
print(df.info())

# Menampilkan jumlah data dan kolom
print("\nJumlah Data:", df.shape)

# Menampilkan ringkasan statistik
print("\n=== Statistik Deskriptif ===")
print(df.describe())

# Mengecek nilai kosong
print("\n=== Jumlah Nilai Kosong ===")
print(df.isnull().sum())

# Visualisasi distribusi kelas target (Addiction)
plt.figure(figsize=(6,4))
sns.countplot(x='Addiction', data=df, palette='Set2')
plt.title("Distribusi Variabel Target: Addiction")
plt.xlabel("Kategori Addiction (N = Tidak Kecanduan, Y = Kecanduan)")
plt.ylabel("Jumlah Data")
plt.show()

# Encode variabel kategorikal (Gender dan skala teks)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Mengecek hasil encoding
print("\n=== Data Setelah Encoding ===")
print(df.head())

# Memisahkan variabel fitur (X) dan target (y)
X = df.drop('Addiction', axis=1)
y = df['Addiction']

# Membagi data menjadi data latih dan data uji (80% : 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

print("Jumlah Data Latih:", X_train.shape)
print("Jumlah Data Uji:", X_test.shape)

# Membuat model Decision Tree dengan parameter default
model = DecisionTreeClassifier(criterion='entropy', random_state=0)

# Melatih model dengan data latih
model.fit(X_train, y_train)

# Melakukan prediksi pada data uji
y_pred = model.predict(X_test)

# Menghitung akurasi model
akurasi = metrics.accuracy_score(y_test, y_pred)
print(f"Akurasi Model Decision Tree: {akurasi:.2f}")

# Menampilkan confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Visualisasi confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Tidak Kecanduan', 'Kecanduan'], yticklabels=['Tidak Kecanduan', 'Kecanduan'])
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.show()

# Laporan klasifikasi
print("\n=== Classification Report ===")
print(metrics.classification_report(y_test, y_pred))

from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(model,
          feature_names=X.columns,
          class_names=['Tidak Kecanduan', 'Kecanduan'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Visualisasi Pohon Keputusan - Prediksi Kecanduan Game Online")
plt.show()

# Ambil nilai importance dari model Decision Tree
feature_importance = pd.DataFrame({
    'Fitur': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Tampilkan tabel importance
print("\n=== Nilai Importance Setiap Fitur ===")
print(feature_importance)

# Visualisasi feature importance
plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Fitur', data=feature_importance, palette='viridis')
plt.title("Feature Importance - Decision Tree")
plt.xlabel("Tingkat Pengaruh terhadap Prediksi")
plt.ylabel("Nama Fitur")
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score

# Prediksi probabilitas (bukan label)
y_prob = model.predict_proba(X_test)[:,1]

# Hitung nilai ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# Plot ROC Curve
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Kurva ROC - Decision Tree')
plt.legend(loc='lower right')
plt.show()