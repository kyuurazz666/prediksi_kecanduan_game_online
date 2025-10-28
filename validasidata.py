# ============================================================
# TP-05 | Decision Tree
# Dataset: Online Gaming Prediction using 7 Scale Dataset
# Tujuan: Memuat dataset & melihat distribusi variabel target (Addiction)
# ============================================================

# Import library yang diperlukan
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Membaca dataset dari file Excel
# Ganti path di bawah sesuai lokasi file kamu
df = pd.read_excel("Online Gaming Prediction using 7 scale Dataset.xlsx")

# Menampilkan 5 baris pertama dataset
print("=== 5 Baris Pertama Data ===")
print(df.head())

# Menampilkan informasi umum dataset
print("\n=== Informasi Dataset ===")
print(df.info())

# Mengecek apakah ada nilai yang kosong (missing values)
print("\n=== Jumlah Nilai Kosong di Setiap Kolom ===")
print(df.isnull().sum())

# Visualisasi distribusi variabel target 'Addiction'
plt.figure(figsize=(6,4))
sns.countplot(x='Addiction', data=df, palette='Set2')
plt.title('Distribusi Kelas Target: Addiction')
plt.xlabel('Kategori Addiction (N = Tidak Kecanduan, Y = Kecanduan)')
plt.ylabel('Jumlah Responden')
plt.show()
