import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    print("=== 🚀 AUTOMATED PREPROCESSING PIPELINE ===\n")

    # --- 1. MENANGANI DATA KOSONG ---
    print("📍 Menangani Missing Values...")
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100

    for col in df.columns:
        if missing_percent[col] > 0:
            if missing_percent[col] < 5:
                df = df[df[col].notnull()]
                print(f"✅ '{col}' ({missing_percent[col]:.2f}%) → dihapus")
            else:
                strategy = 'median' if df[col].dtype in ['float64', 'int64'] else 'most_frequent'
                imputer = SimpleImputer(strategy=strategy)
                df[[col]] = imputer.fit_transform(df[[col]])
                print(f"⚙️ '{col}' ({missing_percent[col]:.2f}%) → imputasi ({strategy})")

    print("✅ Penanganan missing values selesai.\n")

    # --- 2. MENGHAPUS DATA DUPLIKAT ---
    print("📍 Menghapus Data Duplikat...")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"✅ {duplicates} baris duplikat dihapus.")
    else:
        print("Tidak ditemukan data duplikat.\n")

    # --- 3. NORMALISASI & STANDARISASI FITUR ---
    print("\n📍 Normalisasi dan Standarisasi Fitur...")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    df[num_cols] = minmax_scaler.fit_transform(df[num_cols])
    df[num_cols] = standard_scaler.fit_transform(df[num_cols])
    print("✅ Normalisasi dan standarisasi selesai.\n")

    # --- 4. DETEKSI & PENANGANAN OUTLIER ---
    print("📍 Deteksi dan Penanganan Outlier...")
    def handle_outliers(data, column):
        Q1, Q3 = data[column].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        low, up = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        data[column] = np.where(data[column] < low, low,
                                np.where(data[column] > up, up, data[column]))
        return data

    for col in num_cols:
        df = handle_outliers(df, col)
    print("✅ Outlier selesai ditangani.\n")

    # --- 5. ENCODING DATA KATEGORIKAL ---
    print("📍 Encoding Data Kategorikal...")
    cat_cols = df.select_dtypes(exclude=['int64', 'float64']).columns
    for col in cat_cols:
        unique_count = df[col].nunique()
        if unique_count <= 2:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        elif unique_count < 10:
            encoded = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df.drop(col, axis=1), encoded], axis=1)
        else:
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq)
    print("✅ Encoding selesai.\n")

    # --- 6. BINNING (PENGELOMPOKAN DATA) ---
    print("📍 Melakukan Binning Data Numerik...")
    n_bins = 3
    for col in num_cols:
        unique_vals = df[col].nunique()
        if unique_vals > n_bins:
            try:
                df[col + '_bin'] = pd.cut(df[col], bins=n_bins, labels=['Rendah', 'Sedang', 'Tinggi'])
            except:
                continue
    print("✅ Binning selesai.\n")

    print("🎯 Semua tahapan preprocessing selesai. Data siap digunakan untuk model.\n")
    return df


# --- EKSEKUSI OTOMATIS ---
if __name__ == "__main__":
    print("=== 🔧 UJI COBA PIPELINE PREPROCESSING ===\n")
input_path = r"C:\Users\User\Downloads\SMSML_Muhammad-Taufiqurrahman\Eksperimen_SML_Muhammad-Taufiqurrahman\preprocessing\Telco-Customer-Churn_raw.csv"
output_path = r"C:\Users\User\Downloads\SMSML_Muhammad-Taufiqurrahman\Eksperimen_SML_Muhammad-Taufiqurrahman\preprocessing\Telco-Customer-Churn_preprocessing.csv"

df = pd.read_csv(input_path)
hasil = preprocess_data(df)

# Pastikan folder ada dulu
os.makedirs(os.path.dirname(output_path), exist_ok=True)

hasil.to_csv(output_path, index=False)
print("✅ File hasil preprocessing berhasil disimpan di:", output_path)