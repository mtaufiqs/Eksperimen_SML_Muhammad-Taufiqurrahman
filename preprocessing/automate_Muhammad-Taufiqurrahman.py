import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

    # --- 3. IDENTIFIKASI KOLOM NUMERIK & KATEGORIK ---
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(exclude=['int64', 'float64']).columns

    # --- 4. NORMALISASI & STANDARISASI ---
    print("\n📍 Normalisasi & Standarisasi Fitur...")
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    df_normalized = df.copy()
    df_normalized[num_cols] = minmax_scaler.fit_transform(df[num_cols])

    df_standardized = df.copy()
    df_standardized[num_cols] = standard_scaler.fit_transform(df[num_cols])

    print("✅ Normalisasi dan standarisasi selesai.\n")

    # --- 5. DETEKSI & PENANGANAN OUTLIER ---
    print("📍 Deteksi dan Penanganan Outlier...")
    def detect_outliers_iqr(data, column):
        Q1, Q3 = data[column].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        low, up = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = data[(data[column] < low) | (data[column] > up)]
        return outliers, low, up

    df_cleaned = df.copy()
    for col in num_cols:
        outliers, low, up = detect_outliers_iqr(df, col)
        perc = (len(outliers) / len(df)) * 100
        if perc < 5:
            df_cleaned = df_cleaned[(df_cleaned[col] >= low) & (df_cleaned[col] <= up)]
        else:
            df_cleaned[col] = np.where(df_cleaned[col] < low, low,
                                       np.where(df_cleaned[col] > up, up, df_cleaned[col]))
    print("✅ Outlier selesai ditangani.\n")

    # --- 6. ENCODING DATA KATEGORIKAL ---
    print("📍 Encoding Data Kategorikal...")
    df_encoded = df_cleaned.copy()
    for col in cat_cols:
        unique_count = df[col].nunique()
        if unique_count <= 2:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col])
        elif unique_count < 10:
            encoded = pd.get_dummies(df_encoded[col], prefix=col)
            df_encoded = pd.concat([df_encoded.drop(col, axis=1), encoded], axis=1)
        else:
            freq = df[col].value_counts(normalize=True)
            df_encoded[col] = df[col].map(freq)
    print("✅ Encoding selesai.\n")

    # --- 7. BINNING (PENGELOMPOKAN DATA) ---
    print("📍 Melakukan Binning Data Numerik...")
    df_binned = df_encoded.copy()
    n_bins = 3
    for col in num_cols:
        unique_vals = df[col].nunique()
        if unique_vals > n_bins:
            try:
                df_binned[col + '_bin_width'] = pd.cut(df[col], bins=n_bins, labels=['Rendah', 'Sedang', 'Tinggi'])
                df_binned[col + '_bin_freq'] = pd.qcut(df[col], q=n_bins, labels=['Rendah', 'Sedang', 'Tinggi'], duplicates='drop')
            except:
                continue
    print("✅ Binning selesai.\n")

    print("🎯 Semua tahapan preprocessing selesai.")
    print("Data siap digunakan untuk pelatihan model.\n")

    return {
        'cleaned': df_cleaned,
        'normalized': df_normalized,
        'standardized': df_standardized,
        'encoded': df_encoded,
        'binned': df_binned
    }

# Contoh penggunaan
if __name__ == "__main__":
    print("=== UJI COBA ===")
    # Contoh pemanggilan: ganti dengan dataset kamu
    # df = pd.read_csv("dataset.csv")
    # hasil = preprocess_data(df)
    # print(hasil['cleaned'].head())
