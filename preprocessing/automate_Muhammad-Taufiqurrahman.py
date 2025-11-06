import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    print("=== 🚀 AUTOMATED PREPROCESSING PIPELINE ===\n")

    # --- 1. BUAT FOLDER OUTPUT (PASTI ADA) ---
    output_dir = r"C:\Users\User\Downloads\SMSML_Muhammad-Taufiqurrahman\Eksperimen_SML_Muhammad-Taufiqurrahman\preprocessing\Telco-Customer-Churn_preprocessing"
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. MISSING VALUES ---
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

    # --- 3. DUPLIKAT ---
    print("📍 Menghapus Data Duplikat...")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"✅ {duplicates} baris duplikat dihapus.")
    else:
        print("Tidak ditemukan data duplikat.\n")

    # --- 4. KOLOM NUMERIK & KATEGORIK ---
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(exclude=['int64', 'float64']).columns

    # --- 5. OUTLIER ---
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
    cleaned_path = os.path.join(output_dir, "Telco-Customer-Churn_cleaned.csv")
    df_cleaned.to_csv(cleaned_path, index=False)
    print(f"💾 Cleaned data disimpan di: {cleaned_path}\n")

    # --- 6. NORMALISASI ---
    print("📍 Normalisasi Data...")
    df_normalized = df_cleaned.copy()
    if len(num_cols) > 0:
        df_normalized[num_cols] = MinMaxScaler().fit_transform(df_cleaned[num_cols])
    normalized_path = os.path.join(output_dir, "Telco-Customer-Churn_normalized.csv")
    df_normalized.to_csv(normalized_path, index=False)
    print(f"💾 Normalized data disimpan di: {normalized_path}\n")

    # --- 7. STANDARISASI ---
    print("📍 Standarisasi Data...")
    df_standardized = df_cleaned.copy()
    if len(num_cols) > 0:
        df_standardized[num_cols] = StandardScaler().fit_transform(df_cleaned[num_cols])
    standardized_path = os.path.join(output_dir, "Telco-Customer-Churn_standardized.csv")
    df_standardized.to_csv(standardized_path, index=False)
    print(f"💾 Standardized data disimpan di: {standardized_path}\n")

    # --- 8. ENCODING ---
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
    encoded_path = os.path.join(output_dir, "Telco-Customer-Churn_encoded.csv")
    df_encoded.to_csv(encoded_path, index=False)
    print(f"💾 Encoded data disimpan di: {encoded_path}\n")

    # --- 9. BINNING ---
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
    binned_path = os.path.join(output_dir, "Telco-Customer-Churn_binned.csv")
    df_binned.to_csv(binned_path, index=False)
    print(f"💾 Binned data disimpan di: {binned_path}\n")

    print("🎯 Semua file hasil preprocessing berhasil disimpan di folder:")
    print(f"📁 {os.path.abspath(output_dir)}\n")

    return {
        'cleaned': df_cleaned,
        'normalized': df_normalized,
        'standardized': df_standardized,
        'encoded': df_encoded,
        'binned': df_binned
    }


# --- EKSEKUSI ---
if __name__ == "__main__":
    print("=== UJI COBA ===")
    input_path = r"C:\Users\User\Downloads\SMSML_Muhammad-Taufiqurrahman\Eksperimen_SML_Muhammad-Taufiqurrahman\preprocessing\Telco-Customer-Churn_raw.csv"
    df_raw = pd.read_csv(input_path)
    hasil = preprocess_data(df_raw)
