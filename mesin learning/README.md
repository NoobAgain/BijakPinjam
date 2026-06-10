# 🚀 AI Loan Decision Support System - BijakPinjam

**Tim:** PJK-GM032  
**Proyek:** Pengembangan Model Machine Learning untuk Klasifikasi Risiko Pinjaman

---

## 📌 Deskripsi Proyek
Proyek ini bertujuan untuk membangun model Machine Learning yang mampu mengklasifikasikan tingkat risiko nasabah peminjam (Lancar vs. Berisiko) menggunakan data historis pinjaman. Model ini dikembangkan sebagai inti (core) dari aplikasi pengambilan keputusan **BijakPinjam**, yang dirancang untuk mencegah terjadinya *debt trap* pada pengguna.

## 📊 Dataset
Dataset yang digunakan adalah data yang telah melalui proses *preprocessing*, *cleaning*, *encoding*, dan *feature engineering* (`Dataset_Bersih_Siap_ML_500k.xlsx`).
* **Jumlah Baris:** 500.000 data
* **Jumlah Fitur (X):** 9 fitur (contoh: `pendapatan_tahunan`, `total_utang_saat_ini`, `bunga_tahunan`, `debt_to_income_ratio`, dll.)
* **Target Prediksi (y):** `status_kredit_encoded` 
    * **0:** Lancar (Mayoritas)
    * **1:** Berisiko (Minoritas)

---

## ⚙️ Alur Kerja (Workflow) Kodingan

### 1. Persiapan & Pembagian Data
* Dataset diimpor menggunakan `pandas`.
* Pemisahan fitur (X) dan target (y).
* Data dibagi menjadi **80% Data Training** (400.000 baris) dan **20% Data Testing** (100.000 baris) menggunakan parameter `stratify=y` untuk menjaga proporsi kelas.

### 2. Pelatihan Model Baseline (Random Forest)
* Melatih model `RandomForestClassifier` standar dengan 100 *estimators*.
* **Temuan Awal:** Terjadi fenomena *Accuracy Paradox*. Model mencapai akurasi tinggi (80%), namun gagal total mendeteksi kelas minoritas (Recall Kelas 1 = 0.00). Model cenderung hanya menebak Kelas 0.

### 3. Analisis Feature Importance
Menemukan 5 fitur yang paling berpengaruh terhadap prediksi risiko:
1.  Bunga Tahunan
2.  Total Utang Saat Ini
3.  Jumlah Pinjaman Diajukan
4.  Cicilan Bulanan (Monthly Installment)
5.  Pendapatan Tahunan

### 4. Perbaikan Model: Penanganan Imbalanced Data
Untuk mengatasi ketidakseimbangan kelas, dilakukan tiga strategi bertahap:
* **Strategi A (Random Forest + Class Weight & Tuning):** Menerapkan `class_weight='balanced'` dan melakukan *Hyperparameter Tuning* dengan `GridSearchCV`. Fokus metrik evaluasi diubah dari *Accuracy* menjadi `f1_macro`.
* **Strategi B (Custom Thresholding):** Menurunkan ambang batas (*threshold*) probabilitas prediksi untuk Kelas 1 dari 0.50 menjadi 0.40 agar model lebih sensitif.
* **Strategi C (Migrasi ke XGBoost):** Beralih dari algoritma *Bagging* ke *Boosting* dengan melatih `XGBClassifier`. Menambahkan parameter penalti `scale_pos_weight=4` untuk menyeimbangkan kepentingan antar kelas.

---

## 📈 Hasil Evaluasi

Berikut adalah rangkuman hasil performa dari setiap tahapan pemodelan pada data testing (100.000 baris):

| Model / Strategi | Accuracy | Recall Kelas 0 (Lancar) | Recall Kelas 1 (Berisiko) | F1-Score (Macro) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline Random Forest** | 80% | 1.00 | **0.00** | 0.45 |
| **RF Tuning (`f1_macro`)** | 48% | 0.47 | **0.53** | 0.44 |
| **RF (Threshold 0.40)** | 20% | 0.00 | **1.00** | 0.17 |
| **XGBoost (Final Model)** | 51% | 0.51 | **0.49** | **0.46** |

**Kesimpulan Evaluasi:**
Model **XGBoost** dipilih sebagai model final karena memberikan performa yang paling stabil dan masuk akal dalam menyeimbangkan kemampuan mendeteksi pelanggan berisiko (Recall 49%) tanpa mengorbankan terlalu banyak deteksi pada pelanggan lancar.

---

## 📁 File yang Dihasilkan
Proses pengembangan ini menghasilkan beberapa *file binary* model (`.pkl`) yang dapat diintegrasikan ke sistem *backend* (misalnya Streamlit):
1.  `model_risiko_pinjaman.pkl`: Model baseline awal.
2.  `model_risiko_pinjaman_fixed.pkl`: Model Random Forest hasil Hyperparameter Tuning.
3.  **`model_risiko_pinjaman_xgb.pkl`**: Model XGBoost final yang siap digunakan untuk *deployment*.

---

## 🚀 Cara Menjalankan (How to Run)
1. Pastikan seluruh *requirements* telah terpasang:
   ```bash
   pip install pandas numpy scikit-learn openpyxl xgboost
   Jalankan Jupyter Notebook untuk Eksperimen Model

Buka file model_development_classification.ipynb dan jalankan sel (Run All) secara berurutan.
 Jalankan Aplikasi Streamlit 
Jika model final (model_risiko_pinjaman_xgb.pkl) sudah dihubungkan ke antarmuka aplikasi Streamlit, kamu bisa menjalankannya dengan perintah:

Bash
streamlit run app.py