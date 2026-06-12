# 🚀 BijakPinjam: AI Loan Decision Support System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.0-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-1.2.0-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20.0-red)

**Tim:** PJK-GM032  
**Proyek:** Sistem Pendukung Keputusan Kelayakan Pinjaman (Klasifikasi Risiko)

---

## 📌 Deskripsi Proyek
**BijakPinjam** adalah purwarupa sistem berbasis *Machine Learning* yang dirancang untuk mengklasifikasikan tingkat risiko calon peminjam (Lancar vs. Macet/Berisiko). Tujuan utama dari proyek ini adalah membantu institusi finansial dalam mengambil keputusan kredit yang objektif sekaligus mencegah nasabah terjebak dalam masalah utang (*debt trap*).

Sistem ini didukung oleh antarmuka interaktif berbasis **Streamlit** yang memungkinkan analisis kelayakan kredit secara *real-time* dengan memproses profil finansial nasabah.

---

## ⚙️ Alur Pengembangan (Workflow)

Proyek ini dibangun melalui beberapa fase iteratif yang berfokus pada kualitas data dan ketajaman algoritma:

### 1. Data Ingestion & Rekayasa Logika Finansial
Dataset disimulasikan sebanyak **600.000 baris** (`Dataset_Bersih_Siap_ML_600k.xlsx`). Alih-alih membuat target kredit macet secara acak, target (`status_kredit`) digenerasi menggunakan **logika finansial nyata**, yaitu berdasarkan **Debt-to-Income Ratio (DTI)**. Semakin besar beban cicilan terhadap pendapatan, semakin tinggi probabilitas nasabah tersebut diklasifikasikan sebagai "Macet".

### 2. Feature Engineering & Preprocessing
Untuk memperkuat daya prediksi model, diekstrak 3 fitur finansial baru:
* **Monthly Installment:** Estimasi cicilan bulanan.
* **Debt-to-Income Ratio (DTI):** Rasio beban utang terhadap pendapatan bulanan.
* **Remaining Income:** Sisa pendapatan bersih nasabah.

Data kemudian diproses menggunakan `LabelEncoder` (untuk kolom kategorikal) dan `StandardScaler` (untuk menormalisasi skala kolom numerik seperti pendapatan dan nominal pinjaman).

### 3. Model Training & Evaluasi
Dataset dibagi menjadi **80% Data Training** dan **20% Data Testing** menggunakan metode stratifikasi.
* **Baseline Model (Random Forest):** Digunakan sebagai tolok ukur awal untuk melihat bagaimana fitur membedakan kelas target.
* **Final Model (XGBoost Classifier):** Digunakan sebagai model utama karena arsitektur *Boosting*-nya sangat tangguh dalam menangani data tabular. Model ini dikonfigurasi dengan parameter `scale_pos_weight=1.5` untuk memberikan keseimbangan dan fokus ekstra dalam mendeteksi nasabah berisiko.

### 4. Deployment
Model Machine Learning dan objek preprocessing diekspor, kemudian diintegrasikan ke dalam antarmuka **Streamlit**.

---

## 📊 Hasil Machine Learning

Berkat dataset yang memiliki logika finansial yang kuat (DTI), model tidak lagi mengalami *accuracy paradox* atau menebak secara buta. 

* **Model Terpilih:** XGBoost Classifier
* **Stabilitas:** Model mampu dengan sangat cerdas membaca batas ambang batas risiko (misalnya nasabah dengan DTI ringan < 30% langsung diklasifikasikan aman dengan probabilitas tinggi).
* **Feature Importance:** Faktor paling penentu dalam keputusan persetujuan kredit di model ini adalah **Rasio Beban Utang (DTI)**, **Cicilan Bulanan**, dan **Total Utang Saat Ini**.

---

## 📁 Struktur File & Artifacts

Proyek ini menghasilkan beberapa *file binary* yang krusial untuk *deployment*:
1. `model_risiko_pinjaman_xgb.pkl`: Otak utama model Machine Learning (XGBoost).
2. `scaler_finansial.joblib`: Objek standarisasi numerik.
3. `encoder_status_pekerjaan.joblib`: Objek pengubah teks kategori pekerjaan.
4. `encoder_status_kredit.joblib`: Objek pengubah target prediksi.
5. `app.py`: *Script* utama antarmuka Streamlit.

---

## 🚀 Cara Menjalankan Aplikasi (Local)

1. Pastikan seluruh *libraries* yang dibutuhkan telah terpasang:
   ```bash
   pip install pandas numpy scikit-learn xgboost streamlit joblib openpyxl