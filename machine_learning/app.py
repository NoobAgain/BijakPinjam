import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. SETTING HALAMAN UTAMA
st.set_page_config(
    page_title="BijakPinjam - AI Loan Decision Support System",
    page_icon="💰",
    layout="centered"
)

# 2. LOAD MODEL XGBOOST YANG SUDAH DIOPTIMASI
@st.cache_resource
def load_model():
    # Memuat model final XGBoost yang telah kita simpan sebelumnya
    return joblib.load('model_risiko_pinjaman_xgb.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file 'model_risiko_pinjaman_xgb.pkl' ada di folder yang sama. Error: {e}")

# 3. HEADER UTAMA APLIKASI
st.title("💰 BijakPinjam")
st.subheader("AI Loan Decision Support System (Tim PJK-GM032)")
st.write(
    "Aplikasi berbasis AI untuk membantu menganalisis kondisi finansial Anda "
    "dan mendeteksi risiko pinjaman sebelum mengambil keputusan keuangan."
)
st.divider()

# 4. FORM INPUT DATA FINANSIAL PENGGUNA
st.header("📋 Formulir Analisis Finansial")
st.write("Silakan masukkan data keuangan Anda di bawah ini dengan jujur:")

# Membuat kolom input agar tampilan lebih rapi
col1, col2 = st.columns(2)

with col1:
    pendapatan_tahunan = st.number_input("Pendapatan Tahunan (Rupiah/Nilai Skala)", value=0.0, step=0.1)
    total_utang_saat_ini = st.number_input("Total Utang Saat Ini", value=0.0, step=0.1)
    jumlah_pinjaman_diajukan = st.number_input("Jumlah Pinjaman yang Diajukan", value=0.0, step=0.1)
    tenor_bulan = st.number_input("Tenor Pinjaman (Bulan)", min_value=1, max_value=120, value=12)
    bunga_tahunan = st.number_input("Bunga Tahunan (Kontrak/Persentase Skala)", value=0.0, step=0.01)

with col2:
    monthly_installment = st.number_input("Angsuran Bulanan (Monthly Installment)", value=0.0, step=0.1)
    debt_to_income_ratio = st.number_input("Rasio Utang terhadap Pendapatan (DTI)", value=0.0, step=0.1)
    remaining_income = st.number_input("Sisa Pendapatan Bersih (Remaining Income)", value=0.0, step=0.1)
    
    # Dropdown untuk fitur kategorikal yang sudah di-encode sebelumnya
    status_pekerjaan = st.selectbox("Status Pekerjaan", options=[0, 1, 2, 3], 
                                    format_func=lambda x: f"Kategori Pekerjaan {x}")
    status_kredit = st.selectbox("Status Kredit / Riwayat BI Checking", options=[0, 1, 2], 
                                 format_func=lambda x: f"Kategori Riwayat {x}")

st.divider()

# 5. PROSES PREDIKSI DAN OUTPUT REKOMENDASI
if st.button("🚀 Analisis Risiko Pinjaman", type="primary"):
    
    # Menyusun data input menjadi DataFrame dengan 9 fitur yang dilatih oleh model
    input_data = pd.DataFrame([{
        'pendapatan_tahunan': pendapatan_tahunan,
        'total_utang_saat_ini': total_utang_saat_ini,
        'jumlah_pinjaman_diajukan': jumlah_pinjaman_diajukan,
        'tenor_bulan': tenor_bulan,
        'bunga_tahunan': bunga_tahunan,
        'monthly_installment': monthly_installment,
        'debt_to_income_ratio': debt_to_income_ratio,
        'remaining_income': remaining_income,
        'status_pekerjaan_encoded': status_pekerjaan
    }])
    
    # Menjalankan fungsi prediksi model
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1] # Probabilitas risiko (Kelas 1)
    
    st.header("📊 Hasil Analisis AI BijakPinjam")
    
    # Menampilkan indikator probabilitas risiko
    st.metric(label="Tingkat Keyakinan Risiko AI", value=f"{prediction_proba * 100:.2f}%")
    
    # Aturan Output Rekomendasi
    if prediction == 1:
        st.error("🚨 **PERINGATAN: RISIKO TINGGI (BERISIKO GAGAL BAYAR)**")
        st.write(
            "**Rekomendasi Edukatif BijakPinjam:**\n"
            "Kondisi finansial Anda saat ini menunjukkan indikasi kuat kerentanan keuangan. "
            "Kami menyarankan Anda untuk **menunda pengajuan pinjaman ini** guna menghindari efek "
            "lingkaran setan utang (*debt trap*)."
        )
    else:
        st.success("✅ **KONDISI AMAN (RISIKO RENDAH)**")
        st.write(
            "**Rekomendasi Edukatif BijakPinjam:**\n"
            "Kondisi keuangan Anda dinilai sehat untuk mengambil pinjaman ini. "
            "Pastikan Anda menggunakan dana secara produktif dan membayar cicilan tepat waktu!"
        )