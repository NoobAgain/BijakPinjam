import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Konfigurasi Halaman Web
st.set_page_config(
    page_title="BijakPinjam - Credit Risk Assessment",
    page_icon="🚀",
    layout="centered"
)

# Kustomisasi Desain Header
st.title("🚀 AI Loan Decision Support System - BijakPinjam")
st.markdown("---")
st.markdown("### Aplikasi Analisis Risiko Kelayakan Pinjaman Nasabah")
st.write("Aplikasi ini menggunakan model XGBoost untuk memprediksi tingkat risiko nasabah secara real-time guna menghindari *debt trap*.")

# 2. Memuat File Artifacts (Model & Preprocessing Objects)
@st.cache_resource
def load_artifacts():
    # Memuat objek model dan preprocessing yang sudah diekspor sebelumnya
    model = joblib.load('model_risiko_pinjaman_xgb.pkl')
    scaler = joblib.load('scaler_finansial.joblib')
    encoder_pekerjaan = joblib.load('encoder_status_pekerjaan.joblib')
    encoder_kredit = joblib.load('encoder_status_kredit.joblib')
    return model, scaler, encoder_pekerjaan, encoder_kredit

try:
    model, scaler, encoder_pekerjaan, encoder_kredit = load_artifacts()
    st.sidebar.success("✅ Semua file model & preprocessing berhasil dimuat!")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat file (.pkl/.joblib). Pastikan file berada di folder yang sama.")
    st.error(f"Detail Error: {e}")
    st.stop()

# 3. Formulir Input Data Profil Finansial Nasabah
st.markdown("#### 📋 Masukkan Profil Finansial Nasabah")

col1, col2 = st.columns(2)

with col1:
    pendapatan_tahunan = st.number_input("Pendapatan Tahunan (Rp)", min_value=0, value=60000000, step=1000000)
    total_utang_saat_ini = st.number_input("Total Utang Saat Ini (Rp)", min_value=0, value=10000000, step=500000)
    jumlah_pinjaman_diajukan = st.number_input("Jumlah Pinjaman yang Diajukan (Rp)", min_value=0, value=15000000, step=500000)

with col2:
    tenor_bulan = st.selectbox("Tenor Pinjaman (Bulan)", [12, 24, 36, 48, 60], index=1)
    bunga_tahunan_persen = st.slider("Bunga Tahunan (%)", min_value=5.0, max_value=20.0, value=12.0, step=0.1)
    bunga_tahunan = bunga_tahunan_persen / 100
    status_pekerjaan = st.selectbox("Status Pekerjaan", ['Pegawai Tetap', 'Kontrak', 'Freelance', 'Wirausaha'])

st.markdown("---")

# 4. Eksekusi Prediksi saat Tombol di-Klik
if st.button("🔮 Analisis Risiko Kredit", use_container_width=True):
    
    # --- PROSES FEATURE ENGINEERING (Sama persis dengan notebook training) ---
    bunga_bulanan = bunga_tahunan / 12
    monthly_installment = (jumlah_pinjaman_diajukan / tenor_bulan) + (jumlah_pinjaman_diajukan * bunga_bulanan)
    
    pendapatan_bulanan = pendapatan_tahunan / 12
    estimasi_cicilan_lama = total_utang_saat_ini * 0.05
    
    debt_to_income_ratio = (monthly_installment + estimasi_cicilan_lama) / pendapatan_bulanan
    remaining_income = pendapatan_bulanan - (monthly_installment + estimasi_cicilan_lama)
    
    # --- PROSES ENCODING ---
    try:
        status_pekerjaan_encoded = encoder_pekerjaan.transform([status_pekerjaan])[0]
    except:
        status_pekerjaan_encoded = 0

    # --- MEMBENTUK DATAFRAME SESUAI URUTAN FITUR TRAINING ---
    input_data = pd.DataFrame([{
        'pendapatan_tahunan': pendapatan_tahunan,
        'total_utang_saat_ini': total_utang_saat_ini,
        'jumlah_pinjaman_diajukan': jumlah_pinjaman_diajukan,
        'tenor_bulan': tenor_bulan,
        'bunga_tahunan': bunga_tahunan,
        'monthly_installment': monthly_installment,
        'debt_to_income_ratio': debt_to_income_ratio,
        'remaining_income': remaining_income,
        'status_pekerjaan_encoded': status_pekerjaan_encoded
    }])

    # --- PROSES SCALING (Hanya pada kolom numerik) ---
    kolom_numerik = ['pendapatan_tahunan', 'total_utang_saat_ini', 'jumlah_pinjaman_diajukan', 
                     'monthly_installment', 'debt_to_income_ratio', 'remaining_income']
    
    input_data[kolom_numerik] = scaler.transform(input_data[kolom_numerik])

    # Memastikan urutan kolom tidak bergeser sebelum masuk ke model prediksi
    feature_order = [
        'pendapatan_tahunan', 'total_utang_saat_ini', 'jumlah_pinjaman_diajukan',
        'tenor_bulan', 'bunga_tahunan', 'monthly_installment', 
        'debt_to_income_ratio', 'remaining_income', 'status_pekerjaan_encoded'
    ]
    input_data = input_data[feature_order]

    # --- PROSES PREDIKSI MODEL ---
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    # Menentukan teks label status berdasarkan encoder
    try:
        target_label = encoder_kredit.inverse_transform([prediction])[0]
    except:
        target_label = "Lancar" if prediction == 0 else "Macet"

    # --- TAMPILAN OUTPUT HASIL EVALUASI ---
    st.markdown("### 📊 Hasil Penilaian Risiko")
    
    if prediction == 0:
        st.success(f"### 🎉 **REKOMENDASI: DISETUJUI ({target_label.upper()})**")
        st.write(f"Sistem menilai nasabah ini memiliki risiko rendah. Probabilitas Kelancaran: **{prediction_proba[0]*100:.2f}%**")
    else:
        st.error(f"### ⚠️ **REKOMENDASI: DITOLAK ({target_label.upper()})**")
        st.write(f"Sistem menilai nasabah ini berisiko tinggi mengalami gagal bayar. Probabilitas Macet: **{prediction_proba[1]*100:.2f}%**")

    # Menampilkan rincian metrik finansial hasil kalkulasi engine
    st.markdown("#### 🔍 Detail Metrik Finansial Hasil Kalkulasi Engine:")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Cicilan Bulanan (Estimasi)", f"Rp {monthly_installment:,.0f}")
    col_m2.metric("Rasio Beban Utang (DTI)", f"{debt_to_income_ratio:.2f}")
    col_m3.metric("Sisa Pendapatan Bersih", f"Rp {remaining_income:,.0f}")
    
    # Alert system tambahan berdasarkan teori finansial
    if debt_to_income_ratio > 0.45:
        st.warning("⚠️ **Peringatan Sistem:** Rasio Debt-to-Income (DTI) nasabah melebihi batas aman (45%). Beban cicilan terlalu berat dibandingkan total pendapatan bulanan.")