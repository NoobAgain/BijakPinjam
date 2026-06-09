import pandas as pd
import joblib

def hitung_risiko_pinjaman(pendapatan_bulanan, suku_bunga_baru, jumlah_pinjaman_baru, tenor_pinjaman_baru, status_pekerjaan, pengeluaran_bulanan=0, jumlah_pinjaman_sekarang=0, suku_bunga_sekarang=0, tenor_pinjaman_sekarang=0):
    # Cicilan bulanan
    bunga_desimal_baru = suku_bunga_baru / 100
    bunga_bulanan_baru = bunga_desimal_baru / 12
    monthly_installment = (jumlah_pinjaman_baru / tenor_pinjaman_baru) + (jumlah_pinjaman_baru * bunga_bulanan_baru)

    # Debt-to-Income Ratio (DTI)
    pendapatan_tahunan = pendapatan_bulanan * 12
    estimasi_cicilan_lama = jumlah_pinjaman_sekarang * 0.05
    debt_to_income_ratio = (estimasi_cicilan_lama + monthly_installment) / pendapatan_bulanan

    remaining_income = pendapatan_bulanan - (monthly_installment + estimasi_cicilan_lama)

    # Load encoder untuk status pekerjaan dan transformasi data kategorikal
    encoder_status_pekerjaan = joblib.load('./models/encoder_status_pekerjaan.joblib')
    status_pekerjaan_encoded = encoder_status_pekerjaan.transform([status_pekerjaan])[0]

    # Load scaler dan transformasi data numerik
    scaler = joblib.load('./models/scaler_finansial.joblib')
    kolom_kontinu = ['pendapatan_tahunan', 'total_utang_saat_ini', 'jumlah_pinjaman_diajukan', 'monthly_installment', 'debt_to_income_ratio', 'remaining_income']
    data_kontinu = pd.DataFrame([[pendapatan_tahunan, jumlah_pinjaman_sekarang, jumlah_pinjaman_baru, monthly_installment, debt_to_income_ratio, remaining_income]], columns=kolom_kontinu)
    data_kontinu_scaled = scaler.transform(data_kontinu)
    df_final = pd.DataFrame(data_kontinu_scaled, columns=kolom_kontinu)

    # Tambahkan kolom yang tidak diskalakan
    df_final["tenor_bulan"] = tenor_pinjaman_baru
    df_final["bunga_tahunan"] = bunga_desimal_baru
    df_final["status_pekerjaan_encoded"] = status_pekerjaan_encoded

    # Atur ulang kolom sesuai urutan 
    urutan_kolom = ['pendapatan_tahunan', 'total_utang_saat_ini', 'jumlah_pinjaman_diajukan', 'tenor_bulan', 'bunga_tahunan', 'monthly_installment', 'debt_to_income_ratio', 'remaining_income', 'status_pekerjaan_encoded']
    df_final = df_final[urutan_kolom]

    return df_final