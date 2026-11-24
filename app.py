import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from difflib import SequenceMatcher
from io import BytesIO
import os
import time
from google import genai
from google.genai import types

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS
# ==========================================
st.set_page_config(
    page_title="AuditInsight - Multi-Module AI Audit",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {font-size: 2.1rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0;}
    .sub-header {font-size: 1rem; color: #64748B; margin-bottom: 1.2rem;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI BANTU & PEMBERSIHAN DATA
# ==========================================
def clean_currency(val):
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    val_str = str(val).replace('Rp', '').replace('rp', '').replace(' ', '').strip()
    is_negative = False
    if val_str.startswith('(') and val_str.endswith(')'):
        is_negative = True
        val_str = val_str[1:-1].strip()
    if val_str.startswith('-'):
        is_negative = True
        val_str = val_str[1:].strip()

    if ',' in val_str and '.' in val_str:
        if val_str.rfind(',') > val_str.rfind('.'):
            val_str = val_str.replace('.', '').replace(',', '.')
        else:
            val_str = val_str.replace(',', '')
    elif '.' in val_str:
        parts = val_str.split('.')
        if len(parts[-1]) == 3:
            val_str = val_str.replace('.', '')
    elif ',' in val_str:
        parts = val_str.split(',')
        if len(parts[-1]) == 3:
            val_str = val_str.replace(',', '')
        else:
            val_str = val_str.replace(',', '.')
    try:
        num = float(val_str)
        return -num if is_negative else num
    except ValueError:
        return 0.0

def find_similar_vendors(vendors):
    unique_vendors = list(set([str(v).strip() for v in vendors if pd.notna(v)]))
    similar_map = {}
    for i in range(len(unique_vendors)):
        for j in range(i + 1, len(unique_vendors)):
            v1, v2 = unique_vendors[i], unique_vendors[j]
            if v1.lower() != v2.lower():
                ratio = SequenceMatcher(None, v1.lower(), v2.lower()).ratio()
                if ratio >= 0.85:
                    similar_map[v1] = v2
                    similar_map[v2] = v1
    return similar_map

def benford_analysis(amounts):
    valid_amounts = [abs(a) for a in amounts if abs(a) >= 10]
    if len(valid_amounts) < 10:
        return None, set()
    first_digits = [int(str(int(a))[0]) for a in valid_amounts]
    total = len(first_digits)
    benford_expected = {d: np.log10(1 + 1/d) * 100 for d in range(1, 10)}
    actual_counts = {d: first_digits.count(d) for d in range(1, 10)}
    actual_pct = {d: (actual_counts[d] / total) * 100 for d in range(1, 10)}
    anomalous_digits = {d for d in range(1, 10) if (actual_pct[d] - benford_expected[d]) > 5.0}
    df_benford = pd.DataFrame({
        'Digit': list(range(1, 10)),
        'Aktual (%)': [round(actual_pct[d], 2) for d in range(1, 10)],
        'Benford (%)': [round(benford_expected[d], 2) for d in range(1, 10)]
    })
    return df_benford, anomalous_digits

def to_excel_bytes(df_export, sheet_name="Hasil_Audit"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_export.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()

def get_gemini_response(api_key, model_name, system_instruction, messages, max_retries=5):
    client = genai.Client(api_key=api_key)
    contents = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=m["content"])]
            )
        )

    fallback_models = [model_name, "gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-pro"]
    models_to_try = []
    for m_name in fallback_models:
        if m_name not in models_to_try:
            models_to_try.append(m_name)

    last_error = None
    for current_model in models_to_try:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=current_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.3,
                        max_output_tokens=1600
                    )
                )
                return response.text or "Tidak ada respons teks yang dihasilkan oleh Gemini."
            except Exception as e:
                last_error = e
                err_str = str(e)
                is_temporary = ("503" in err_str) or ("UNAVAILABLE" in err_str) or ("429" in err_str)
                is_not_found = ("404" in err_str) or ("NOT_FOUND" in err_str)
                if is_not_found:
                    break
                if is_temporary and attempt < max_retries - 1:
                    time.sleep(min(2 ** (attempt + 1), 16))
                    continue
                break
    raise last_error

def get_preset_audit_answer(question_key, mode_name, ai_context_summary, top_str):
    header = f"**Analisis Terarah ({mode_name})**\n\n- **Ringkasan Data:** {ai_context_summary}\n\n"
    if question_key == "Ringkasan Temuan Utama & Penyebab Non-Fraud":
        return (
            header
            + "1. **Indikasi Risiko / Anomali Utama:** Fokuskan pemeriksaan pada baris dengan skor risiko tertinggi dan nilai/perubahan paling mencolok pada cuplikan prioritas di bawah.\n"
            + "2. **Kemungkinan Penyebab Non-Fraud:** Kesalahan input/format angka, klasifikasi akun atau pos yang kurang tepat, perbedaan waktu pencatatan (*cut-off*), atau kondisi operasional yang memang sah.\n"
            + "3. **Prosedur Audit Lanjutan:** Bandingkan baris prioritas dengan dokumen pendukung asli, konfirmasi ke unit/bagian terkait, lalu catat hasil verifikasi pada kolom *Keputusan Auditor* di kertas kerja Excel.\n\n"
            + f"```\n{top_str}\n```"
        )
    elif question_key == "Prosedur Verifikasi Dokumen & Bukti Pendukung":
        return (
            header
            + "1. **Uji Keberadaan & Keterjadian:** Telusuri transaksi/akun berisiko tinggi ke dokumen sumber (faktur, kuitansi, kontrak, berita acara, atau bukti persetujuan).\n"
            + "2. **Uji Kewajaran & Otorisasi:** Pastikan transaksi/realisasi disetujui oleh pihak berwenang sesuai batas wewenang dan tidak terpecah-pecah tanpa dasar yang jelas.\n"
            + "3. **Dokumentasi Kertas Kerja:** Cantumkan nomor bukti yang diperiksa serta status temuan (wajar, salah saji/koreksi, atau perlu klarifikasi lanjutan)."
        )
    elif question_key == "Prioritas Tindak Lanjut Auditor":
        return (
            header
            + "1. **Prioritas 1 (Risiko Tinggi / Melampaui Batas):** Periksa lebih dulu baris dengan skor tertinggi, nilai terbesar, atau penyimpangan paling tajam.\n"
            + "2. **Prioritas 2 (Risiko Sedang / Perlu Perhatian):** Tinjau pola berulang, nominal bulat, atau akun dengan perubahan signifikan.\n"
            + "3. **Prioritas 3 (Risiko Rendah / Wajar):** Lakukan *sampling* ringan atau cukup pantau bila ada perubahan parameter aturan di *sidebar*."
        )
    else:
        return (
            header
            + "1. **Cek Kesalahan Teknis:** Pastikan pemetaan kolom sudah sesuai dengan struktur Excel dan pembersihan angka (`clean_currency`) membaca format ribuan/desimal dengan benar.\n"
            + "2. **Cek Kebijakan & Klasifikasi:** Verifikasi apakah terdapat reklasifikasi akun, perubahan anggaran/pagu, atau transaksi musiman yang memengaruhi hasil.\n"
            + "3. **Kesimpulan Sementara:** Perlakukan setiap *red flag* sebagai indikator pemeriksaan awal sebelum menarik kesimpulan audit."
        )

def auto_match_col(columns, keywords):
    for col in columns:
        c_low = str(col).lower()
        for kw in keywords:
            if kw in c_low:
                return col
    return columns[0] if columns else None

# ==========================================
# 3. MESIN AUDIT MODUL 1: TRANSAKSI
# ==========================================
def run_transaction_audit(df_raw, col_tgl, col_vnd, col_fak, col_jml, col_crt, col_app, col_npwp, auth_limit, round_mult, contamination):
    df = df_raw.copy()
    df['Baris Excel'] = df.index + 2
    df['Tanggal_Parsed'] = pd.to_datetime(df[col_tgl], errors='coerce')
    df['Vendor_Clean'] = df[col_vnd].astype(str).str.strip()
    df['Faktur_Clean'] = df[col_fak].astype(str).str.strip()
    df['Jumlah_Clean'] = df[col_jml].apply(clean_currency)

    dup_faktur = df[df.duplicated(subset=['Faktur_Clean'], keep=False)]['Faktur_Clean'].unique()
    dup_exact = df[df.duplicated(subset=['Vendor_Clean', 'Jumlah_Clean', 'Tanggal_Parsed'], keep=False)].index

    df_sorted = df.sort_values('Tanggal_Parsed')
    dup_7days_idx = set()
    for vendor, group in df_sorted.groupby('Vendor_Clean'):
        records = group[['Tanggal_Parsed', 'Jumlah_Clean']].to_dict('records')
        indices = group.index.tolist()
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                if pd.notna(records[i]['Tanggal_Parsed']) and pd.notna(records[j]['Tanggal_Parsed']):
                    diff_days = abs((records[j]['Tanggal_Parsed'] - records[i]['Tanggal_Parsed']).days)
                    if diff_days <= 7 and records[i]['Jumlah_Clean'] == records[j]['Jumlah_Clean'] and records[i]['Jumlah_Clean'] > 0:
                        dup_7days_idx.add(indices[i])
                        dup_7days_idx.add(indices[j])

    daily_vendor_sum = df.groupby(['Vendor_Clean', 'Tanggal_Parsed'])['Jumlah_Clean'].transform('sum')
    daily_vendor_cnt = df.groupby(['Vendor_Clean', 'Tanggal_Parsed'])['Jumlah_Clean'].transform('count')
    split_tx_idx = set(df[(daily_vendor_cnt > 1) & (daily_vendor_sum > auth_limit) & (df['Jumlah_Clean'] <= auth_limit)].index)

    vendor_counts = df['Vendor_Clean'].value_counts()
    one_time_vendors = set(vendor_counts[vendor_counts == 1].index)
    similar_vendor_map = find_similar_vendors(df['Vendor_Clean'].tolist())

    df_benford, anomalous_digits = benford_analysis(df['Jumlah_Clean'].tolist())

    if len(df) >= 10:
        iso = IsolationForest(contamination=contamination, random_state=42)
        df['Outlier_Pred'] = iso.fit_predict(df[['Jumlah_Clean']].fillna(0))
    else:
        df['Outlier_Pred'] = 1

    scores, reasons_list, all_flags = [], [], []
    for idx, row in df.iterrows():
        score = 0
        reasons = []
        amt = row['Jumlah_Clean']
        tgl = row['Tanggal_Parsed']
        vnd = row['Vendor_Clean']
        fak = row['Faktur_Clean']

        if fak in dup_faktur and fak.lower() not in ['nan', '', 'none', '-']:
            score += 35
            reasons.append("Duplikat No Faktur")
            all_flags.append("Duplikat No Faktur")
        if idx in dup_exact:
            score += 30
            reasons.append("Transaksi Identik (Vendor, Jumlah, Tanggal)")
            all_flags.append("Transaksi Identik")
        if idx in dup_7days_idx and idx not in dup_exact:
            score += 20
            reasons.append("Nominal & Vendor Sama dalam <=7 Hari")
            all_flags.append("Duplikat <=7 Hari")
        if idx in split_tx_idx:
            score += 25
            reasons.append("Indikasi Split Transaction")
            all_flags.append("Split Transaction")
        if amt > auth_limit:
            score += 20
            reasons.append(f"Melebihi Batas Otorisasi (> Rp {auth_limit:,.0f})")
            all_flags.append("Melebihi Batas Otorisasi")
        elif 0.90 * auth_limit <= amt <= auth_limit:
            score += 15
            reasons.append("Mendekati Batas Otorisasi (90-100%)")
            all_flags.append("Mendekati Batas Otorisasi")
        if amt > 0 and round_mult > 0 and (amt % round_mult == 0):
            score += 10
            reasons.append(f"Nominal Bulat Kelipatan Rp {round_mult:,.0f}")
            all_flags.append("Nominal Bulat")
        if pd.notna(tgl) and tgl.weekday() >= 5:
            score += 15
            reasons.append("Transaksi Akhir Pekan")
            all_flags.append("Transaksi Akhir Pekan")
        if vnd in one_time_vendors and amt >= (0.5 * auth_limit):
            score += 10
            reasons.append("Vendor Sekali Pakai Bernilai Besar")
            all_flags.append("Vendor Sekali Pakai")
        if vnd in similar_vendor_map:
            score += 15
            reasons.append(f"Nama Vendor Mirip '{similar_vendor_map[vnd]}'")
            all_flags.append("Nama Vendor Mirip")
        if col_crt != "(Tidak Ada)" and col_app != "(Tidak Ada)":
            c = str(row[col_crt]).strip().lower()
            a = str(row[col_app]).strip().lower()
            if c == a and c not in ['nan', '', 'none', '-']:
                score += 30
                reasons.append("Self-Approval (Pembuat & Penyetuju Sama)")
                all_flags.append("Self-Approval")
        if col_npwp != "(Tidak Ada)":
            npwp_val = str(row[col_npwp]).strip().lower()
            if npwp_val in ['nan', '', 'none', '-', '0']:
                score += 10
                reasons.append("Vendor Tanpa NPWP")
                all_flags.append("Tanpa NPWP")
        if row['Outlier_Pred'] == -1:
            score += 15
            reasons.append("Anomali Statistik (Isolation Forest)")
            all_flags.append("ML Outlier")
        if abs(amt) >= 10:
            first_digit = int(str(int(abs(amt)))[0])
            if first_digit in anomalous_digits:
                score += 5
                reasons.append(f"Digit Awal ({first_digit}) Menyimpang dari Benford")
                all_flags.append("Penyimpangan Benford")

        final_score = min(score, 100)
        scores.append(final_score)
        reasons_list.append("; ".join(reasons) if reasons else "Normal (Tidak ada red flag)")

    df['Skor Risiko'] = scores
    df['Level Risiko'] = df['Skor Risiko'].apply(lambda s: 'Tinggi' if s >= 60 else ('Sedang' if s >= 30 else 'Rendah'))
    df['Alasan Red Flag'] = reasons_list
    df_res = df.sort_values(by=['Skor Risiko', 'Jumlah_Clean'], ascending=[False, False]).reset_index(drop=True)
    return df_res, df_benford, all_flags

# ==========================================
# 4. MESIN ANALISIS MODUL 2: NERACA & LABA RUGI
# ==========================================
def run_fs_analysis(df_raw, col_akun, col_kat, col_curr, col_prev, spike_threshold):
    df = df_raw.copy()
    df['Baris Excel'] = df.index + 2
    df['Akun_Clean'] = df[col_akun].astype(str).str.strip()
    df['Nilai_Kini'] = df[col_curr].apply(clean_currency)
    has_prev = (col_prev != "(Tidak Ada)")
    if has_prev:
        df['Nilai_Lalu'] = df[col_prev].apply(clean_currency)
        df['Selisih_Rp'] = df['Nilai_Kini'] - df['Nilai_Lalu']
        df['Perubahan_%'] = np.where(
            df['Nilai_Lalu'] != 0,
            ((df['Nilai_Kini'] - df['Nilai_Lalu']) / df['Nilai_Lalu'].abs()) * 100,
            np.where(df['Nilai_Kini'] != 0, 100.0, 0.0)
        )
    else:
        df['Nilai_Lalu'] = 0.0
        df['Selisih_Rp'] = 0.0
        df['Perubahan_%'] = 0.0

    total_abs = df['Nilai_Kini'].abs().sum()
    df['Proporsi_%'] = np.where(total_abs > 0, (df['Nilai_Kini'].abs() / total_abs) * 100, 0.0)

    scores, flags = [], []
    for _, r in df.iterrows():
        sc = 0
        fl = []
        if has_prev and abs(r['Perubahan_%']) >= spike_threshold:
            sc += 45
            fl.append(f"Perubahan Tajam ({r['Perubahan_%']:+.1f}%)")
        if r['Proporsi_%'] >= 25.0:
            sc += 25
            fl.append(f"Konsentrasi Nilai Tinggi ({r['Proporsi_%']:.1f}%)")
        if r['Nilai_Kini'] < 0 and col_kat != "(Tidak Ada)":
            kat_low = str(r[col_kat]).lower()
            if any(k in kat_low for k in ['aset', 'kas', 'piutang', 'persediaan', 'pendapatan']):
                sc += 30
                fl.append("Saldo Negatif Tidak Lazim pada Akun Debit/Pendapatan")
        scores.append(min(sc, 100))
        flags.append("; ".join(fl) if fl else "Wajar")

    df['Skor Anomali'] = scores
    df['Status Review'] = df['Skor Anomali'].apply(lambda x: 'Perlu Diperiksa' if x >= 45 else ('Perhatian' if x >= 25 else 'Normal'))
    df['Catatan Anomali'] = flags

    summary_info = {}
    if col_kat != "(Tidak Ada)":
        df['Kat_Clean'] = df[col_kat].astype(str).str.strip()
        kat_sum = df.groupby('Kat_Clean')[['Nilai_Kini', 'Nilai_Lalu']].sum()
        summary_info['by_cat'] = kat_sum

        def sum_matching(keywords):
            total = 0.0
            for idx_k, val_k in kat_sum['Nilai_Kini'].items():
                if any(k in str(idx_k).lower() for k in keywords):
                    total += val_k
            return total

        aset = sum_matching(['aset', 'aktiva', 'asset'])
        liabilitas = sum_matching(['liabilitas', 'kewajiban', 'hutang', 'utang', 'liability'])
        ekuitas = sum_matching(['ekuitas', 'modal', 'equity'])
        pendapatan = sum_matching(['pendapatan', 'penjualan', 'revenue', 'income'])
        beban = sum_matching(['beban', 'biaya', 'expense'])
        kas = sum_matching(['kas', 'bank', 'cash'])

        summary_info['aset'] = aset
        summary_info['liabilitas'] = liabilitas
        summary_info['ekuitas'] = ekuitas
        summary_info['selisih_neraca'] = aset - (liabilitas + ekuitas)
        summary_info['pendapatan'] = pendapatan
        summary_info['beban'] = beban
        summary_info['laba_bersih'] = pendapatan - beban
        summary_info['kas'] = kas

    return df.sort_values(by=['Skor Anomali', 'Nilai_Kini'], ascending=[False, False]).reset_index(drop=True), summary_info

# ==========================================
# 5. MESIN ANALISIS MODUL 3: ANGGARAN VS REALISASI
# ==========================================
def run_budget_analysis(df_raw, col_pos, col_pagu, col_real, round_mult):
    df = df_raw.copy()
    df['Baris Excel'] = df.index + 2
    df['Pos_Clean'] = df[col_pos].astype(str).str.strip()
    df['Pagu_Clean'] = df[col_pagu].apply(clean_currency)
    df['Real_Clean'] = df[col_real].apply(clean_currency)
    df['Selisih_Anggaran'] = df['Pagu_Clean'] - df['Real_Clean']
    df['Penyerapan_%'] = np.where(df['Pagu_Clean'] > 0, (df['Real_Clean'] / df['Pagu_Clean']) * 100, 0.0)

    scores, statuses, notes = [], [], []
    for _, r in df.iterrows():
        sc = 0
        nt = []
        pct = r['Penyerapan_%']
        real = r['Real_Clean']
        pagu = r['Pagu_Clean']

        if pagu == 0 and real > 0:
            sc += 75
            nt.append("Realisasi Muncul Tanpa Pagu Anggaran")
        elif pct > 100.0:
            sc += 70
            nt.append(f"Melampaui Pagu ({pct:.1f}%)")
        elif abs(pct - 100.0) < 0.01 and real > 0:
            sc += 25
            nt.append("Penyerapan Pas 100.00% (Perlu Cek Kewajaran)")
        elif 0 < pct < 20.0:
            sc += 20
            nt.append(f"Penyerapan Sangat Rendah ({pct:.1f}%)")

        if real > 0 and round_mult > 0 and (real % round_mult == 0):
            sc += 15
            nt.append(f"Realisasi Bulat Kelipatan Rp {round_mult:,.0f}")

        scores.append(min(sc, 100))
        statuses.append('Tinggi' if sc >= 60 else ('Sedang' if sc >= 25 else 'Rendah'))
        notes.append("; ".join(nt) if nt else "Dalam Batas Wajar")

    df['Skor Risiko'] = scores
    df['Level Risiko'] = statuses
    df['Temuan Anggaran'] = notes
    return df.sort_values(by=['Skor Risiko', 'Real_Clean'], ascending=[False, False]).reset_index(drop=True)

# ==========================================
# 6. SIDEBAR: KONFIGURASI & GEMINI AI
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/financial-analytics.png", width=56)
    st.title("AuditInsight")
    mode = st.radio(
        "Pilih Jenis Data / Modul:",
        options=[
            "Transaksi & Pembayaran",
            "Neraca & Laba Rugi",
            "Anggaran vs Realisasi"
        ]
    )
    st.divider()

    if mode == "Transaksi & Pembayaran":
        st.subheader("Parameter Aturan Transaksi")
        auth_limit = st.number_input("Batas Otorisasi (Rp)", min_value=100000, value=50000000, step=5000000, format="%d")
        round_mult = st.number_input("Kelipatan Nominal Bulat (Rp)", min_value=10000, value=1000000, step=100000, format="%d")
        contamination = st.slider("Sensitivitas Outlier (ML)", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    elif mode == "Neraca & Laba Rugi":
        st.subheader("Parameter Laporan Keuangan")
        spike_threshold = st.slider("Ambang Lonjakan Perubahan (%)", min_value=15, max_value=200, value=50, step=5)
    else:
        st.subheader("Parameter Anggaran")
        round_mult = st.number_input("Kelipatan Realisasi Bulat (Rp)", min_value=10000, value=1000000, step=100000, format="%d")

    st.divider()
    st.subheader("Konfigurasi AI (Gemini)")
    default_api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    gemini_api_key = st.text_input("Gemini API Key", value=default_api_key, type="password")
    model_choice = st.selectbox(
        "Model Gemini",
        options=["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
        index=0
    )

# ==========================================
# 7. HALAMAN UTAMA & UNGGAH FILE
# ==========================================
st.markdown('<p class="main-header">🔍 AuditInsight Multi-Modul</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">Mode Aktif: <b>{mode}</b> — Didukung Gemini AI. Unggah file Excel (.xlsx) Anda langsung tanpa perlu memakai template khusus.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Unggah File Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        selected_sheet = st.selectbox("Pilih Sheet di File Excel:", options=sheet_names, index=0)
        df_raw = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        df_raw.columns = [str(c).strip() for c in df_raw.columns]

        if df_raw.empty or len(df_raw.columns) == 0:
            st.error("Sheet yang dipilih kosong.")
            st.stop()

        cols = list(df_raw.columns)
        none_opt = ["(Tidak Ada)"]

        # --------------------------------------
        # MODUL 1: TRANSAKSI & PEMBAYARAN
        # --------------------------------------
        if mode == "Transaksi & Pembayaran":
            with st.expander("🗂️ Pemetaan Kolom File Anda (Sesuaikan Jika Nama Kolom Berbeda)", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                col_tgl = c1.selectbox("Kolom Tanggal *", cols, index=cols.index(auto_match_col(cols, ['tgl', 'tanggal', 'date'])))
                col_vnd = c2.selectbox("Kolom Vendor / Penerima *", cols, index=cols.index(auto_match_col(cols, ['vendor', 'supplier', 'rekanan', 'penerima', 'nama'])))
                col_fak = c3.selectbox("Kolom No Faktur / Dokumen *", cols, index=cols.index(auto_match_col(cols, ['faktur', 'inv', 'invoice', 'dok', 'sp2d', 'no'])))
                col_jml = c4.selectbox("Kolom Jumlah / Nominal *", cols, index=cols.index(auto_match_col(cols, ['jumlah', 'nilai', 'nominal', 'amount', 'total', 'realisasi'])))

                o1, o2, o3 = st.columns(3)
                col_crt = o1.selectbox("Kolom Dibuat Oleh (Opsional)", none_opt + cols, index=0)
                col_app = o2.selectbox("Kolom Disetujui Oleh (Opsional)", none_opt + cols, index=0)
                col_npwp = o3.selectbox("Kolom NPWP / Identitas (Opsional)", none_opt + cols, index=0)

            df_res, df_benford, all_flags = run_transaction_audit(
                df_raw, col_tgl, col_vnd, col_fak, col_jml, col_crt, col_app, col_npwp,
                auth_limit, round_mult, contamination
            )

            high_df = df_res[df_res['Level Risiko'] == 'Tinggi']
            med_df = df_res[df_res['Level Risiko'] == 'Sedang']
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Transaksi", f"{len(df_res):,}")
            k2.metric("Risiko Tinggi", f"{len(high_df):,}")
            k3.metric("Risiko Sedang", f"{len(med_df):,}")
            k4.metric("Nilai Risiko Tinggi", f"Rp {high_df['Jumlah_Clean'].sum():,.0f}")

            st.divider()
            t1, t2, t3, t4 = st.tabs(["📊 Grafik", "📋 Tabel Temuan", "🤖 Analisis & Pilihan Tanya Jawab", "📥 Unduh Excel"])

            with t1:
                g1, g2 = st.columns(2)
                with g1:
                    st.plotly_chart(px.histogram(df_res, x="Skor Risiko", color="Level Risiko", nbins=20, title="Sebaran Skor Risiko"), use_container_width=True)
                with g2:
                    if all_flags:
                        fc = pd.Series(all_flags).value_counts().reset_index()
                        fc.columns = ['Red Flag', 'Jumlah']
                        st.plotly_chart(px.bar(fc, x='Jumlah', y='Red Flag', orientation='h', title="Frekuensi Red Flag"), use_container_width=True)
                if df_benford is not None:
                    fb = go.Figure()
                    fb.add_trace(go.Bar(x=df_benford['Digit'], y=df_benford['Aktual (%)'], name='Aktual (%)'))
                    fb.add_trace(go.Scatter(x=df_benford['Digit'], y=df_benford['Benford (%)'], name='Benford (%)', mode='lines+markers'))
                    fb.update_layout(title="Uji Benford's Law")
                    st.plotly_chart(fb, use_container_width=True)

            with t2:
                lvl_filter = st.multiselect("Filter Level Risiko", ['Tinggi', 'Sedang', 'Rendah'], default=['Tinggi', 'Sedang'])
                show_cols = ['Baris Excel', col_tgl, col_vnd, col_fak, col_jml, 'Skor Risiko', 'Level Risiko', 'Alasan Red Flag']
                st.dataframe(df_res[df_res['Level Risiko'].isin(lvl_filter)][show_cols], use_container_width=True, hide_index=True)

            with t3:
                ai_context_summary = f"Modul: Transaksi & Pembayaran. Total: {len(df_res)}, Tinggi: {len(high_df)}, Sedang: {len(med_df)}, Nilai Tinggi: Rp {high_df['Jumlah_Clean'].sum():,.0f}."
                top_str = df_res[['Baris Excel', col_vnd, col_fak, col_jml, 'Skor Risiko', 'Level Risiko', 'Alasan Red Flag']].head(15).to_string(index=False)

            with t4:
                exp_df = df_res.drop(columns=['Tanggal_Parsed', 'Vendor_Clean', 'Faktur_Clean', 'Jumlah_Clean', 'Outlier_Pred'], errors='ignore')
                exp_df['Keputusan Auditor'] = ""
                exp_df['Catatan Auditor'] = ""
                st.download_button("📥 Unduh Kertas Kerja (.xlsx)", data=to_excel_bytes(exp_df, "Audit_Transaksi"), file_name=f"AuditInsight_Transaksi_{uploaded_file.name}")

        # --------------------------------------
        # MODUL 2: NERACA & LABA RUGI
        # --------------------------------------
        elif mode == "Neraca & Laba Rugi":
            with st.expander("🗂️ Pemetaan Kolom Laporan Keuangan (Neraca / Laba Rugi)", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                col_akun = c1.selectbox("Kolom Nama / Kode Akun *", cols, index=cols.index(auto_match_col(cols, ['akun', 'uraian', 'keterangan', 'account', 'pos'])))
                col_kat = c2.selectbox("Kolom Kelompok / Kategori (Opsional)", none_opt + cols, index=(none_opt + cols).index(auto_match_col(cols, ['kelompok', 'kategori', 'jenis', 'tipe', 'grup'])) if auto_match_col(cols, ['kelompok', 'kategori', 'jenis', 'tipe', 'grup']) else 0)
                col_curr = c3.selectbox("Kolom Nilai Periode Berjalan *", cols, index=cols.index(auto_match_col(cols, ['2026', '2025', 'berjalan', 'kini', 'realisasi', 'saldo', 'jumlah', 'nilai'])))
                col_prev = c4.selectbox("Kolom Nilai Periode Lalu (Opsional)", none_opt + cols, index=0)

            df_fs, info_fs = run_fs_analysis(df_raw, col_akun, col_kat, col_curr, col_prev, spike_threshold)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Jumlah Baris Akun", f"{len(df_fs):,}")
            k2.metric("Perlu Diperiksa", f"{len(df_fs[df_fs['Status Review']=='Perlu Diperiksa']):,}")
            if 'selisih_neraca' in info_fs:
                k3.metric("Selisih (Aset - [Liab+Ekuitas])", f"Rp {info_fs['selisih_neraca']:,.0f}")
            else:
                k3.metric("Total Nilai Periode Ini", f"Rp {df_fs['Nilai_Kini'].sum():,.0f}")
            if 'laba_bersih' in info_fs:
                k4.metric("Laba / (Rugi) Bersih", f"Rp {info_fs['laba_bersih']:,.0f}")
            else:
                k4.metric("Perubahan Rata-Rata (%)", f"{df_fs['Perubahan_%'].mean():+.1f}%")

            st.divider()
            t1, t2, t3, t4 = st.tabs(["📊 Grafik & Struktur", "📋 Tabel Akun", "🤖 Analisis & Pilihan Tanya Jawab", "📥 Unduh Excel"])

            with t1:
                st.plotly_chart(px.bar(df_fs.head(15), x=col_akun, y=['Nilai_Kini', 'Nilai_Lalu'], barmode='group', title="Top 15 Akun: Periode Berjalan vs Periode Lalu"), use_container_width=True)
                if 'by_cat' in info_fs:
                    st.dataframe(info_fs['by_cat'].reset_index(), use_container_width=True, hide_index=True)

            with t2:
                fs_cols = ['Baris Excel', col_akun] + ([col_kat] if col_kat != "(Tidak Ada)" else []) + ['Nilai_Lalu', 'Nilai_Kini', 'Selisih_Rp', 'Perubahan_%', 'Skor Anomali', 'Status Review', 'Catatan Anomali']
                st.dataframe(df_fs[fs_cols], use_container_width=True, hide_index=True)

            with t3:
                ai_context_summary = f"Modul: Neraca & Laba Rugi. Jumlah akun: {len(df_fs)}, Perlu diperiksa: {len(df_fs[df_fs['Status Review']=='Perlu Diperiksa'])}, Info ringkas: { {k: v for k, v in info_fs.items() if k != 'by_cat'} }."
                top_str = df_fs[['Baris Excel', col_akun, 'Nilai_Lalu', 'Nilai_Kini', 'Perubahan_%', 'Status Review', 'Catatan Anomali']].head(15).to_string(index=False)

            with t4:
                exp_fs = df_fs.drop(columns=['Akun_Clean', 'Nilai_Kini', 'Nilai_Lalu', 'Selisih_Rp', 'Perubahan_%', 'Proporsi_%', 'Skor Anomali', 'Kat_Clean'], errors='ignore')
                exp_fs['Keputusan Auditor'] = ""
                exp_fs['Catatan Auditor'] = ""
                st.download_button("📥 Unduh Hasil Analisis LK (.xlsx)", data=to_excel_bytes(exp_fs, "Analisis_LK"), file_name=f"AuditInsight_LK_{uploaded_file.name}")

        # --------------------------------------
        # MODUL 3: ANGGARAN VS REALISASI
        # --------------------------------------
        else:
            with st.expander("🗂️ Pemetaan Kolom Anggaran vs Realisasi", expanded=True):
                c1, c2, c3 = st.columns(3)
                col_pos = c1.selectbox("Kolom Pos / Akun / Program *", cols, index=cols.index(auto_match_col(cols, ['pos', 'akun', 'uraian', 'program', 'kegiatan', 'nama'])))
                col_pagu = c2.selectbox("Kolom Pagu / Anggaran *", cols, index=cols.index(auto_match_col(cols, ['pagu', 'anggaran', 'budget', 'plafon', 'target'])))
                col_real = c3.selectbox("Kolom Realisasi *", cols, index=cols.index(auto_match_col(cols, ['realisasi', 'real', 'terpakai', 'serapan', 'aktual'])))

            df_bud = run_budget_analysis(df_raw, col_pos, col_pagu, col_real, round_mult)
            over_df = df_bud[df_bud['Penyerapan_%'] > 100.0]

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Pos Anggaran", f"{len(df_bud):,}")
            k2.metric("Total Pagu", f"Rp {df_bud['Pagu_Clean'].sum():,.0f}")
            k3.metric("Total Realisasi", f"Rp {df_bud['Real_Clean'].sum():,.0f}")
            k4.metric("Pos Melampaui Pagu", f"{len(over_df):,}")

            st.divider()
            t1, t2, t3, t4 = st.tabs(["📊 Grafik Penyerapan", "📋 Tabel Anggaran", "🤖 Analisis & Pilihan Tanya Jawab", "📥 Unduh Excel"])

            with t1:
                st.plotly_chart(px.bar(df_bud.head(15), x=col_pos, y='Penyerapan_%', color='Level Risiko', title="Persentase Penyerapan Anggaran (Top 15)"), use_container_width=True)

            with t2:
                b_cols = ['Baris Excel', col_pos, 'Pagu_Clean', 'Real_Clean', 'Selisih_Anggaran', 'Penyerapan_%', 'Skor Risiko', 'Level Risiko', 'Temuan Anggaran']
                st.dataframe(df_bud[b_cols], use_container_width=True, hide_index=True)

            with t3:
                ai_context_summary = f"Modul: Anggaran vs Realisasi. Total Pagu: Rp {df_bud['Pagu_Clean'].sum():,.0f}, Total Realisasi: Rp {df_bud['Real_Clean'].sum():,.0f}, Pos melampaui pagu: {len(over_df)}."
                top_str = df_bud[['Baris Excel', col_pos, 'Pagu_Clean', 'Real_Clean', 'Penyerapan_%', 'Temuan Anggaran']].head(15).to_string(index=False)

            with t4:
                exp_bud = df_bud.drop(columns=['Pos_Clean', 'Pagu_Clean', 'Real_Clean', 'Selisih_Anggaran', 'Penyerapan_%', 'Skor Risiko'], errors='ignore')
                exp_bud['Keputusan Auditor'] = ""
                exp_bud['Catatan Auditor'] = ""
                st.download_button("📥 Unduh Hasil Anggaran (.xlsx)", data=to_excel_bytes(exp_bud, "Anggaran_Realisasi"), file_name=f"AuditInsight_Anggaran_{uploaded_file.name}")

        # --------------------------------------
        # TAB ANALISIS & PILIHAN TANYA JAWAB
        # --------------------------------------
        with t3:
            st.subheader("Asisten Penjelasan Temuan Audit")
            preset_questions = [
                "Ringkasan Temuan Utama & Penyebab Non-Fraud",
                "Prosedur Verifikasi Dokumen & Bukti Pendukung",
                "Prioritas Tindak Lanjut Auditor",
                "Evaluasi Kemungkinan Kesalahan Pencatatan / Klasifikasi"
            ]

            selected_q = st.selectbox(
                "Pilih pertanyaan audit yang ingin dibahas:",
                options=preset_questions,
                index=0
            )

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                ask_gemini_btn = st.button("✨ Jawab dengan Gemini AI", type="primary", use_container_width=True)
            with col_btn2:
                ask_instant_btn = st.button("📋 Tampilkan Panduan Instan (Tanpa API)", use_container_width=True)

            system_prompt = f"""Anda adalah asisten auditor profesional untuk aplikasi AuditInsight.
Prinsip wajib:
1. Gunakan istilah 'indikasi risiko', 'anomali', atau 'red flag'. Jangan menyimpulkan kecurangan/fraud secara pasti.
2. Jelaskan temuan berdasarkan data yang dihitung kode, sertakan kemungkinan penyebab non-fraud (kesalahan pencatatan, klasifikasi akun, atau kondisi operasional), dan beri prosedur audit lanjutan.
3. Jawab dalam Bahasa Indonesia yang jelas dan terstruktur.

{ai_context_summary}

Cuplikan 15 Baris Prioritas:
{top_str}
"""

            if ask_instant_btn:
                st.session_state["preset_answer"] = get_preset_audit_answer(
                    selected_q, mode, ai_context_summary, top_str
                )

            if ask_gemini_btn:
                if not gemini_api_key:
                    st.warning("Gemini API Key belum diisi. Menampilkan panduan terstruktur dari data yang dianalisis:")
                    st.session_state["preset_answer"] = get_preset_audit_answer(
                        selected_q, mode, ai_context_summary, top_str
                    )
                else:
                    with st.spinner("Menyiapkan jawaban untuk pertanyaan terpilih..."):
                        try:
                            resp = get_gemini_response(
                                gemini_api_key,
                                model_choice,
                                system_prompt,
                                [{"role": "user", "content": f"Jawab pertanyaan audit berikut secara ringkas, jelas, dan terstruktur: {selected_q}"}]
                            )
                            st.session_state["preset_answer"] = resp
                        except Exception as e:
                            st.warning(f"Gemini sedang sibuk ({e}). Menampilkan jawaban terstruktur otomatis:")
                            st.session_state["preset_answer"] = get_preset_audit_answer(
                                selected_q, mode, ai_context_summary, top_str
                            )

            if "preset_answer" in st.session_state:
                st.divider()
                st.markdown(st.session_state["preset_answer"])

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
else:
    st.info("Silakan unggah file Excel (.xlsx) milik Anda. Setelah file terunggah, Anda bisa memilih sheet dan mencocokkan kolomnya secara langsung.")
