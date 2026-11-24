import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from difflib import SequenceMatcher
from io import BytesIO
import os
from groq import Groq

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS
# ==========================================
st.set_page_config(
    page_title="AuditInsight - AI Audit Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0;}
    .sub-header {font-size: 1rem; color: #64748B; margin-bottom: 1.5rem;}
    .metric-card {background-color: #F8FAFC; padding: 15px; border-radius: 10px; border-left: 5px solid #1E3A8A;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI PEMROSESAN & DETEKSI RED FLAG
# ==========================================
def clean_currency(val):
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).replace('Rp', '').replace(' ', '').strip()
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
        return float(val_str)
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
    valid_amounts = [a for a in amounts if a >= 10]
    if len(valid_amounts) < 10:
        return None, set()
    
    first_digits = [int(str(int(a))[0]) for a in valid_amounts]
    total = len(first_digits)
    
    benford_expected = {d: np.log10(1 + 1/d) * 100 for d in range(1, 10)}
    actual_counts = {d: first_digits.count(d) for d in range(1, 10)}
    actual_pct = {d: (actual_counts[d] / total) * 100 for d in range(1, 10)}
    
    # Tandai digit yang menyimpang > 5% di atas ekspektasi Benford
    anomalous_digits = {d for d in range(1, 10) if (actual_pct[d] - benford_expected[d]) > 5.0}
    
    df_benford = pd.DataFrame({
        'Digit': list(range(1, 10)),
        'Aktual (%)': [round(actual_pct[d], 2) for d in range(1, 10)],
        'Benford (%)': [round(benford_expected[d], 2) for d in range(1, 10)]
    })
    return df_benford, anomalous_digits

def run_audit_engine(df_raw, auth_limit, round_mult, contamination):
    df = df_raw.copy()
    # Simpan nomor baris Excel asli (header baris 1, data mulai baris 2)
    df['Baris Excel'] = df.index + 2
    
    # Normalisasi Kolom Wajib
    df['Tanggal_Parsed'] = pd.to_datetime(df['Tanggal'], errors='coerce')
    df['Vendor_Clean'] = df['Vendor'].astype(str).str.strip()
    df['Faktur_Clean'] = df['No Faktur'].astype(str).str.strip()
    df['Jumlah_Clean'] = df['Jumlah'].apply(clean_currency)
    
    has_creator = 'Dibuat Oleh' in df.columns
    has_approver = 'Disetujui Oleh' in df.columns
    has_npwp = 'NPWP Vendor' in df.columns

    # Pre-kalkulasi agregat untuk aturan red flag
    dup_faktur = df[df.duplicated(subset=['Faktur_Clean'], keep=False)]['Faktur_Clean'].unique()
    dup_exact = df[df.duplicated(subset=['Vendor_Clean', 'Jumlah_Clean', 'Tanggal_Parsed'], keep=False)].index
    
    # Duplikat vendor & jumlah dalam 7 hari
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

    # Split transaction: Beberapa transaksi ke vendor yang sama di hari yang sama melampaui batas otorisasi
    daily_vendor_sum = df.groupby(['Vendor_Clean', 'Tanggal_Parsed'])['Jumlah_Clean'].transform('sum')
    daily_vendor_cnt = df.groupby(['Vendor_Clean', 'Tanggal_Parsed'])['Jumlah_Clean'].transform('count')
    split_tx_idx = set(df[(daily_vendor_cnt > 1) & (daily_vendor_sum > auth_limit) & (df['Jumlah_Clean'] <= auth_limit)].index)

    # Vendor sekali pakai & nama mirip
    vendor_counts = df['Vendor_Clean'].value_counts()
    one_time_vendors = set(vendor_counts[vendor_counts == 1].index)
    similar_vendor_map = find_similar_vendors(df['Vendor_Clean'].tolist())

    # Uji Benford
    df_benford, anomalous_digits = benford_analysis(df['Jumlah_Clean'].tolist())

    # Machine Learning: Isolation Forest untuk Outlier
    if len(df) >= 10:
        iso = IsolationForest(contamination=contamination, random_state=42)
        features = df[['Jumlah_Clean']].fillna(0)
        df['Outlier_Pred'] = iso.fit_predict(features)
    else:
        df['Outlier_Pred'] = 1

    # Evaluasi 11 Aturan Red Flag + ML + Benford
    scores = []
    reasons_list = []
    all_flags = []

    for idx, row in df.iterrows():
        score = 0
        reasons = []
        amt = row['Jumlah_Clean']
        tgl = row['Tanggal_Parsed']
        vnd = row['Vendor_Clean']
        fak = row['Faktur_Clean']

        # Rule 1: Pembayaran ganda (No Faktur duplikat)
        if fak in dup_faktur and fak.lower() not in ['nan', '', 'none', '-']:
            score += 35
            reasons.append("Duplikat No Faktur")
            all_flags.append("Duplikat No Faktur")

        # Rule 2: Duplikat identik (Vendor, Jumlah, Tanggal sama)
        if idx in dup_exact:
            score += 30
            reasons.append("Transaksi Identik (Vendor, Jumlah, Tanggal)")
            all_flags.append("Transaksi Identik")

        # Rule 3: Duplikat nominal & vendor dalam 7 hari
        if idx in dup_7days_idx and idx not in dup_exact:
            score += 20
            reasons.append("Nominal & Vendor Sama dalam <=7 Hari")
            all_flags.append("Duplikat <=7 Hari")

        # Rule 4: Pemecahan transaksi (Split Transaction)
        if idx in split_tx_idx:
            score += 25
            reasons.append("Indikasi Split Transaction (Hindari Batas Otorisasi)")
            all_flags.append("Split Transaction")

        # Rule 5: Melebihi batas otorisasi tunggal
        if amt > auth_limit:
            score += 20
            reasons.append(f"Melebihi Batas Otorisasi (> Rp {auth_limit:,.0f})")
            all_flags.append("Melebihi Batas Otorisasi")

        # Rule 6: Mendekati batas otorisasi (90% - 100% dari batas)
        elif 0.90 * auth_limit <= amt <= auth_limit:
            score += 15
            reasons.append("Nominal Tepat di Bawah Batas Otorisasi (90-100%)")
            all_flags.append("Mendekati Batas Otorisasi")

        # Rule 7: Nominal angka bulat (Round Amount)
        if amt > 0 and round_mult > 0 and (amt % round_mult == 0):
            score += 10
            reasons.append(f"Nominal Bulat Kelipatan Rp {round_mult:,.0f}")
            all_flags.append("Nominal Bulat")

        # Rule 8: Transaksi akhir pekan (Sabtu / Minggu)
        if pd.notna(tgl) and tgl.weekday() >= 5:
            score += 15
            reasons.append("Transaksi Akhir Pekan (Sabtu/Minggu)")
            all_flags.append("Transaksi Akhir Pekan")

        # Rule 9: Vendor sekali pakai (One-time vendor) dengan nilai signifikan
        if vnd in one_time_vendors and amt >= (0.5 * auth_limit):
            score += 10
            reasons.append("Vendor Sekali Pakai Bernilai Besar")
            all_flags.append("Vendor Sekali Pakai")

        # Rule 10: Nama vendor sangat mirip (Potensi vendor fiktif / duplikat)
        if vnd in similar_vendor_map:
            score += 15
            reasons.append(f"Nama Vendor Mirip dengan '{similar_vendor_map[vnd]}'")
            all_flags.append("Nama Vendor Mirip")

        # Rule 11: Self-approval (Opsional)
        if has_creator and has_approver:
            c = str(row['Dibuat Oleh']).strip().lower()
            a = str(row['Disetujui Oleh']).strip().lower()
            if c == a and c not in ['nan', '', 'none', '-']:
                score += 30
                reasons.append("Self-Approval (Pembuat & Penyetuju Sama)")
                all_flags.append("Self-Approval")

        # Rule 12: Vendor tanpa identitas pajak / NPWP (Opsional)
        if has_npwp:
            npwp_val = str(row['NPWP Vendor']).strip().lower()
            if npwp_val in ['nan', '', 'none', '-', '0']:
                score += 10
                reasons.append("Vendor Tanpa NPWP")
                all_flags.append("Tanpa NPWP")

        # Machine Learning: Isolation Forest Outlier
        if row['Outlier_Pred'] == -1:
            score += 15
            reasons.append("Anomali Statistik (Isolation Forest Outlier)")
            all_flags.append("ML Outlier")

        # Benford's Law Anomaly
        if amt >= 10:
            first_digit = int(str(int(amt))[0])
            if first_digit in anomalous_digits:
                score += 5
                reasons.append(f"Digit Awal ({first_digit}) Menyimpang dari Benford's Law")
                all_flags.append("Penyimpangan Benford")

        final_score = min(score, 100)
        scores.append(final_score)
        reasons_list.append("; ".join(reasons) if reasons else "Normal (Tidak ada red flag)")

    df['Skor Risiko'] = scores
    df['Level Risiko'] = df['Skor Risiko'].apply(
        lambda s: 'Tinggi' if s >= 60 else ('Sedang' if s >= 30 else 'Rendah')
    )
    df['Alasan Red Flag'] = reasons_list

    # Urutkan dari skor tertinggi
    df_result = df.sort_values(by=['Skor Risiko', 'Jumlah_Clean'], ascending=[False, False]).reset_index(drop=True)
    return df_result, df_benford, all_flags

def generate_excel_export(df_result):
    output = BytesIO()
    cols_to_drop = ['Tanggal_Parsed', 'Vendor_Clean', 'Faktur_Clean', 'Jumlah_Clean', 'Outlier_Pred']
    export_df = df_result.drop(columns=[c for c in cols_to_drop if c in df_result.columns]).copy()
    
    # Pindahkan Baris Excel, Skor Risiko, Level Risiko, Alasan Red Flag ke depan
    priority_cols = ['Baris Excel', 'Skor Risiko', 'Level Risiko', 'Alasan Red Flag']
    other_cols = [c for c in export_df.columns if c not in priority_cols]
    export_df = export_df[priority_cols + other_cols]
    
    # Tambahkan kolom kosong untuk review auditor
    export_df['Keputusan Auditor'] = ""
    export_df['Catatan Auditor'] = ""

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df.to_excel(writer, index=False, sheet_name='Hasil_Audit_Insight')
    return output.getvalue()

def get_groq_response(api_key, model_name, messages):
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.3,
        max_tokens=1500
    )
    return completion.choices[0].message.content

# ==========================================
# 3. SIDEBAR: PENGATURAN & KONFIGURASI AI
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/financial-analytics.png", width=60)
    st.title("Pengaturan Audit")
    
    st.subheader("1. Parameter Aturan")
    auth_limit = st.number_input(
        "Batas Otorisasi (Rp)", 
        min_value=100000, 
        value=50000000, 
        step=5000000,
        format="%d"
    )
    round_mult = st.number_input(
        "Kelipatan Nominal Bulat (Rp)", 
        min_value=10000, 
        value=1000000, 
        step=100000,
        format="%d"
    )
    contamination = st.slider(
        "Sensitivitas Outlier (ML)", 
        min_value=0.01, 
        max_value=0.20, 
        value=0.05, 
        step=0.01,
        help="Proporsi transaksi yang dianggap anomali ekstrem oleh Isolation Forest."
    )

    st.divider()
    st.subheader("2. Konfigurasi AI (Groq)")
    default_api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    groq_api_key = st.text_input("Groq API Key", value=default_api_key, type="password")
    
    model_choice = st.selectbox(
        "Pilih Model LLM",
        options=[
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
            "llama3-8b-8192",
            "mixtral-8x7b-32768"
        ],
        index=0
    )

    st.divider()
    st.caption("**Prinsip AuditInsight:**\nKode yang menghitung, AI yang menjelaskan, auditor yang memutuskan. Skor merupakan indikasi red flag, bukan bukti fraud.")

# ==========================================
# 4. HALAMAN UTAMA & UNGGAH DATA
# ==========================================
st.markdown('<p class="main-header">🔍 AuditInsight</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Asisten Audit Berbasis AI untuk Pemeriksaan Populasi Transaksi Secara Menyeluruh</p>', unsafe_allow_html=True)

# Template Excel contoh untuk memudahkan pengguna
with st.expander("ℹ️ Format File Excel & Unduh Template Contoh"):
    st.write("Pastikan baris pertama file Excel Anda memiliki kolom wajib: **Tanggal, Vendor, No Faktur, Jumlah**. Kolom opsional: **Dibuat Oleh, Disetujui Oleh, NPWP Vendor**.")
    sample_data = pd.DataFrame({
        'Tanggal': ['2026-03-12', '2026-03-14', '2026-03-15', '2026-03-16', '2026-03-16', '2026-03-18'],
        'Vendor': ['PT Sinar Abadi', 'CV Maju Jaya', 'PT Sinar Abadi', 'PT Mega Karya', 'PT Mega Karya', 'PT Sinar Abadii'],
        'No Faktur': ['INV-1001', 'INV-2001', 'INV-1001', 'INV-301', 'INV-302', 'INV-1009'],
        'Jumlah': [15250000, 65000000, 15250000, 30000000, 30000000, 10000000],
        'Dibuat Oleh': ['Andi', 'Budi', 'Andi', 'Citra', 'Citra', 'Deni'],
        'Disetujui Oleh': ['Eko', 'Budi', 'Eko', 'Eko', 'Eko', 'Eko'],
        'NPWP Vendor': ['01.234.567.8-901.000', '', '01.234.567.8-901.000', '02.345.678.9-012.000', '02.345.678.9-012.000', '']
    })
    buf_sample = BytesIO()
    with pd.ExcelWriter(buf_sample, engine='openpyxl') as writer:
        sample_data.to_excel(writer, index=False)
    st.download_button(
        label="📥 Unduh Template Excel Contoh",
        data=buf_sample.getvalue(),
        file_name="template_transaksi_auditinsight.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

uploaded_file = st.file_uploader("Unggah File Transaksi Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        
        required_cols = ['Tanggal', 'Vendor', 'No Faktur', 'Jumlah']
        missing_cols = [c for c in required_cols if c not in df_raw.columns]
        
        if missing_cols:
            st.error(f"❌ Kolom wajib berikut tidak ditemukan di file Excel Anda: **{', '.join(missing_cols)}**")
            st.stop()
            
        # Jalankan Mesin Audit
        df_res, df_benford, all_flags = run_audit_engine(df_raw, auth_limit, round_mult, contamination)
        
        # ==========================================
        # 5. KARTU RINGKASAN (KPI)
        # ==========================================
        total_tx = len(df_res)
        high_risk_df = df_res[df_res['Level Risiko'] == 'Tinggi']
        med_risk_df = df_res[df_res['Level Risiko'] == 'Sedang']
        high_risk_val = high_risk_df['Jumlah_Clean'].sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transaksi Diperiksa", f"{total_tx:,}")
        col2.metric("Risiko Tinggi (Skor ≥ 60)", f"{len(high_risk_df):,}", delta="Perlu Prioritas", delta_color="inverse")
        col3.metric("Risiko Sedang (Skor 30-59)", f"{len(med_risk_df):,}")
        col4.metric("Nilai Risiko Tinggi", f"Rp {high_risk_val:,.0f}")

        st.divider()

        # ==========================================
        # 6. TABS DASHBOARD, TABEL, & AI ASSISTANT
        # ==========================================
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Dashboard Visual", 
            "🚨 Tabel Transaksi & Vendor", 
            "🤖 Analisis & Chat AI", 
            "📥 Unduh Kertas Kerja"
        ])

        # --- TAB 1: DASHBOARD VISUAL ---
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                fig_hist = px.histogram(
                    df_res, x="Skor Risiko", color="Level Risiko",
                    color_discrete_map={'Tinggi': '#DC2626', 'Sedang': '#F59E0B', 'Rendah': '#10B981'},
                    title="Sebaran Skor Risiko Transaksi",
                    nbins=20
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with c2:
                if all_flags:
                    flag_counts = pd.Series(all_flags).value_counts().reset_index()
                    flag_counts.columns = ['Jenis Red Flag', 'Jumlah']
                    fig_flags = px.bar(
                        flag_counts, x='Jumlah', y='Jenis Red Flag', orientation='h',
                        title="Frekuensi Kemunculan Red Flag",
                        color='Jumlah', color_continuous_scale='Reds'
                    )
                    fig_flags.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_flags, use_container_width=True)
                else:
                    st.info("Tidak ditemukan red flag pada dataset ini.")

            c3, c4 = st.columns(2)
            with c3:
                vendor_agg = df_res.groupby('Vendor_Clean').agg(
                    Jumlah_Tx=('Jumlah_Clean', 'count'),
                    Total_Nilai=('Jumlah_Clean', 'sum'),
                    Skor_Rata2=('Skor Risiko', 'mean'),
                    Skor_Maks=('Skor Risiko', 'max')
                ).reset_index()
                fig_vendor = px.scatter(
                    vendor_agg, x='Jumlah_Tx', y='Total_Nilai',
                    size='Skor_Maks', color='Skor_Rata2',
                    hover_name='Vendor_Clean',
                    color_continuous_scale='YlOrRd',
                    title="Peta Risiko Vendor (Frekuensi vs Nilai vs Skor)"
                )
                st.plotly_chart(fig_vendor, use_container_width=True)

            with c4:
                if df_benford is not None:
                    fig_benford = go.Figure()
                    fig_benford.add_trace(go.Bar(
                        x=df_benford['Digit'], y=df_benford['Aktual (%)'],
                        name='Aktual (%)', marker_color='#2563EB'
                    ))
                    fig_benford.add_trace(go.Scatter(
                        x=df_benford['Digit'], y=df_benford['Benford (%)'],
                        name='Ekspektasi Benford (%)', mode='lines+markers',
                        line=dict(color='#DC2626', width=3)
                    ))
                    fig_benford.update_layout(
                        title="Uji Benford's Law (Distribusi Digit Pertama)",
                        xaxis_title="Digit Pertama (1-9)",
                        yaxis_title="Persentase (%)"
                    )
                    st.plotly_chart(fig_benford, use_container_width=True)
                else:
                    st.info("Data tidak cukup (minimal 10 transaksi bernilai ≥ 10) untuk menampilkan grafik Benford's Law.")

        # --- TAB 2: TABEL TRANSAKSI & VENDOR ---
        with tab2:
            st.subheader("Daftar Transaksi Berdasarkan Tingkat Risiko")
            filter_level = st.multiselect(
                "Filter Level Risiko:",
                options=['Tinggi', 'Sedang', 'Rendah'],
                default=['Tinggi', 'Sedang']
            )
            
            display_cols = ['Baris Excel', 'Tanggal', 'Vendor', 'No Faktur', 'Jumlah', 'Skor Risiko', 'Level Risiko', 'Alasan Red Flag']
            filtered_df = df_res[df_res['Level Risiko'].isin(filter_level)][display_cols]
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Ringkasan Risiko per Vendor")
            vendor_table = df_res.groupby('Vendor_Clean').agg(
                Jumlah_Transaksi=('Jumlah_Clean', 'count'),
                Total_Nilai=('Jumlah_Clean', 'sum'),
                Skor_Rata_Rata=('Skor Risiko', lambda x: round(x.mean(), 1)),
                Skor_Tertinggi=('Skor Risiko', 'max')
            ).sort_values(by=['Skor_Tertinggi', 'Total_Nilai'], ascending=False).reset_index()
            vendor_table.rename(columns={'Vendor_Clean': 'Vendor'}, inplace=True)
            st.dataframe(vendor_table, use_container_width=True, hide_index=True)

        # --- TAB 3: ANALISIS & CHAT AI ---
        with tab3:
            st.subheader("Asisten Penjelasan Temuan Audit (AI)")
            if not groq_api_key:
                st.warning("⚠️ Masukkan **Groq API Key** di sidebar sebelah kiri untuk mengaktifkan fitur analisis dan tanya-jawab AI.")
            else:
                # Siapkan konteks ringkas & 15 transaksi risiko tertinggi (menjaga privasi & efisiensi token)
                top_15 = df_res[['Baris Excel', 'Tanggal', 'Vendor', 'No Faktur', 'Jumlah', 'Skor Risiko', 'Level Risiko', 'Alasan Red Flag']].head(15)
                top_15_str = top_15.to_string(index=False)
                
                system_prompt = f"""Anda adalah asisten auditor profesional yang membantu menganalisis hasil deteksi anomali transaksi dari sistem AuditInsight.
Prinsip wajib Anda:
1. Gunakan istilah 'indikasi risiko', 'anomali', atau 'red flag'. JANGAN PERNAH menyimpulkan bahwa 'fraud' atau kecurangan pasti telah terjadi.
2. Jelaskan secara objektif mengapa transaksi tertentu mendapat skor tinggi berdasarkan aturan yang terpicu.
3. Selalu sertakan kemungkinan penyebab non-fraud (misalnya kesalahan input administrasi, urgensi operasional, atau kebijakan khusus).
4. Berikan rekomendasi prosedur audit substantif/lanjutan yang konkret bagi auditor (misalnya vouching dokumen fisik, konfirmasi pihak ketiga, pemeriksaan PO/GRN).
5. Gunakan Bahasa Indonesia yang formal, jelas, dan terstruktur.

Ringkasan Data Audit Saat Ini:
- Total Transaksi: {total_tx}
- Risiko Tinggi: {len(high_risk_df)} transaksi (Total nilai: Rp {high_risk_val:,.0f})
- Risiko Sedang: {len(med_risk_df)} transaksi
- Batas Otorisasi Ditetapkan: Rp {auth_limit:,.0f}

15 Transaksi dengan Skor Risiko Tertinggi:
{top_15_str}
"""
                if st.button("✨ Buat Ringkasan Temuan Otomatis", type="primary"):
                    with st.spinner("AI sedang menganalisis pola temuan..."):
                        try:
                            summary_prompt = "Buatkan ringkasan eksekutif dari temuan utama di atas, jelaskan pola red flag yang paling menonjol, sebutkan kemungkinan penyebab non-fraud, dan susun langkah prosedur audit lanjutan yang disarankan."
                            ai_summary = get_groq_response(
                                groq_api_key, 
                                model_choice, 
                                [{"role": "system", "content": system_prompt}, {"role": "user", "content": summary_prompt}]
                            )
                            st.session_state['ai_summary'] = ai_summary
                        except Exception as e:
                            st.error(f"Terjadi kesalahan koneksi ke Groq API: {e}")

                if 'ai_summary' in st.session_state:
                    st.markdown(st.session_state['ai_summary'])

                st.divider()
                st.subheader("💬 Chat Tanya-Jawab Lanjutan")
                
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []

                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                user_q = st.chat_input("Tanyakan sesuatu, misal: 'Mengapa transaksi di baris Excel 3 berisiko tinggi?'")
                if user_q:
                    st.session_state.chat_history.append({"role": "user", "content": user_q})
                    with st.chat_message("user"):
                        st.markdown(user_q)

                    with st.chat_message("assistant"):
                        with st.spinner("Menyusun jawaban..."):
                            try:
                                api_messages = [{"role": "system", "content": system_prompt}] + st.session_state.chat_history
                                reply = get_groq_response(groq_api_key, model_choice, api_messages)
                                st.markdown(reply)
                                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                            except Exception as e:
                                st.error(f"Gagal memproses pertanyaan: {e}")

        # --- TAB 4: UNDUH KERTAS KERJA ---
        with tab4:
            st.subheader("Ekspor Kertas Kerja Audit (Excel)")
            st.write("File Excel yang diunduh memuat seluruh transaksi yang diurutkan dari skor risiko tertinggi, lengkap dengan nomor baris Excel asli serta kolom kosong **Keputusan Auditor** dan **Catatan Auditor** untuk dokumentasi review Anda.")
            
            excel_bytes = generate_excel_export(df_res)
            st.download_button(
                label="📥 Unduh Hasil Audit (.xlsx)",
                data=excel_bytes,
                file_name=f"AuditInsight_Hasil_{uploaded_file.name}",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file Excel: {e}")
else:
    st.info("Silakan unggah file Excel transaksi Anda di atas untuk memulai pemeriksaan otomatis.")
