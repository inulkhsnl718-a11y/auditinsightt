import os
from difflib import SequenceMatcher
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sklearn.ensemble import IsolationForest

# ------------------------------------------------------------------
# Konfigurasi halaman (harus perintah Streamlit pertama)
# ------------------------------------------------------------------
st.set_page_config(page_title="AuditLens", page_icon="🔍", layout="wide")

# ------------------------------------------------------------------
# Load API key secara aman (.env atau Streamlit Secrets)
# ------------------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        GROQ_API_KEY = None

if not GROQ_API_KEY:
    st.error("🚨 API Key tidak ditemukan! Isi di Streamlit Secrets atau file .env.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ------------------------------------------------------------------
# Definisi aturan red flag: nama -> (bobot skor, penjelasan)
# ------------------------------------------------------------------
RULES = {
    "Duplikat persis": (40, "Vendor, nomor faktur, dan nominal sama persis (indikasi pembayaran ganda)"),
    "Duplikat nominal ≤7 hari": (20, "Vendor dan nominal sama dalam 7 hari dengan faktur berbeda"),
    "Nominal bulat": (10, "Nominal kelipatan bulat (indikasi estimasi, bukan transaksi riil)"),
    "Transaksi akhir pekan": (10, "Transaksi pada Sabtu/Minggu"),
    "Mendekati batas otorisasi": (15, "Nominal 90-100% dari batas persetujuan"),
    "Indikasi pemecahan transaksi": (25, "Beberapa transaksi vendor sama di hari sama, total melewati batas otorisasi"),
    "Self-approval": (30, "Pembuat dan penyetuju transaksi adalah orang yang sama"),
    "Outlier statistik": (15, "Pola nominal/waktu menyimpang (Isolation Forest)"),
    "Nama vendor mirip": (20, "Nama vendor sangat mirip dengan vendor lain (indikasi vendor fiktif/duplikat)"),
    "Vendor sekali pakai bernilai besar": (15, "Vendor hanya muncul sekali dengan nominal di 10% teratas"),
    "NPWP vendor kosong": (15, "Data NPWP vendor tidak tersedia"),
}

REQUIRED_COLUMNS = ["Tanggal", "Vendor", "No Faktur", "Jumlah"]
OPTIONAL_COLUMNS = ["Dibuat Oleh", "Disetujui Oleh", "NPWP Vendor"]


# ------------------------------------------------------------------
# Fungsi analisis (semua angka dihitung oleh kode, BUKAN oleh LLM)
# ------------------------------------------------------------------
def load_data(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    df["Baris Excel"] = df.index + 2  # jejak audit: baris asli di file
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
    df["Jumlah"] = pd.to_numeric(df["Jumlah"], errors="coerce")
    df = df.dropna(subset=["Tanggal", "Jumlah"]).reset_index(drop=True)
    df["Vendor"] = df["Vendor"].astype(str).str.strip()
    df["No Faktur"] = df["No Faktur"].astype(str).str.strip()
    return df


def run_rules(df: pd.DataFrame, approval_limit: float, round_unit: float, contamination: float) -> pd.DataFrame:
    flags = pd.DataFrame(False, index=df.index, columns=list(RULES.keys()))

    # 1. Duplikat persis
    flags["Duplikat persis"] = df.duplicated(["Vendor", "No Faktur", "Jumlah"], keep=False)

    # 2. Duplikat nominal dalam 7 hari (faktur berbeda)
    for _, g in df.groupby(["Vendor", "Jumlah"]):
        if len(g) > 1:
            g = g.sort_values("Tanggal")
            gap_prev = g["Tanggal"].diff().dt.days
            gap_next = -g["Tanggal"].diff(-1).dt.days
            hit = (gap_prev <= 7) | (gap_next <= 7)
            flags.loc[g.index[hit.values], "Duplikat nominal ≤7 hari"] = True

    # 3. Nominal bulat
    flags["Nominal bulat"] = (df["Jumlah"] >= round_unit) & (df["Jumlah"] % round_unit == 0)

    # 4. Akhir pekan
    flags["Transaksi akhir pekan"] = df["Tanggal"].dt.dayofweek >= 5

    # 5. Mendekati batas otorisasi
    flags["Mendekati batas otorisasi"] = (df["Jumlah"] >= 0.9 * approval_limit) & (df["Jumlah"] <= approval_limit)

    # 6. Pemecahan transaksi (split) per vendor per hari
    grp = df.groupby(["Vendor", df["Tanggal"].dt.date])["Jumlah"]
    total_hari = grp.transform("sum")
    jumlah_trx = grp.transform("count")
    flags["Indikasi pemecahan transaksi"] = (
        (jumlah_trx > 1) & (total_hari > approval_limit) & (df["Jumlah"] <= approval_limit)
    )

    # 7. Self-approval
    if "Dibuat Oleh" in df.columns and "Disetujui Oleh" in df.columns:
        same = df["Dibuat Oleh"].astype(str).str.strip().str.lower() == df["Disetujui Oleh"].astype(str).str.strip().str.lower()
        flags["Self-approval"] = same & df["Dibuat Oleh"].notna()

    # 8. Outlier statistik (Isolation Forest) - butuh data cukup
    if len(df) >= 30:
        fitur = pd.DataFrame(
            {
                "log_jumlah": np.log1p(df["Jumlah"].clip(lower=0)),
                "hari_dlm_bulan": df["Tanggal"].dt.day,
                "hari_dlm_minggu": df["Tanggal"].dt.dayofweek,
            }
        )
        model = IsolationForest(contamination=contamination, random_state=42)
        flags["Outlier statistik"] = model.fit_predict(fitur) == -1

    # 9. Nama vendor mirip (indikasi vendor fiktif / duplikat master data)
    vendors = df["Vendor"].unique().tolist()
    mirip = set()
    for i in range(len(vendors)):
        for j in range(i + 1, len(vendors)):
            a, b = vendors[i].lower(), vendors[j].lower()
            if SequenceMatcher(None, a, b).ratio() >= 0.88:
                mirip.update([vendors[i], vendors[j]])
    flags["Nama vendor mirip"] = df["Vendor"].isin(mirip)

    # 10. Vendor sekali pakai dengan nominal besar
    freq = df["Vendor"].map(df["Vendor"].value_counts())
    batas_besar = df["Jumlah"].quantile(0.90)
    flags["Vendor sekali pakai bernilai besar"] = (freq == 1) & (df["Jumlah"] >= batas_besar)

    # 11. NPWP kosong
    if "NPWP Vendor" in df.columns:
        flags["NPWP vendor kosong"] = df["NPWP Vendor"].isna() | (df["NPWP Vendor"].astype(str).str.strip() == "")

    # Hitung skor (0-100) dan alasan
    bobot = pd.Series({k: v[0] for k, v in RULES.items()})
    skor = (flags.astype(int) * bobot).sum(axis=1).clip(upper=100)

    hasil = df.copy()
    hasil["Skor Risiko"] = skor
    hasil["Level"] = pd.cut(skor, bins=[-1, 29, 59, 100], labels=["Rendah", "Sedang", "Tinggi"])
    hasil["Alasan"] = flags.apply(lambda r: "; ".join(r.index[r.values]), axis=1)
    return hasil


def benford_analysis(amounts: pd.Series):
    """Uji digit pertama (Benford's Law) + MAD menurut kriteria Nigrini."""
    a = amounts[amounts > 0]
    first = a.astype(str).str.replace(r"[^0-9]", "", regex=True).str.lstrip("0").str[0].astype(int)
    observed = first.value_counts(normalize=True).reindex(range(1, 10), fill_value=0)
    expected = pd.Series({d: np.log10(1 + 1 / d) for d in range(1, 10)})
    mad = (observed - expected).abs().mean()
    if mad < 0.006:
        status = "Sangat sesuai Benford"
    elif mad < 0.012:
        status = "Cukup sesuai"
    elif mad < 0.015:
        status = "Batas (marginal)"
    else:
        status = "Tidak sesuai Benford (perlu ditelusuri)"
    return observed, expected, mad, status, len(a)


def rupiah(x: float) -> str:
    return "Rp " + f"{x:,.0f}".replace(",", ".")


def to_excel(df: pd.DataFrame) -> bytes:
    out = BytesIO()
    hasil = df.copy()
    hasil["Tanggal"] = hasil["Tanggal"].dt.strftime("%Y-%m-%d")
    hasil["Keputusan Auditor"] = ""  # diisi manual oleh auditor
    hasil["Catatan Auditor"] = ""
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        hasil.sort_values("Skor Risiko", ascending=False).to_excel(writer, index=False, sheet_name="Hasil Skor")
    return out.getvalue()


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🔍 AuditLens – Skoring Risiko Transaksi & Red Flag")
st.write(
    "Unggah data transaksi (pembayaran/jurnal). Agen menghitung skor risiko dengan aturan & machine learning, "
    "lalu AI menjelaskan temuan. **Hasil adalah indikasi, bukan kesimpulan kecurangan. Keputusan ada di auditor.**"
)

with st.sidebar:
    st.header("⚙️ Parameter Agen")
    selected_model = st.selectbox(
        "🤖 Pilih Model AI",
        ["openai/gpt-oss-20b", "llama-3.3-70b-versatile", "openai/gpt-oss-120b"],
        index=0,
    )
    approval_limit = st.number_input("Batas otorisasi (Rp)", min_value=0.0, value=50_000_000.0, step=1_000_000.0)
    round_unit = st.number_input("Kelipatan nominal bulat (Rp)", min_value=1.0, value=1_000_000.0, step=100_000.0)
    contamination = st.slider("Sensitivitas outlier (Isolation Forest)", 0.01, 0.10, 0.03, 0.01)
    st.caption(
        "Kolom wajib: Tanggal, Vendor, No Faktur, Jumlah.\n\n"
        "Kolom opsional: Dibuat Oleh, Disetujui Oleh, NPWP Vendor."
    )

uploaded_file = st.file_uploader("📂 Upload data transaksi (Excel)", type=["xlsx"])

if uploaded_file:
    raw = pd.read_excel(uploaded_file, nrows=0)
    uploaded_file.seek(0)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        st.error(f"⚠️ Kolom wajib belum ada: {', '.join(missing)}")
        st.stop()

    df = load_data(uploaded_file)
    scored = run_rules(df, approval_limit, round_unit, contamination)

    n_total = len(scored)
    n_tinggi = int((scored["Level"] == "Tinggi").sum())
    n_sedang = int((scored["Level"] == "Sedang").sum())
    nilai_tinggi = scored.loc[scored["Level"] == "Tinggi", "Jumlah"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total transaksi", f"{n_total:,}")
    c2.metric("Risiko Tinggi", n_tinggi)
    c3.metric("Risiko Sedang", n_sedang)
    c4.metric("Nilai risiko tinggi", rupiah(nilai_tinggi))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Ringkasan", "🚩 Transaksi Berisiko", "🏢 Vendor", "🔢 Benford", "🤖 AI Agent"]
    )

    # --- Tab 1: ringkasan ---
    with tab1:
        colA, colB = st.columns(2)
        with colA:
            fig_lvl = px.histogram(
                scored, x="Skor Risiko", nbins=20, color="Level",
                color_discrete_map={"Rendah": "#2ca02c", "Sedang": "#ff7f0e", "Tinggi": "#d62728"},
                title="Distribusi skor risiko",
            )
            st.plotly_chart(fig_lvl, use_container_width=True)
        with colB:
            hitung = {}
            for _, row in scored["Alasan"].items():
                for r in filter(None, row.split("; ")):
                    hitung[r] = hitung.get(r, 0) + 1
            if hitung:
                df_rule = pd.DataFrame({"Red flag": hitung.keys(), "Jumlah": hitung.values()}).sort_values("Jumlah")
                fig_rule = px.bar(df_rule, x="Jumlah", y="Red flag", orientation="h", title="Frekuensi red flag")
                st.plotly_chart(fig_rule, use_container_width=True)
            else:
                st.info("Tidak ada red flag terdeteksi.")

        with st.expander("📖 Daftar aturan dan bobot skor"):
            st.dataframe(
                pd.DataFrame([{"Aturan": k, "Bobot": v[0], "Penjelasan": v[1]} for k, v in RULES.items()]),
                use_container_width=True, hide_index=True,
            )

    # --- Tab 2: transaksi berisiko ---
    with tab2:
        level_filter = st.multiselect("Filter level", ["Tinggi", "Sedang", "Rendah"], default=["Tinggi", "Sedang"])
        tampil = scored[scored["Level"].isin(level_filter)].sort_values("Skor Risiko", ascending=False)
        kolom = ["Baris Excel", "Tanggal", "Vendor", "No Faktur", "Jumlah", "Skor Risiko", "Level", "Alasan"]
        st.dataframe(tampil[kolom], use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Unduh hasil (Excel, dengan kolom review auditor)",
            data=to_excel(scored),
            file_name="hasil_auditlens.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # --- Tab 3: vendor ---
    with tab3:
        vendor_df = (
            scored.groupby("Vendor")
            .agg(Transaksi=("Jumlah", "count"), Total=("Jumlah", "sum"),
                 Skor_Rata2=("Skor Risiko", "mean"), Skor_Maks=("Skor Risiko", "max"))
            .sort_values("Skor_Maks", ascending=False)
            .reset_index()
        )
        vendor_df["Skor_Rata2"] = vendor_df["Skor_Rata2"].round(1)
        st.dataframe(vendor_df, use_container_width=True, hide_index=True)
        fig_v = px.scatter(
            vendor_df, x="Transaksi", y="Skor_Maks", size="Total", hover_name="Vendor",
            title="Peta risiko vendor (ukuran gelembung = total nilai)",
        )
        st.plotly_chart(fig_v, use_container_width=True)

    # --- Tab 4: Benford ---
    with tab4:
        observed, expected, mad, status, n_benford = benford_analysis(scored["Jumlah"])
        fig_b = go.Figure()
        fig_b.add_bar(x=list(range(1, 10)), y=observed.values, name="Aktual")
        fig_b.add_scatter(x=list(range(1, 10)), y=expected.values, name="Benford", mode="lines+markers")
        fig_b.update_layout(title="Distribusi digit pertama vs Benford's Law", xaxis_title="Digit pertama", yaxis_title="Proporsi")
        st.plotly_chart(fig_b, use_container_width=True)
        st.write(f"**MAD:** {mad:.4f} → **{status}** (n = {n_benford})")
        if n_benford < 100:
            st.warning("Data kurang dari 100 transaksi, uji Benford kurang andal.")
        st.caption("Penyimpangan Benford hanya sinyal awal, bukan bukti kecurangan.")

    # --- Tab 5: AI Agent ---
    with tab5:
        top = scored[scored["Skor Risiko"] > 0].sort_values("Skor Risiko", ascending=False).head(15).copy()
        top["Tanggal"] = top["Tanggal"].dt.strftime("%Y-%m-%d")
        top_text = top[["Baris Excel", "Tanggal", "Vendor", "No Faktur", "Jumlah", "Skor Risiko", "Alasan"]].to_string(index=False)

        # Hanya RINGKASAN & 15 transaksi teratas yang dikirim ke LLM (bukan seluruh data)
        context = (
            f"Total transaksi: {n_total}\n"
            f"Risiko Tinggi: {n_tinggi}, Sedang: {n_sedang}\n"
            f"Nilai transaksi risiko tinggi: {rupiah(nilai_tinggi)}\n"
            f"Benford MAD: {mad:.4f} ({status})\n\n"
            f"15 transaksi skor tertinggi:\n{top_text}"
        )

        SYSTEM_PROMPT = """Anda adalah AI Risk & Fraud Analyst pendukung auditor.
Aturan wajib:
1. Gunakan HANYA angka dan data pada konteks yang diberikan. Jangan mengarang angka atau transaksi.
2. Jangan menyimpulkan bahwa kecurangan telah terjadi. Gunakan istilah 'indikasi' atau 'red flag'.
3. Untuk setiap temuan utama, berikan: penjelasan risiko, kemungkinan penyebab non-fraud (false positive),
   dan prosedur audit lanjutan yang disarankan.
4. Rujuk transaksi dengan 'Baris Excel' agar bisa ditelusuri.
5. Jawab dalam Bahasa Indonesia, terstruktur, ringkas."""

        if st.button("🚀 Buat Analisis AI"):
            try:
                resp = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Hasil skoring agen:\n{context}\n\nRingkas temuan utama dan prioritas investigasi."},
                    ],
                    model=selected_model,
                    temperature=0.2,
                )
                st.session_state["ai_summary"] = resp.choices[0].message.content
            except Exception as e:
                st.error(f"⚠️ Permintaan AI gagal: {e}")

        if "ai_summary" in st.session_state:
            st.markdown("**AI Initial Analysis:**")
            st.write(st.session_state["ai_summary"])

        st.divider()

        # Chat berantai (persisten di session_state)
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_query = st.text_input("💬 Tanya Fraud Agent (mis. 'Mengapa vendor X berisiko?', 'Prosedur audit apa selanjutnya?')")
        col_chat1, col_chat2 = st.columns([4, 1])
        with col_chat1:
            send_btn = st.button("Send")
        with col_chat2:
            reset_btn = st.button("🔄 Reset Chat")

        if reset_btn:
            st.session_state.chat_history = []
            st.success("Riwayat chat dihapus!")

        if send_btn and user_query:
            try:
                chat_resp = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *st.session_state.chat_history,
                        {"role": "user", "content": f"Konteks hasil skoring:\n{context}\n\nPertanyaan: {user_query}"},
                    ],
                    model=selected_model,
                    temperature=0.2,
                )
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.session_state.chat_history.append({"role": "assistant", "content": chat_resp.choices[0].message.content})
            except Exception as e:
                st.error(f"⚠️ Chat AI gagal: {e}")

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**👤 Anda:** {msg['content']}")
            else:
                st.markdown(f"**🤖 Agent:** {msg['content']}")
