import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from difflib import SequenceMatcher
from io import BytesIO
import os
import time
import html as html_lib
from google import genai
from google.genai import types

# ==========================================
# 1. KONFIGURASI HALAMAN & TEMA VISUAL
# ==========================================
st.set_page_config(
    page_title="AuditInsight | AI Audit Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

RISK_COLORS = {"Tinggi": "#E5484D", "Sedang": "#F5A524", "Rendah": "#12A594"}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

:root {
    --ink: #0B1B3A;
    --ink-2: #12285A;
    --blue: #1D4ED8;
    --teal: #0D9488;
    --cyan: #B7A66A;
    --amber: #F5A524;
    --muted: #5B6B86;
    --line: rgba(29, 78, 216, .14);
}

/* ---------- Dasar ---------- */
.stApp, .stMarkdown, label, p, h1, h2, h3, h4, button, input, textarea {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 { font-family: 'Space Grotesk', 'Inter', sans-serif; letter-spacing: -.01em; }

.stApp {
    background:
        radial-gradient(1100px 560px at 95% -8%, rgba(34, 211, 238, .16), transparent 60%),
        radial-gradient(900px 520px at -8% 12%, rgba(29, 78, 216, .10), transparent 55%),
        linear-gradient(180deg, #F5F8FD 0%, #EAF0F9 100%);
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
footer { visibility: hidden; }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #1D4ED8, #0D9488); border-radius: 10px; }
::-webkit-scrollbar-track { background: transparent; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #482A33 0%, #603B43 58%, #59613A 100%);
    border-right: 1px solid rgba(255, 255, 255, .07);
}
section[data-testid="stSidebar"] * { color: #F3EDE2; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #FFFFFF; }
section[data-testid="stSidebar"] hr { border-color: rgba(255, 255, 255, .10); }

.brand { display: flex; align-items: center; gap: .8rem; margin: .2rem 0 1.2rem; }
.brand svg { filter: drop-shadow(0 6px 14px rgba(34, 211, 238, .45)); animation: floaty 5s ease-in-out infinite; }
.brand b { display: block; font-family: 'Space Grotesk', sans-serif; font-size: 1.45rem; color: #fff; line-height: 1.1; }
.brand span { font-size: .78rem; color: #8FB4E8; }

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255, 255, 255, .04);
    border: 1px solid rgba(255, 255, 255, .09);
    border-radius: 12px;
    padding: .6rem .85rem;
    margin-bottom: .45rem;
    transition: transform .25s ease, background .25s ease, border-color .25s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    transform: translateX(5px);
    background: rgba(34, 211, 238, .10);
    border-color: rgba(34, 211, 238, .4);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, rgba(29, 78, 216, .75), rgba(13, 148, 136, .75));
    border-color: rgba(34, 211, 238, .65);
    box-shadow: 0 10px 24px -10px rgba(34, 211, 238, .55);
}
section[data-testid="stSidebar"] div[data-baseweb="input"],
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: rgba(255, 255, 255, .07) !important;
    border: 1px solid rgba(255, 255, 255, .14) !important;
    border-radius: 10px !important;
}
section[data-testid="stSidebar"] input { background: transparent !important; color: #fff !important; }
section[data-testid="stSidebar"] div[data-baseweb="input"] button { background: rgba(255, 255, 255, .08) !important; }

/* ---------- Kartu metrik ---------- */
div[data-testid="stMetric"] {
    position: relative;
    overflow: hidden;
    background: rgba(255, 255, 255, .88);
    backdrop-filter: blur(8px);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1.05rem 1.25rem 1.05rem 1.4rem;
    box-shadow: 0 14px 32px -18px rgba(11, 27, 58, .45);
    transition: transform .3s ease, box-shadow .3s ease;
}
div[data-testid="stMetric"]::before {
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px;
    background: linear-gradient(180deg, var(--blue), var(--teal));
}
div[data-testid="stMetric"]:hover { transform: translateY(-5px); box-shadow: 0 22px 40px -18px rgba(29, 78, 216, .5); }
div[data-testid="stMetricLabel"] p { color: var(--muted); font-weight: 600; font-size: .8rem; }
div[data-testid="stMetricValue"] { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-weight: 700; }

/* ---------- Tab ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: rgba(255, 255, 255, .8); padding: 6px;
    border-radius: 14px; border: 1px solid var(--line);
}
.stTabs [data-baseweb="tab"] { border-radius: 10px; height: auto; padding: .55rem 1.05rem; transition: background .25s ease; }
.stTabs [data-baseweb="tab"] p { font-weight: 600; color: #3B4B69; }
.stTabs [data-baseweb="tab"]:hover { background: rgba(29, 78, 216, .08); }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, var(--blue), var(--teal)); box-shadow: 0 8px 18px -8px rgba(29, 78, 216, .7); }
.stTabs [aria-selected="true"] p { color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- Tombol ---------- */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px; font-weight: 600; padding: .6rem 1.1rem;
    border: 1px solid rgba(29, 78, 216, .28); background: #fff; color: var(--ink);
    transition: transform .2s ease, box-shadow .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px); box-shadow: 0 12px 22px -12px rgba(29, 78, 216, .65);
    border-color: var(--blue); color: var(--blue);
}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, var(--blue), var(--teal)); color: #fff; border: none;
    background-size: 200% 200%; animation: sheen 6s ease infinite;
}
.stButton > button[kind="primary"] p, .stButton > button[data-testid="stBaseButton-primary"] p { color: #fff; }
.stDownloadButton > button { background: linear-gradient(135deg, #0B1B3A, #12285A); color: #fff; border: none; }
.stDownloadButton > button p { color: #fff; }
.stDownloadButton > button:hover { color: #fff; }

/* ---------- Komponen lain ---------- */
section[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed rgba(29, 78, 216, .4); border-radius: 18px;
    background: linear-gradient(135deg, rgba(255, 255, 255, .95), rgba(230, 240, 252, .95));
    transition: border-color .3s ease, box-shadow .3s ease;
}
section[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--teal); box-shadow: 0 16px 32px -18px rgba(13, 148, 136, .6); }

div[data-testid="stExpander"] {
    border-radius: 16px; border: 1px solid var(--line);
    background: rgba(255, 255, 255, .82); box-shadow: 0 12px 28px -20px rgba(11, 27, 58, .5);
}
div[data-testid="stDataFrame"], div[data-testid="stPlotlyChart"] {
    background: #fff; border-radius: 16px; border: 1px solid var(--line);
    box-shadow: 0 14px 30px -20px rgba(11, 27, 58, .5); padding: .35rem; overflow: hidden;
}
div[data-testid="stAlert"] { border-radius: 14px; border: 1px solid var(--line); }
hr { border: none !important; height: 1px; background: linear-gradient(90deg, transparent, rgba(29, 78, 216, .35), transparent) !important; }

/* ---------- Kartu fitur (halaman kosong) ---------- */
.feature-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.1rem; margin-top: 1.4rem; perspective: 1100px; }
.feature-card {
    background: rgba(255, 255, 255, .9); border: 1px solid var(--line); border-radius: 18px;
    padding: 1.4rem 1.4rem 1.5rem; box-shadow: 0 16px 34px -22px rgba(11, 27, 58, .55);
    transition: transform .35s ease, box-shadow .35s ease; transform-style: preserve-3d;
}
.feature-card:hover { transform: rotateX(5deg) rotateY(-6deg) translateY(-8px); box-shadow: 0 30px 50px -24px rgba(29, 78, 216, .55); }
.feature-card .ico {
    width: 50px; height: 50px; border-radius: 14px; display: grid; place-items: center; font-size: 1.5rem; margin-bottom: .9rem;
    background: linear-gradient(135deg, var(--blue), var(--teal)); box-shadow: 0 12px 22px -10px rgba(29, 78, 216, .7);
    transform: translateZ(30px);
}
.feature-card h4 { margin: 0 0 .35rem; font-family: 'Space Grotesk', sans-serif; color: var(--ink); font-size: 1.1rem; }
.feature-card p { margin: 0; color: var(--muted); font-size: .92rem; line-height: 1.55; }
.app-footer { text-align: center; color: #7A8AA6; font-size: .8rem; margin: 2.5rem 0 .5rem; }

@keyframes floaty { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-5px); } }
@keyframes sheen { 0%, 100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
div[data-testid="stMetric"], .feature-card { animation: rise .6s ease both; }

@media (max-width: 900px) { .feature-grid { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }

/* ---------- Penyempurnaan visual profesional ---------- */
:root {
    --ink: #101B35;
    --ink-2: #172B55;
    --blue: #3157D5;
    --teal: #0E9F9A;
    --cyan: #55D6E8;
    --muted: #64748B;
    --line: rgba(49, 87, 213, .13);
}
.stApp {
    background:
        radial-gradient(ellipse at 88% 0%, rgba(86, 210, 232, .18), transparent 34%),
        radial-gradient(ellipse at 4% 18%, rgba(89, 120, 232, .11), transparent 30%),
        linear-gradient(180deg, #F8FAFF 0%, #EEF3FB 55%, #F7F9FD 100%);
}
.block-container { max-width: 1440px; padding-top: 1.6rem; padding-bottom: 2.5rem; }
section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 12% 0%, rgba(57, 114, 221, .28), transparent 34%),
        linear-gradient(165deg, #09152E 0%, #102750 58%, #0B3C4B 100%);
}
.brand { padding: .65rem .55rem 1rem; }
.brand b { letter-spacing: -.045em; }
.brand span { letter-spacing: .025em; }
div[data-testid="stMetric"] {
    border: 1px solid rgba(255,255,255,.92);
    border-radius: 20px;
    background: linear-gradient(145deg, rgba(255,255,255,.97), rgba(247,250,255,.88));
    box-shadow: 0 15px 38px -25px rgba(18, 40, 82, .52), inset 0 1px 0 #fff;
}
div[data-testid="stMetric"]::before {
    width: 4px;
    background: linear-gradient(180deg, #4F73E6, #19B5AD, #62D5E5);
}
div[data-testid="stMetricValue"] { letter-spacing: -.045em; }
.stTabs [data-baseweb="tab-list"] {
    border: 1px solid rgba(49,87,213,.12);
    box-shadow: 0 8px 25px -20px rgba(16, 27, 53, .55);
}
.stTabs [data-baseweb="tab"] { padding: .7rem 1.15rem; }
.stButton > button, .stDownloadButton > button {
    min-height: 2.7rem;
    border-radius: 13px;
    box-shadow: 0 5px 15px -13px rgba(16, 27, 53, .6);
}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(115deg, #3157D5 0%, #176FBA 48%, #0E9F9A 100%);
    box-shadow: 0 10px 24px -13px rgba(49, 87, 213, .85);
}
.stTextInput input, .stSelectbox [data-baseweb="select"] > div,
.stNumberInput input, .stTextArea textarea {
    border-radius: 12px !important;
}
div[data-testid="stExpander"] {
    border: 1px solid rgba(49,87,213,.12);
    border-radius: 18px;
    background: rgba(255,255,255,.78);
}
div[data-testid="stDataFrame"], div[data-testid="stPlotlyChart"] {
    border-radius: 18px;
    border: 1px solid rgba(49,87,213,.11);
    box-shadow: 0 14px 34px -25px rgba(16,27,53,.5);
}
section[data-testid="stFileUploaderDropzone"] {
    border-radius: 20px;
    border: 1.5px dashed rgba(49,87,213,.35);
    background:
        radial-gradient(circle at 90% 10%, rgba(85,214,232,.12), transparent 35%),
        linear-gradient(135deg, rgba(255,255,255,.98), rgba(239,245,255,.9));
}
.feature-grid { gap: 1.25rem; margin-top: 1.7rem; }
.feature-card {
    position: relative;
    min-height: 190px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,.95);
    border-radius: 22px;
    background:
        radial-gradient(circle at 100% 0%, rgba(85,214,232,.13), transparent 38%),
        linear-gradient(145deg, rgba(255,255,255,.98), rgba(244,248,255,.94));
    box-shadow: 0 18px 42px -28px rgba(16,27,53,.58);
}
.feature-card::after {
    content: "";
    position: absolute;
    width: 120px; height: 120px;
    right: -54px; bottom: -62px;
    border: 1px solid rgba(49,87,213,.12);
    border-radius: 30px;
    transform: rotate(35deg);
    pointer-events: none;
}
.feature-card .ico {
    width: 54px; height: 54px;
    border-radius: 16px;
    color: white;
    background: linear-gradient(145deg, #4269E4, #0E9F9A);
    box-shadow: 0 12px 25px -13px rgba(49,87,213,.85);
}
.feature-card h4 { letter-spacing: -.025em; }
.feature-card p { line-height: 1.7; }
.app-footer {
    padding-top: 1rem;
    border-top: 1px solid rgba(49,87,213,.1);
    letter-spacing: .02em;
}
@media (max-width: 900px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .feature-grid { grid-template-columns: 1fr; }
}

/* ---------- Tema marun & zaitun: hangat, natural, dan kontras nyaman ---------- */
:root {
    --ink: #38252A; --ink-2: #56343D; --blue: #743B49; --teal: #78804A;
    --cyan: #B7A66A; --amber: #B68A4B; --muted: #74686A; --line: rgba(116,59,73,.16);
}
.stApp {
    background: radial-gradient(900px 500px at 96% 0%, rgba(120,128,74,.12), transparent 62%),
                radial-gradient(850px 500px at 0% 10%, rgba(116,59,73,.08), transparent 58%),
                linear-gradient(180deg,#FAF8F4 0%,#F2EFE8 100%);
    color: #382F2E;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#482A33 0%,#603B43 58%,#59613A 100%);
}
section[data-testid="stSidebar"] * { color: #F8F4ED; }
section[data-testid="stSidebar"] [data-testid="stRadio"] label,
section[data-testid="stSidebar"] input { color: #F8F4ED !important; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg,#743B49,#78804A); }
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(120deg,#743B49,#78804A) !important;
    border: 1px solid rgba(255,255,255,.28) !important; color: #fff !important;
    box-shadow: 0 7px 18px rgba(82,57,54,.13) !important;
    animation: none !important; transition: transform .18s ease, filter .18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); filter: brightness(1.06); }
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,.68); border: 1px solid rgba(116,59,73,.12); }
.stTabs [aria-selected="true"] { background: linear-gradient(120deg,#743B49,#78804A) !important; box-shadow: none !important; }
.stTabs [data-baseweb="tab"]:hover { background: rgba(120,128,74,.10); }
div[data-testid="stMetric"] { background: rgba(255,255,255,.84); border: 1px solid rgba(116,59,73,.12); box-shadow: 0 10px 24px rgba(64,47,42,.06); }
[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid rgba(116,59,73,.12); border-radius: 12px; }
.feature-card { background: rgba(255,255,255,.86); border-color: rgba(116,59,73,.13); }
hr { background: linear-gradient(90deg,transparent,rgba(120,128,74,.35),transparent) !important; }
.app-footer { border-color: rgba(116,59,73,.14); color: #7A6D68; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

HERO_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{background:transparent;font-family:'Inter',sans-serif;overflow:hidden}
.hero{position:relative;height:310px;border-radius:24px;overflow:hidden;
 background:radial-gradient(760px 380px at 82% 10%,rgba(183,166,106,.30),transparent 60%),
 radial-gradient(620px 360px at 0% 100%,rgba(120,128,74,.30),transparent 60%),
 linear-gradient(135deg,#38252A,#56343D 58%,#59613A);
 box-shadow:0 28px 60px -26px rgba(7,18,41,.8)}
.grid{position:absolute;inset:0;opacity:.7;
 background-image:linear-gradient(rgba(255,255,255,.05) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.05) 1px,transparent 1px);
 background-size:38px 38px;-webkit-mask-image:linear-gradient(100deg,transparent 10%,#000 60%);mask-image:linear-gradient(100deg,transparent 10%,#000 60%)}
canvas{position:absolute;inset:0;width:100%;height:100%}
.copy{position:relative;z-index:2;padding:40px 46px;max-width:60%;animation:in .9s ease both}
.badge{display:inline-flex;align-items:center;gap:8px;font-size:12px;font-weight:600;color:#E9DDBB;
 background:rgba(120,128,74,.16);border:1px solid rgba(183,166,106,.35);padding:6px 12px;border-radius:99px}
.badge i{width:8px;height:8px;border-radius:50%;background:#B7A66A;box-shadow:0 0 0 0 rgba(34,211,238,.7);animation:pulse 2s infinite}
h1{margin:16px 0 10px;font-family:'Space Grotesk',sans-serif;font-size:46px;line-height:1.05;color:#fff;letter-spacing:-.02em}
h1 span{background:linear-gradient(90deg,#D7C69B,#B6B98A);-webkit-background-clip:text;background-clip:text;color:transparent}
p{color:#B9C9E6;font-size:15px;line-height:1.6;max-width:560px}
p b{color:#fff}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}
.chips em{font-style:normal;font-size:12px;font-weight:500;color:#F3EDE2;padding:5px 11px;border-radius:8px;
 background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.14)}
#fb{display:none;position:absolute;right:14%;top:50%;margin-top:-70px;width:140px;height:140px;perspective:700px}
.cube{width:100%;height:100%;position:relative;transform-style:preserve-3d;animation:spin 12s linear infinite}
.cube div{position:absolute;width:140px;height:140px;border:1px solid rgba(125,211,252,.7);background:rgba(34,211,238,.14)}
.cube div:nth-child(1){transform:translateZ(70px)}.cube div:nth-child(2){transform:rotateY(90deg) translateZ(70px)}
.cube div:nth-child(3){transform:rotateY(180deg) translateZ(70px)}.cube div:nth-child(4){transform:rotateY(-90deg) translateZ(70px)}
.cube div:nth-child(5){transform:rotateX(90deg) translateZ(70px)}.cube div:nth-child(6){transform:rotateX(-90deg) translateZ(70px)}
@keyframes spin{to{transform:rotateX(360deg) rotateY(360deg)}}
@keyframes pulse{70%{box-shadow:0 0 0 9px rgba(34,211,238,0)}100%{box-shadow:0 0 0 0 rgba(34,211,238,0)}}
@keyframes in{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@media(max-width:760px){.copy{max-width:100%;padding:30px 24px}h1{font-size:34px}}
@media(prefers-reduced-motion:reduce){*{animation:none!important}}
</style></head><body>
<div class="hero" id="hero">
 <div class="grid"></div>
 <canvas id="c"></canvas>
 <div id="fb"><div class="cube"><div></div><div></div><div></div><div></div><div></div><div></div></div></div>
 <div class="copy">
  <span class="badge"><i></i>Audit berbasis AI</span>
  <h1>AuditInsight <span>Multi-Modul</span></h1>
  <p>Mode Aktif: <b>__MODE__</b> — Didukung Gemini AI. Unggah file Excel (.xlsx) Anda langsung tanpa perlu memakai template khusus.</p>
  <div class="chips"><em>Isolation Forest</em><em>Hukum Benford</em><em>Analisis Rasio</em><em>Gemini AI</em></div>
 </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function(){
 var cv=document.getElementById('c'),fb=document.getElementById('fb'),hero=document.getElementById('hero');
 if(typeof THREE==='undefined'){cv.style.display='none';fb.style.display='block';return;}
 var W=hero.clientWidth,H=hero.clientHeight;
 var renderer=new THREE.WebGLRenderer({canvas:cv,antialias:true,alpha:true});
 renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));renderer.setSize(W,H,false);
 var scene=new THREE.Scene();
 var cam=new THREE.PerspectiveCamera(40,W/H,0.1,100);cam.position.set(0,1,12);
 scene.add(new THREE.AmbientLight(0xffffff,.6));
 var l1=new THREE.PointLight(0x38bdf8,1.7,60);l1.position.set(6,6,9);scene.add(l1);
 var l2=new THREE.PointLight(0x2dd4bf,1.3,60);l2.position.set(-7,2,7);scene.add(l2);
 var group=new THREE.Group();scene.add(group);
 var cols=[0x1d4ed8,0x2563eb,0x0d9488,0x22d3ee,0x38bdf8],hs=[1.6,2.5,2.0,3.3,2.7],bars=[];
 for(var i=0;i<5;i++){
  var m=new THREE.Mesh(new THREE.BoxGeometry(.72,1,.72),
   new THREE.MeshStandardMaterial({color:cols[i],metalness:.55,roughness:.22,emissive:cols[i],emissiveIntensity:.18}));
  m.position.x=(i-2)*1.05;group.add(m);bars.push({m:m,h:hs[i]});
 }
 var disc=new THREE.Mesh(new THREE.CylinderGeometry(3.5,3.5,.1,64),
  new THREE.MeshStandardMaterial({color:0x0b1b3a,metalness:.8,roughness:.3,transparent:true,opacity:.85}));
 disc.position.y=-1.6;group.add(disc);
 var rim=new THREE.Mesh(new THREE.TorusGeometry(3.5,.03,8,96),new THREE.MeshBasicMaterial({color:0x22d3ee}));
 rim.rotation.x=Math.PI/2;rim.position.y=-1.55;group.add(rim);
 var mag=new THREE.Group();
 var ring=new THREE.Mesh(new THREE.TorusGeometry(1.25,.13,24,96),
  new THREE.MeshStandardMaterial({color:0xfbbf24,metalness:.9,roughness:.2,emissive:0xf59e0b,emissiveIntensity:.25}));
 var glass=new THREE.Mesh(new THREE.CircleGeometry(1.2,48),
  new THREE.MeshBasicMaterial({color:0x7dd3fc,transparent:true,opacity:.16,side:THREE.DoubleSide}));
 var handle=new THREE.Mesh(new THREE.CylinderGeometry(.14,.18,1.5,20),
  new THREE.MeshStandardMaterial({color:0xfbbf24,metalness:.9,roughness:.25}));
 handle.position.set(1.15,-1.15,0);handle.rotation.z=Math.PI/4;
 mag.add(ring,glass,handle);mag.position.set(.9,2.5,1.6);group.add(mag);
 var orbs=[];
 for(var k=0;k<3;k++){
  var o=new THREE.Mesh(new THREE.SphereGeometry(.17,20,20),new THREE.MeshBasicMaterial({color:k==1?0xfbbf24:0x5eead4}));
  group.add(o);orbs.push(o);
 }
 var N=240,pos=new Float32Array(N*3);
 for(var p=0;p<N;p++){pos[p*3]=(Math.random()-.5)*30;pos[p*3+1]=(Math.random()-.5)*9;pos[p*3+2]=(Math.random()-.5)*10-2;}
 var pg=new THREE.BufferGeometry();pg.setAttribute('position',new THREE.BufferAttribute(pos,3));
 var pts=new THREE.Points(pg,new THREE.PointsMaterial({size:.05,color:0x93c5fd,transparent:true,opacity:.7}));
 scene.add(pts);
 function place(){
  var aspect=W/H,vis=2*cam.position.z*Math.tan(THREE.Math.degToRad(cam.fov/2));
  group.scale.setScalar(W<760?.55:.82);
  group.position.set(W<760?vis*aspect*.30:vis*aspect*.27,-.4,0);
 }
 place();
 var mx=0,my=0;
 window.addEventListener('pointermove',function(e){mx=(e.clientX/W-.5);my=(e.clientY/H-.5);});
 window.addEventListener('resize',function(){W=hero.clientWidth;H=hero.clientHeight;renderer.setSize(W,H,false);cam.aspect=W/H;cam.updateProjectionMatrix();place();});
 var reduce=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches,t=0;
 function loop(){
  requestAnimationFrame(loop);if(!reduce)t+=.012;
  group.rotation.y=-.5+Math.sin(t*.6)*.35+mx*.5;group.rotation.x=.12+my*.2;
  bars.forEach(function(b,i){var s=b.h*(1+.1*Math.sin(t*1.6+i*.9));b.m.scale.y=s;b.m.position.y=s/2-1.55;});
  mag.rotation.z=Math.sin(t)*.12;mag.position.y=2.6+Math.sin(t*1.3)*.18;
  orbs.forEach(function(o,i){var a=t*1.1+i*2.09;o.position.set(Math.cos(a)*4.1,Math.sin(a*1.4)*1.2+.8,Math.sin(a)*4.1);});
  pts.rotation.y=t*.05;
  renderer.render(scene,cam);
 }
 loop();
})();
</script></body></html>"""

def render_hero(mode_name):
    components.html(HERO_HTML.replace("__MODE__", html_lib.escape(mode_name)), height=325)

def style_fig(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#514446"),
        title=dict(font=dict(size=16, color="#56343D")),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(bgcolor="rgba(255,255,255,0)"),
        hoverlabel=dict(bgcolor="#56343D", font_color="#FFFFFF"),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.25)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.25)", zeroline=False)
    return fig

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

@st.cache_data(ttl=900, show_spinner=False)
def get_available_gemini_models(api_key):
    """Ambil model teks yang benar-benar tersedia untuk API key; cache 15 menit."""
    if not api_key or not api_key.strip():
        return []
    client = genai.Client(api_key=api_key.strip())
    available = []
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None) or []
        name = (getattr(model, "name", "") or "").replace("models/", "").strip()
        low = name.lower()
        # Hanya model Gemini untuk teks; hindari model image/audio/live dan model lama
        # gemini-2.5-pro yang pada akun pengguna memunculkan 404.
        excluded = any(x in low for x in ("image", "audio", "live", "tts", "transcrib", "embedding"))
        if "generateContent" in actions and low.startswith("gemini-") and not excluded and low != "gemini-2.5-pro":
            available.append(name)
    priority = [
        "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
        "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
        "gemini-2.5-flash", "gemini-2.5-flash-lite"
    ]
    ordered = [name for name in priority if name in available]
    ordered.extend(sorted(name for name in available if name not in ordered))
    return ordered


def get_gemini_response(api_key, model_name, system_instruction, messages, max_retries=2):
    """Kirim pertanyaan ke Gemini dan otomatis pindah model bila model pilihan 404."""
    if not api_key or not api_key.strip():
        raise ValueError("API key Gemini belum diisi.")

    client = genai.Client(api_key=api_key.strip())
    contents = []
    for message in messages:
        role = "user" if message.get("role") == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=str(message.get("content", "")))]))

    try:
        available = get_available_gemini_models(api_key)
    except Exception:
        available = []

    candidates = []
    # Model pilihan hanya dicoba jika memang tercantum di daftar model aktif.
    selected = (model_name or "").replace("models/", "").strip()
    if selected and (not available or selected in available) and selected != "gemini-2.5-pro":
        candidates.append(selected)
    for name in available:
        if name not in candidates:
            candidates.append(name)
    # Fallback versi Flash yang masih didukung dokumentasi; tetap lanjut bila 404.
    for name in ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash"):
        if name not in candidates:
            candidates.append(name)

    last_error = None
    for current_model in candidates:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=current_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.25,
                        max_output_tokens=4096
                    )
                )
                answer = getattr(response, "text", None)
                if answer and answer.strip():
                    cleaned = answer.strip()
                    # Cegah jawaban satu-dua kalimat yang jelas terpotong agar tidak
                    # dianggap sebagai jawaban sukses; pengguna akan menerima panduan lengkap.
                    if len(cleaned) < 180 or cleaned.rstrip().endswith((":", "...", "…")):
                        raise RuntimeError("Jawaban Gemini terindikasi belum lengkap.")
                    return cleaned
                raise RuntimeError("Gemini tidak mengembalikan teks jawaban.")
            except Exception as error:
                last_error = error
                error_text = str(error).lower()
                is_not_found = "404" in error_text or "not_found" in error_text or "not found" in error_text
                is_temporary = any(token in error_text for token in ("503", "unavailable", "429", "resource_exhausted", "jawaban gemini terindikasi"))
                if is_not_found:
                    break
                if is_temporary and attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                    continue
                break

    raise RuntimeError(
        "Gemini tidak berhasil menghasilkan jawaban lengkap. Periksa API key, kuota, dan akses model. Detail teknis: "
        + str(last_error)
    )

def get_preset_audit_answer(question_key, mode_name, ai_context_summary, top_str):
    header = f"**Analisis Terarah ({mode_name})**\n\n- **Ringkasan Data:** {ai_context_summary}\n\n"
    if question_key == "Ringkasan Temuan Utama & Penyebab Non-Fraud":
        return (
            header
            + "1. **Indikasi Risiko / Anomali Utama:** Fokuskan pemeriksaan pada baris dengan skor risiko tertinggi dan nilai/perubahan paling mencolok pada cuplikan prioritas di bawah.\n"
            + "2. **Kemungkinan Penyebab Non-Fraud:** Kesalahan input/format angka, klasifikasi akun atau pos yang kurang tepat, perbedaan waktu pencatatan (*cut-off*), atau kondisi operasional yang memang sah.\n"
            + "3. **Prosedur Audit Lanjutan:** Bandingkan baris prioritas dengan dokumen pendukung asli, konfirmasi ke unit/bagian terkait, lalu catat hasil verifikasi pada kolom *Keputusan Auditor* di kertas kerja Excel.\n"
            + "4. **Batasan Kesimpulan:** Nilai yang melewati pagu atau pola yang tidak biasa belum membuktikan penyimpangan. Auditor perlu memeriksa dasar perubahan anggaran, otorisasi, bukti realisasi, dan kesesuaian periode sebelum menetapkan temuan.\n\n"
            + "**Cuplikan Data Prioritas untuk Ditelusuri**\n\n"
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
    st.markdown("""
    <div class="brand">
      <svg width="52" height="52" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="g1" x1="0" y1="0" x2="64" y2="64"><stop stop-color="#B7A66A"/><stop offset="1" stop-color="#78804A"/></linearGradient>
          <linearGradient id="g2" x1="0" y1="0" x2="0" y2="64"><stop stop-color="#743B49"/><stop offset="1" stop-color="#78804A"/></linearGradient>
        </defs>
        <rect x="6" y="34" width="9" height="20" rx="2.5" fill="url(#g2)"/>
        <rect x="19" y="24" width="9" height="30" rx="2.5" fill="url(#g2)"/>
        <rect x="32" y="29" width="9" height="25" rx="2.5" fill="url(#g2)"/>
        <circle cx="42" cy="22" r="13" stroke="url(#g1)" stroke-width="5"/>
        <path d="M51.5 31.5L59 39" stroke="#FBBF24" stroke-width="5.5" stroke-linecap="round"/>
      </svg>
      <div><b>AuditInsight</b><span>Platform audit multi-modul</span></div>
    </div>
    """, unsafe_allow_html=True)
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
    fallback_model_options = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
    ]
    if gemini_api_key.strip():
        try:
            model_options = get_available_gemini_models(gemini_api_key)
            if not model_options:
                model_options = fallback_model_options
                st.caption("Daftar model tidak mengembalikan model teks; memakai pilihan model standar.")
        except Exception:
            model_options = fallback_model_options
            st.caption("Model tidak dapat dimuat otomatis. Pilihan standar ditampilkan.")
    else:
        model_options = fallback_model_options

    preferred_model = "gemini-3.8-flash"
    model_index = model_options.index(preferred_model) if preferred_model in model_options else 0
    model_choice = st.selectbox(
        "Model Gemini",
        options=model_options,
        index=model_index,
        help="Daftar model diambil dari model yang mendukung generateContent pada API key Anda."
    )

# ==========================================
# 7. HALAMAN UTAMA & UNGGAH FILE
# ==========================================
render_hero(mode)

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
                    st.plotly_chart(style_fig(px.histogram(df_res, x="Skor Risiko", color="Level Risiko", nbins=20, title="Sebaran Skor Risiko", color_discrete_map=RISK_COLORS)), use_container_width=True)
                with g2:
                    if all_flags:
                        fc = pd.Series(all_flags).value_counts().reset_index()
                        fc.columns = ['Red Flag', 'Jumlah']
                        st.plotly_chart(style_fig(px.bar(fc, x='Jumlah', y='Red Flag', orientation='h', title="Frekuensi Red Flag", color_discrete_sequence=["#743B49"])), use_container_width=True)
                if df_benford is not None:
                    fb = go.Figure()
                    fb.add_trace(go.Bar(x=df_benford['Digit'], y=df_benford['Aktual (%)'], name='Aktual (%)', marker_color="#743B49"))
                    fb.add_trace(go.Scatter(x=df_benford['Digit'], y=df_benford['Benford (%)'], name='Benford (%)', mode='lines+markers', line=dict(color="#F5A524", width=3)))
                    fb.update_layout(title="Uji Benford's Law")
                    st.plotly_chart(style_fig(fb), use_container_width=True)

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
                st.plotly_chart(style_fig(px.bar(df_fs.head(15), x=col_akun, y=['Nilai_Kini', 'Nilai_Lalu'], barmode='group', title="Top 15 Akun: Periode Berjalan vs Periode Lalu", color_discrete_sequence=["#743B49", "#78804A"])), use_container_width=True)
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
                st.plotly_chart(style_fig(px.bar(df_bud.head(15), x=col_pos, y='Penyerapan_%', color='Level Risiko', title="Persentase Penyerapan Anggaran (Top 15)", color_discrete_map=RISK_COLORS)), use_container_width=True)

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

            answer_context_key = f"{mode}|{uploaded_file.name}|{selected_q}"
            if st.session_state.get("preset_answer_context") != answer_context_key:
                st.session_state.pop("preset_answer", None)
                st.session_state["preset_answer_context"] = answer_context_key

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
                                [{"role": "user", "content": f"Jawab pertanyaan audit berikut secara lengkap (minimal 4 bagian dengan uraian yang jelas), spesifik berdasarkan angka dan cuplikan data, jangan berhenti di tengah kalimat: {selected_q}"}]
                            )
                            st.session_state["preset_answer"] = resp
                            st.session_state["preset_answer_context"] = answer_context_key
                        except Exception as e:
                            error_text = str(e)
                            if "404" in error_text or "NOT_FOUND" in error_text.upper():
                                st.error(
                                    "Model Gemini yang dipilih tidak ditemukan atau belum dapat diakses oleh API key ini. "
                                    "Aplikasi sudah mencoba model lain yang tersedia. Periksa akses model dan API key Gemini."
                                )
                            elif "API key" in error_text.lower() or "permission" in error_text.lower() or "unauthorized" in error_text.lower():
                                st.error(
                                    "Gemini menolak autentikasi. Pastikan API key berasal dari Google AI Studio, "
                                    "masih aktif, dan memiliki izin untuk Gemini API."
                                )
                            else:
                                st.warning("Gemini belum dapat memberikan jawaban lengkap. Panduan analisis berbasis data ditampilkan sebagai cadangan agar hasil tidak terpotong.")
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
    st.markdown("""
    <div class="feature-grid">
      <div class="feature-card"><div class="ico">💳</div><h4>Transaksi & Pembayaran</h4>
        <p>Deteksi duplikasi faktur, split transaction, self-approval, dan anomali nominal dengan Isolation Forest serta Hukum Benford.</p></div>
      <div class="feature-card"><div class="ico">📈</div><h4>Neraca & Laba Rugi</h4>
        <p>Tinjau lonjakan akun, konsentrasi nilai, saldo tidak lazim, dan keseimbangan Aset terhadap Liabilitas + Ekuitas.</p></div>
      <div class="feature-card"><div class="ico">🧾</div><h4>Anggaran vs Realisasi</h4>
        <p>Temukan pos yang melampaui pagu, realisasi tanpa anggaran, serta penyerapan yang terlalu rendah atau terlalu pas.</p></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="app-footer">AuditInsight • Hasil analisis berupa indikasi risiko dan perlu diverifikasi auditor</div>', unsafe_allow_html=True)
