import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# 3. Model
BASE_DIR = Path(__file__).resolve().parent.parent
MOD_P = BASE_DIR / "models" / "landslide_pipeline.joblib"
CSV_P = BASE_DIR / "data" / "Book1.csv"

@st.cache_resource
def load_mod(p):
    if not p.exists():
        st.error("Lỗi")
        st.stop()
    return joblib.load(p)

@st.cache_data
def load_data(p):
    if not p.exists(): return pd.DataFrame()
    return pd.read_csv(p)

art = load_mod(MOD_P)
mod = art["pipeline"]
f_cols = art.get("feature_columns", [])
thrs = art.get("risk_thresholds", {"low": 0.15, "high": 0.30})
s_lbls = art.get("soil_labels", [])
df = load_data(CSV_P)

def pred_ls(slp, elv, r_mean, soil, d_riv, r_24h):
    pt = pd.DataFrame([{
        "slope": slp, "elevation": elv, "rain_mean_year": r_mean,
        "soil_type": soil, "dist_river": d_riv, "rain_forecast_24h": r_24h
    }])
    if f_cols: pt = pt[f_cols]

    p = float(mod.predict_proba(pt)[0][1])
    l = float(thrs.get("low", 0.15))
    h = float(thrs.get("high", 0.30))

    if p > h: return p, "Rủi ro Cao"
    elif p > l: return p, "Rủi ro Trung bình"
    return p, "An toàn"
