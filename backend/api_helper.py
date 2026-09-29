import requests
import streamlit as st

@st.cache_resource
def get_ses():
    ses = requests.Session()
    ses.headers.update({"User-Agent": "LandslidePredictor/2.0"})
    return ses

http_ses = get_ses()
API_KEY = "2d4a3206becec3a48aa294ad6c759160"

# 1. API
@st.cache_data(ttl=86400)
def get_coords(addr):
    addr = addr.strip()
    if not addr: raise ValueError("Err")
    url = "https://nominatim.openstreetmap.org/search"
    prm = {"q": addr, "format": "json", "limit": 1}
    r = http_ses.get(url, params=prm, timeout=5)
    r.raise_for_status()
    d = r.json()
    if not d: raise ValueError("Err")
    return float(d[0]["lat"]), float(d[0]["lon"]), d[0].get("display_name", "")

@st.cache_data(ttl=600)
def get_fc(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    r = http_ses.get(url, timeout=5)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=600)
def get_rain_1h(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    r = http_ses.get(url, timeout=5)
    r.raise_for_status()
    return float(r.json().get("rain", {}).get("1h", 0.0))

def calc_rain(lat, lon, h, fc_data=None):
    if fc_data is None: fc_data = get_fc(lat, lon)
    tot = 0.0
    blks = (h + 2) // 3
    for b in fc_data.get("list", [])[:blks]:
        tot += float(b.get("rain", {}).get("3h", 0.0) or 0.0)
    if blks > 0: tot = tot * (h / (blks * 3))
    return tot
