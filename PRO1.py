import streamlit as st
import pandas as pd
import requests
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import folium
from streamlit_folium import st_folium
import rasterio
import numpy as np
import os
import leafmap.foliumap as leafmap
import warnings
import time
import tempfile

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# =========================
# 1️⃣ Cấu hình trang
# =========================
st.set_page_config(page_title="Dự báo Sạt lở & Bản đồ DEM", layout="wide")
st.title("🌋 Dự báo Sạt lở & Ngập lụt")

# =========================
# 2️⃣ Đọc dữ liệu & Train model
# =========================
csv_path = "Book1.csv"

df = pd.read_csv(csv_path)
le = LabelEncoder()
df["soil_type"] = le.fit_transform(df["soil_type"])
soil_labels = list(le.classes_)

X = df[["slope", "elevation", "rain_mean_year", "soil_type", "dist_river", "rain_forecast_24h"]]
y = df["label"]

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X, y)

API_KEY = "2d4a3206becec3a48aa294ad6c759160"

# =========================
# 3️⃣ Đọc DEM & tạo slope map
# =========================
dem_path = "Lao Cai_DEM.tif"

with rasterio.open(dem_path) as src:
    dem = src.read(1, masked=True)
    transform_affine = src.transform
    crs = src.crs
    profile = src.profile.copy()
    xres = transform_affine[0]
    yres = -transform_affine[4]
    gy, gx = np.gradient(dem, yres, xres)
    slope_rad = np.arctan(np.sqrt(gx * gx + gy * gy))
    slope_deg = np.degrees(slope_rad)

tmp_dir = tempfile.gettempdir()
slope_path = "Lao Cai_DEM.tif"
profile.update(dtype=rasterio.float32, count=1, nodata=None)
with rasterio.open(slope_path, "w", **profile) as dst:
    dst.write(slope_deg.astype(rasterio.float32), 1)

from pyproj import Transformer

def get_value_at_latlon(lat, lon):
    """Lấy độ cao và độ dốc từ DEM tại tọa độ (lat, lon WGS84)."""
    with rasterio.open(dem_path) as src1, rasterio.open(slope_path) as src2:
        # Bộ chuyển đổi: từ WGS84 (EPSG:4326) sang CRS của DEM (EPSG:32648)
        transformer = Transformer.from_crs("EPSG:4326", src1.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)

        coords = [(x, y)]
        val_elev = list(src1.sample(coords))[0][0]
        val_slope = list(src2.sample(coords))[0][0]
        return float(val_elev), float(val_slope)


# =========================
# 4️⃣ Hàm tiện ích
# =========================
def get_coordinates_from_osm(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1}
    headers = {"User-Agent": "LandslidePredictorWeb/1.0"}
    res = requests.get(url, params=params, headers=headers, timeout=10)
    data = res.json()
    if not data:
        raise ValueError("Không tìm thấy địa chỉ.")
    return float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("display_name", "")

def get_forecast(lat, lon):
    url_forecast = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    resp = requests.get(url_forecast, timeout=10)
    resp.raise_for_status()
    return resp.json()

def get_rain_last_hour(lat, lon):
    """Lấy lượng mưa 1 giờ gần nhất từ OpenWeatherMap."""
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    resp = requests.get(url, timeout=10)
    data = resp.json()
    rain_1h = data.get("rain", {}).get("1h", 0.0)
    return float(rain_1h)

def sum_rain_for_hours(lat, lon, hours, forecast_json=None):
    total = 0.0
    if forecast_json is None:
        forecast_json = get_forecast(lat, lon)
    blocks_needed = (hours + 2) // 3
    for block in forecast_json.get("list", [])[:blocks_needed]:
        total += float(block.get("rain", {}).get("3h", 0.0) or 0.0)
    return total

def compute_flood_status_from_rain(total_rain_mm, hours, drainage_rate_mm_per_hour=50.0):
    capacity = drainage_rate_mm_per_hour * hours
    effective = total_rain_mm - capacity
    if effective <= 0:
        effective = 0.0
    if effective <= 50:
        flood_status = "Không ngập"
    elif effective <= 100:
        flood_status = "Ngập thấp"
    else:
        flood_status = "Ngập cao"
    return effective, flood_status

def predict_landslide(slope, elevation, rain_mean_year, soil_type, dist_river, rain_24h):
    soil_encoded = le.transform([soil_type])[0]
    new_point = pd.DataFrame([{
        "slope": slope,
        "elevation": elevation,
        "rain_mean_year": rain_mean_year,
        "soil_type": soil_encoded,
        "dist_river": dist_river,
        "rain_forecast_24h": rain_24h
    }])
    prob = model.predict_proba(new_point)[0][1]
    label = "Nguy cơ cao" if prob > 0.3 else "Nguy cơ thấp" if prob > 0.15 else "Không sạt lở"
    return prob, label

# =========================
# 5️⃣ Tabs chính
# =========================
tab1, tab2, tab3 = st.tabs([
    "📊 Dự báo Sạt lở & Ngập lụt",
    "🗺️ Bản đồ DEM",
    "📝 Báo cáo sạt lở"
])


# =============== TAB 1 ===============
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        mode = st.radio("Chọn cách nhập tọa độ:", ["Nhập địa chỉ", "Nhập kinh độ/vĩ độ"])

        if mode == "Nhập địa chỉ":
            address = st.text_input("📍 Địa chỉ:")
            lat = lon = None
            # ⭐ Nút lấy DEM từ địa chỉ
            if st.button("Lấy độ cao & độ dốc từ DEM"):
                try:
                    # Lấy toạ độ từ OSM
                    lat_tmp, lon_tmp, _ = get_coordinates_from_osm(address)

                    # Lấy DEM
                    elev_auto, slope_auto = get_value_at_latlon(lat_tmp, lon_tmp)

                    # Lưu vào session
                    st.session_state["auto_elev"] = elev_auto
                    st.session_state["auto_slope"] = slope_auto

                    st.success(
                        f"✅ Lấy thành công! Độ cao: {elev_auto:.2f} m | Độ dốc: {slope_auto:.2f}°"
                    )
                except Exception as e:
                    st.error(f"Lỗi: {e}")

        else:
            lat = st.number_input("Vĩ độ (latitude):", format="%.6f")
            lon = st.number_input("Kinh độ (longitude):", format="%.6f")
            if st.button("Lấy độ cao & độ dốc từ DEM"):
                try:
                    elev, slope = get_value_at_latlon(lat, lon)
                    st.session_state["auto_elev"] = elev
                    st.session_state["auto_slope"] = slope
                    st.success(f"✅ Lấy thành công! Độ cao: {elev:.2f} m | Độ dốc: {slope:.2f}°")
                except Exception as e:
                    st.error(f"Lỗi: {e}")

        slope = st.number_input("Độ dốc (%)", 0.0, value=st.session_state.get("auto_slope", 0.0))
        elev = st.number_input("Độ cao (m)", 0.0, value=st.session_state.get("auto_elev", 0.0))
        dist_river = st.number_input("Khoảng cách đến sông (km)", 0.0)
        rain_mean_year = 1750
        soil_type = st.selectbox("Loại đất", soil_labels)
        hours = st.selectbox("Khung giờ dự báo mưa", ["Tức thì",1, 3, 6])

        if st.button("🔍 Dự đoán"):
            try:
                if mode == "Nhập địa chỉ":
                    lat, lon, full_addr = get_coordinates_from_osm(address)

                    #  Tự động lấy độ cao và độ dốc từ DEM khi dùng địa chỉ
                    try:
                        elev_auto, slope_auto = get_value_at_latlon(lat, lon)
                        st.session_state["auto_elev"] = elev_auto
                        st.session_state["auto_slope"] = slope_auto
                    except Exception as e:
                        st.error(f"Lỗi khi lấy DEM từ địa chỉ: {e}")

                elif lat and lon:
                    full_addr = f"Tọa độ ({lat:.5f}, {lon:.5f})"
                else:
                    raise ValueError("Chưa nhập đủ tọa độ.")

                forecast_json = get_forecast(lat, lon)
                if hours == "Tức thì":
                    rain_amount = get_rain_last_hour(lat, lon)
                    total_rain = rain_amount
                    effective, flood_status = compute_flood_status_from_rain(rain_amount, 1)
                else:
                    h = hours
                    total_rain = sum_rain_for_hours(lat, lon, h, forecast_json)
                    effective, flood_status = compute_flood_status_from_rain(total_rain, h)

                mean_elev = df["elevation"].mean()
                mean_slope = df["slope"].mean()
                if elev > mean_elev + 1 or slope > 10:
                    if flood_status == "Ngập cao":
                        flood_status = "Ngập thấp"
                    elif flood_status == "Ngập thấp":
                        flood_status = "Không ngập"

                rain_24h = sum_rain_for_hours(lat, lon, 24, forecast_json)
                prob, label = predict_landslide(slope, elev, rain_mean_year, soil_type, dist_river, rain_24h)

                st.session_state["result"] = {
                    "hours": hours,
                    "total_rain": total_rain,
                    "flood_status": flood_status,
                    "label": label,
                    "prob": prob,
                    "lat": lat,
                    "lon": lon,
                    "full_addr": full_addr,
                }

            except Exception as e:
                st.error(f"Lỗi: {e}")

    with col2:
        if "result" in st.session_state:
            res = st.session_state["result"]
            color = "🟢" if res["label"] == "Nguy cơ thấp" else "🟠" if res["label"] == "Nguy cơ trung bình" else "🔴"
            if res["hours"] == "Tức thì":
                rain_text = f"🌧 Mưa hiện tại: `{res['total_rain']:.1f} mm`"
            else:
                rain_text = f"🌧 Mưa {res['hours']}h tới: `{res['total_rain']:.1f} mm`"
            st.markdown(f"""
                ### 🔎 Kết quả dự đoán
                {rain_text}  
                🚨 Ngập: `{res["flood_status"]}`  
                ⛰ Sạt lở: `{res["label"]}` ({res["prob"]*100:.1f}%){color}
            """)
            m = folium.Map(location=[res["lat"], res["lon"]], zoom_start=11)
            folium.Marker([res["lat"], res["lon"]], popup=f"{res['label']}", tooltip=res["full_addr"]).add_to(m)
            st_folium(m, width=700, height=500)

with tab2:
    # --- tạo map ---
    m2 = leafmap.Map(center=[22.35, 104.02], zoom=10, draw_control=False, measure_control=True)
    m2.add_child(folium.Element("<style>.leaflet-container { cursor: crosshair !important; }</style>"))
    m2.add_basemap("OpenTopoMap")
    m2.add_raster(dem_path, colormap="terrain", layer_name="Độ cao (m)", opacity=0.5)
    m2.add_raster(slope_path, colormap="RdYlGn_r", layer_name="Độ dốc (°)", opacity=0.5)
    folium.LayerControl(collapsed=False).add_to(m2)

    # --- nếu đã có marker cũ ---
    if "clicked_info" in st.session_state:
        lat, lon, elev, slopev = st.session_state["clicked_info"]
        folium.Marker(
            [lat, lon],
            popup=f"Độ cao: {elev:.2f} m<br>Độ dốc: {slopev:.2f}°",
            tooltip="Điểm đã chọn",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m2)

    # --- map hiển thị ---
    click = st_folium(m2, width=900, height=600)

    # --- xử lý khi click mới ---
    if click and "last_clicked" in click and click["last_clicked"]:
        lat = click["last_clicked"]["lat"]
        lon = click["last_clicked"]["lng"]
        elev, slopev = get_value_at_latlon(lat, lon)
        st.session_state["clicked_info"] = (lat, lon, elev, slopev)

        # thêm marker trước khi rerun
        folium.Marker(
            [lat, lon],
            popup=f"Độ cao: {elev:.2f} m<br>Độ dốc: {slopev:.2f}°",
            tooltip="Điểm vừa chọn",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m2)
        st.rerun()

    # --- hiển thị thông tin ---
    if "clicked_info" in st.session_state:
        lat, lon, elev, slopev = st.session_state["clicked_info"]
        st.markdown(f"""
        ### 📍 Thông tin tại điểm đã chọn
        - **Vĩ độ:** `{lat:.5f}`
        - **Kinh độ:** `{lon:.5f}`
        - **Độ cao:** `{elev:.2f} m`
        - **Độ dốc:** `{slopev:.2f}°`
        """)

# =============== TAB 3 ===============
with tab3:
    st.header("📝 Báo cáo vụ sạt lở")

    st.markdown("Hãy cung cấp thông tin chi tiết nhất có thể:")

    colA, colB = st.columns(2)

    with colA:
        report_address = st.text_input("📍 Địa điểm xảy ra sạt lở")
        report_lat = st.number_input("Vĩ độ (nếu biết)", format="%.6f")
        report_lon = st.number_input("Kinh độ (nếu biết)", format="%.6f")

        soil_type_report = st.text_input(
            "Loại đất"
        )

    with colB:
        severity = st.selectbox("Mức độ thiệt hại", [
            "Nhẹ – chỉ sạt vài điểm nhỏ",
            "Trung bình – cản trở giao thông",
            "Nặng – sạt lớn, chôn lấp tài sản",
            "Rất nặng – nguy hiểm đến tính mạng"
        ])

        causes = st.multiselect("Nguyên nhân quan sát được", [
            "Mưa lớn kéo dài",
            "Đất bão hòa nước",
            "Gần khu vực sông suối",
            "Hoạt động xây dựng",
            "Không rõ"
        ])

        dist_river_report = st.number_input(
            "Khoảng cách đến sông (km)",
            min_value=0.0,
            max_value=100.0,
            step=0.1
        )

    notes = st.text_area("Ghi chú bổ sung (tùy chọn)")

    if st.button("Gửi Báo cáo"):
        st.success("Cảm ơn bạn đã cung cấp thông tin! Chúng tôi sẽ ghi nhận và xử lý.")

