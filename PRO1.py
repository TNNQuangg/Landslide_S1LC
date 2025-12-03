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
from pathlib import Path
from pyproj import Transformer
import gdown

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
if not os.path.exists(csv_path):
    st.error("⚠️ Không tìm thấy file Book1.csv trong thư mục.")
    st.stop()

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
# 3️⃣ DEM + SLOPE (Không tính slope runtime)
# =========================

# Bạn phải upload slope TIFF lên Google Drive và điền ID vào đây
DEM_FILES = {
    "Lao Cai_DEM.tif": "1Cl_3pDOUN4xJXr2-OroZPs6mbJF--oBm",
    "Lao Cai_DEM_SLOPE.tif": "1IjctcAWGzeINTqkh1nOCVF4aWkXoAyLF",

    "Yen Bai_DEM.tif": "1OSquH03dGdfrMVvoCmt4eMVFlKL6mZZO",
    "Yen Bai_DEM_SLOPE.tif": "1ITsZmNHz-TjVcOvH2QPD6Wp13kUDEsov",

    "Ha Giang_DEM.tif": "1Fh7X7DJNpZ2qvOgcrDm-Vf_YomCprgqK",
    "Ha Giang_DEM_SLOPE.tif": "16AGmHaPIhYui0hqurG2bOSHWdSC2m2vG",

    "Tuyen Quang_DEM.tif": "1g2TTXaV4Ce3-ztXxPxQr327Rqz-S-XwC",
    "Tuyen Quang_DEM_SLOPE.tif": "1E8G9DHq8nf02MjySzXwZ8GHn8UeHEYna"
}

def download_dem_files():
    for filename, file_id in DEM_FILES.items():
        if not os.path.exists(filename):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, filename, quiet=False, use_cookies=False)

@st.cache_resource
def ensure_dem_files():
    download_dem_files()
    return True

ensure_dem_files()

# =========================
# TẠO DANH SÁCH DEM + SLOPE MỘT CÁCH NHẸ
# =========================

dem_infos = []

# Gom DEM + SLOPE theo tên tỉnh
provinces = ["Lao Cai", "Yen Bai", "Ha Giang", "Tuyen Quang"]

for p in provinces:
    dem_path = f"{p}_DEM.tif"
    slope_path = f"{p}_DEM_SLOPE.tif"

    if os.path.exists(dem_path) and os.path.exists(slope_path):
        # Mở raster CHỈ ĐỂ LẤY CRS + bounds (rất nhanh)
        with rasterio.open(dem_path) as src:
            dem_infos.append({
                "province": p,
                "dem_path": dem_path,
                "slope_path": slope_path,
                "crs": src.crs,
                "bounds": src.bounds
            })


if not dem_infos:
    st.error("⚠️ Không tạo được raster độ dốc cho bất kỳ DEM nào.")
    st.stop()

def get_value_at_latlon(lat, lon):
    """Lấy độ cao & độ dốc từ DEM phù hợp (chỉ mở raster khi cần)."""

    for info in dem_infos:
        dem_path = info["dem_path"]
        slope_path = info["slope_path"]
        crs = info["crs"]
        bounds = info["bounds"]

        # Convert WGS84 → CRS DEM
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x, y = transformer.transform(lon, lat)

        # Kiểm tra điểm có nằm trong khu vực DEM
        if not (bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top):
            continue

        # CHỈ mở file khi thực sự cần → tối ưu hoá hoàn toàn
        with rasterio.open(dem_path) as dem_src:
            elev = list(dem_src.sample([(x, y)]))[0][0]

        with rasterio.open(slope_path) as slope_src:
            slope = list(slope_src.sample([(x, y)]))[0][0]

        return float(elev), float(slope)

    raise ValueError("Không tìm thấy DEM nào bao phủ vị trí này.")

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
    new_point = pd.DataFrame(
        [
            {
                "slope": slope,
                "elevation": elevation,
                "rain_mean_year": rain_mean_year,
                "soil_type": soil_encoded,
                "dist_river": dist_river,
                "rain_forecast_24h": rain_24h,
            }
        ]
    )
    prob = model.predict_proba(new_point)[0][1]
    label = "Nguy cơ cao" if prob > 0.3 else "Nguy cơ thấp" if prob > 0.15 else "Không sạt lở"
    return prob, label


# =========================
# 5️⃣ Tabs chính
# =========================
tab1, tab2, tab3 = st.tabs(["📊 Dự báo Sạt lở & Ngập lụt", "🗺️ Bản đồ DEM", "📝 Báo cáo sạt lở"])

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

                    # Lấy DEM (tự động chọn DEM phù hợp)
                    elev_auto, slope_auto = get_value_at_latlon(lat_tmp, lon_tmp)

                    # Lưu vào session
                    st.session_state["auto_elev"] = elev_auto
                    st.session_state["auto_slope"] = slope_auto

                    st.success(f"✅ Lấy thành công! Độ cao: {elev_auto:.2f} m | Độ dốc: {slope_auto:.2f}°")
                except Exception as e:
                    st.error(f"Lỗi: {e}")

        else:
            lat = st.number_input("Vĩ độ (latitude):", format="%.6f")
            lon = st.number_input("Kinh độ (longitude):", format="%.6f")
            if st.button("Lấy độ cao & độ dốc từ DEM"):
                try:
                    elev, slope_val = get_value_at_latlon(lat, lon)
                    st.session_state["auto_elev"] = elev
                    st.session_state["auto_slope"] = slope_val
                    st.success(f"✅ Lấy thành công! Độ cao: {elev:.2f} m | Độ dốc: {slope_val:.2f}°")
                except Exception as e:
                    st.error(f"Lỗi: {e}")

        slope = st.number_input("Độ dốc (°)", 0.0, value=st.session_state.get("auto_slope", 0.0))
        elev = st.number_input("Độ cao (m)", 0.0, value=st.session_state.get("auto_elev", 0.0))
        dist_river = st.number_input("Khoảng cách đến sông (m)", 0.0)
        rain_mean_year = 1750
        soil_type = st.selectbox("Loại đất", soil_labels)
        hours = st.selectbox("Khung giờ dự báo mưa", ["Tức thì", 1, 3, 6])

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
              # --- Box chú thích ---
        st.markdown("""
            <div style="
                background-color: #f8f9fa;
                padding: 10px 15px;
                border-radius: 8px;
                border: 1px solid #ddd;
                margin-bottom: 10px;
            ">
                <h4 style="margin-top:0;">Chú thích loại đất</h4>
                <ul>
                    <li><b>Type 1</b> – Đất feralit</li>
                    <li><b>Type 2</b> – Đất mùn núi cao</li>
                    <li><b>Type 3</b> – Đất phù sa</li>
                    <li><b>Type 4</b> – Đất xám bạc màu</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        if "result" in st.session_state:
            res = st.session_state["result"]
            color = "🟢" if res["label"] == "Không sạt lở" else "🟠" if res["label"] == "Nguy cơ thấp" else "🔴"
            if res["hours"] == "Tức thì":
                rain_text = f"🌧 Mưa hiện tại: `{res['total_rain']:.1f} mm`"
            else:
                rain_text = f"🌧 Mưa {res['hours']}h tới: `{res['total_rain']:.1f} mm`"
            st.markdown(
                f"""
                ### 🔎 Kết quả dự đoán
                {rain_text}  
                🚨 Ngập: `{res["flood_status"]}`  
                ⛰ Sạt lở: `{res["label"]}` ({res["prob"]*100:.1f}%){color}
            """
            )
            m = folium.Map(location=[res["lat"], res["lon"]], zoom_start=11)
            folium.Marker([res["lat"], res["lon"]], popup=f"{res['label']}", tooltip=res["full_addr"]).add_to(m)
            st_folium(m, width=700, height=500)

# =============== TAB 2 ===============
with tab2:
    first = dem_infos[0]
    b = first["bounds"]
    transformer = Transformer.from_crs(first["crs"], "EPSG:4326", always_xy=True)
    center_lon, center_lat = transformer.transform((b.left+b.right)/2, (b.top+b.bottom)/2)

    m2 = leafmap.Map(
    center=[center_lat, center_lon],
    zoom=9,
    draw_control=False,
    measure_control=False)
    
    m2.add_basemap("OpenTopoMap")

    # Thêm plugin đo METRIC
    measure_js = """
    <script>
        setTimeout(function() {
            var measureControl = new L.Control.Measure({
                primaryLengthUnit: 'meters',
                secondaryLengthUnit: 'kilometers',
                primaryAreaUnit: 'sqmeters',
                secondaryAreaUnit: 'hectares',
                activeColor: '#ABE67E'
            });
            measureControl.addTo(window.map);
        }, 500);
    </script>
    """
    m2.add_child(folium.Element(measure_js))

    # Thêm DEM + SLOPE
    for info in dem_infos:
        name = info["province"]
        m2.add_raster(info["dem_path"], layer_name=f"{name} - Elevation", opacity=0.5, colormap="terrain")
        m2.add_raster(info["slope_path"], layer_name=f"{name} - Slope", opacity=0.5, colormap="RdYlGn_r")

    # 🔥 Nếu đã click trước đó, thêm marker TRƯỚC khi render map
    if "clicked_info" in st.session_state:
        lat, lon, elev, slope = st.session_state["clicked_info"]
        folium.Marker(
            [lat, lon],
            popup=f"Độ cao: {elev:.2f} m<br>Độ dốc: {slope:.2f}°",
            tooltip="Điểm đã chọn",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m2)

    # Render map (sau khi đã add marker)
    click = st_folium(m2, height=600, width=900)

    # Xử lý click mới
    if click and "last_clicked" in click and click["last_clicked"]:
        lat = click["last_clicked"]["lat"]
        lon = click["last_clicked"]["lng"]

        try:
            elev, slope = get_value_at_latlon(lat, lon)
            st.session_state["clicked_info"] = (lat, lon, elev, slope)
            st.rerun()
        except Exception as e:
            st.warning(f"Không tìm thấy DEM: {e}")

    if "clicked_info" in st.session_state:
        lat, lon, elev, slope = st.session_state["clicked_info"]
        st.markdown(f"""
        ### 📍 Điểm đã chọn
        - **Vĩ độ:** {lat:.5f}
        - **Kinh độ:** {lon:.5f}
        - **Độ cao:** {elev:.2f} m
        - **Độ dốc:** {slope:.2f} °
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

        soil_type_report = st.text_input("Loại đất")

    with colB:
        severity = st.selectbox(
            "Mức độ thiệt hại",
            [
                "Nhẹ – chỉ sạt vài điểm nhỏ",
                "Trung bình – cản trở giao thông",
                "Nặng – sạt lớn, chôn lấp tài sản",
                "Rất nặng – nguy hiểm đến tính mạng",
            ],
        )

        causes = st.multiselect(
            "Nguyên nhân quan sát được",
            [
                "Mưa lớn kéo dài",
                "Đất bão hòa nước",
                "Gần khu vực sông suối",
                "Hoạt động xây dựng",
                "Không rõ",
            ],
        )

        dist_river_report = st.number_input(
            "Khoảng cách đến sông (m)", min_value=0.0, max_value=100.0, step=0.1
        )

    notes = st.text_area("Ghi chú bổ sung (tùy chọn)")

    if st.button("Gửi Báo cáo"):
        st.success("Cảm ơn bạn đã cung cấp thông tin! Chúng tôi sẽ ghi nhận và xử lý.")












