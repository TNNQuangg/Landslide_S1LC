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

# Khởi tạo mô hình
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X, y)

API_KEY = "2d4a3206becec3a48aa294ad6c759160"

# =========================
# 3️⃣ DEM + SLOPE (Không tính slope runtime)
# =========================

# ID Google Drive của các file DEM và SLOPE
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
    """Tải xuống các tệp DEM từ Google Drive nếu chưa tồn tại."""
    for filename, file_id in DEM_FILES.items():
        if not os.path.exists(filename):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, filename, quiet=False, use_cookies=False)

@st.cache_resource
def ensure_dem_files():
    """Đảm bảo các tệp DEM đã được tải xuống."""
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
    st.error("⚠️ Không tìm thấy tệp DEM và SLOPE phù hợp cho bất kỳ tỉnh nào.")
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
            # Đảm bảo điểm nằm trong khu vực DEM trước khi sample
            if x < dem_src.bounds.left or x > dem_src.bounds.right or \
               y < dem_src.bounds.bottom or y > dem_src.bounds.top:
                continue

            # Lấy giá trị độ cao, kiểm tra no_data
            try:
                elev = list(dem_src.sample([(x, y)]))[0][0]
                if np.isnan(elev) or elev == dem_src.nodata:
                    continue # Điểm nằm trong DEM nhưng là no_data
                elev = float(elev)
            except Exception:
                continue # Lỗi khi sample

        with rasterio.open(slope_path) as slope_src:
            # Lấy giá trị độ dốc, kiểm tra no_data
            try:
                slope = list(slope_src.sample([(x, y)]))[0][0]
                if np.isnan(slope) or slope == slope_src.nodata:
                    continue # Điểm nằm trong SLOPE nhưng là no-data
                slope = float(slope)
            except Exception:
                continue # Lỗi khi sample

        return elev, slope

    raise ValueError("Không tìm thấy DEM nào bao phủ vị trí này.")

# =========================
# 4️⃣ Hàm tiện ích
# =========================
def get_coordinates_from_osm(address):
    """Lấy tọa độ từ địa chỉ bằng OpenStreetMap Nominatim."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1}
    headers = {"User-Agent": "LandslidePredictorWeb/1.0"}
    res = requests.get(url, params=params, headers=headers, timeout=10)
    data = res.json()
    if not data:
        raise ValueError("Không tìm thấy địa chỉ.")
    return float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("display_name", "")


def get_forecast(lat, lon):
    """Lấy dữ liệu dự báo thời tiết từ OpenWeatherMap."""
    url_forecast = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    resp = requests.get(url_forecast, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_rain_last_hour(lat, lon):
    """Lấy lượng mưa 1 giờ gần nhất từ OpenWeatherMap."""
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    resp = requests.get(url, timeout=10)
    data = resp.json()
    # OpenWeatherMap trả về lượng mưa 1h/3h trong trường 'rain'
    rain_1h = data.get("rain", {}).get("1h", 0.0)
    return float(rain_1h)


def sum_rain_for_hours(lat, lon, hours, forecast_json=None):
    """Tính tổng lượng mưa dự báo trong N giờ tới (dựa trên các khối 3 giờ)."""
    total = 0.0
    if forecast_json is None:
        forecast_json = get_forecast(lat, lon)
    
    # Số khối 3 giờ cần thiết để bao phủ N giờ
    blocks_needed = (hours + 2) // 3
    
    for block in forecast_json.get("list", [])[:blocks_needed]:
        # Lấy lượng mưa 3h, nếu không có thì là 0.0
        total += float(block.get("rain", {}).get("3h", 0.0) or 0.0)
    return total


def compute_flood_status_from_rain(total_rain_mm, hours, drainage_rate_mm_per_hour=50.0):
    """Tính toán trạng thái ngập lụt dựa trên lượng mưa và khả năng thoát nước."""
    capacity = drainage_rate_mm_per_hour * hours
    effective = total_rain_mm - capacity # Lượng nước đọng lại
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
    """Dự đoán nguy cơ sạt lở bằng mô hình RandomForestClassifier đã train."""
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
    # Xác suất sạt lở (lớp 1)
    prob = model.predict_proba(new_point)[0][1]
    
    # Phân loại nguy cơ
    if prob > 0.3:
        label = "Nguy cơ cao"
    elif prob > 0.15:
        label = "Nguy cơ thấp"
    else:
        label = "Không sạt lở"
        
    return prob, label


# =========================
# 5️⃣ Tabs chính
# =========================
tab1, tab2, tab3 = st.tabs(["📊 Dự báo Sạt lở & Ngập lụt", "🗺️ Bản đồ DEM", "📝 Báo cáo sạt lở"])

# --- TAB 1: Dự báo Sạt lở & Ngập lụt ---
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("⚙️ Thông tin đầu vào")
        
        # Chọn cách nhập tọa độ
        mode = st.radio("Chọn cách nhập tọa độ:", ["Nhập địa chỉ", "Nhập kinh độ/vĩ độ"])

        # Khởi tạo giá trị mặc định cho độ cao/độ dốc tự động
        if "auto_elev" not in st.session_state:
            st.session_state["auto_elev"] = 0.0
        if "auto_slope" not in st.session_state:
            st.session_state["auto_slope"] = 0.0

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
            lat = st.number_input("Vĩ độ (latitude):", format="%.6f", value=0.0)
            lon = st.number_input("Kinh độ (longitude):", format="%.6f", value=0.0)
            
            if st.button("Lấy độ cao & độ dốc từ DEM"):
                try:
                    elev_val, slope_val = get_value_at_latlon(lat, lon)
                    st.session_state["auto_elev"] = elev_val
                    st.session_state["auto_slope"] = slope_val
                    st.success(f"✅ Lấy thành công! Độ cao: {elev_val:.2f} m | Độ dốc: {slope_val:.2f}°")
                except Exception as e:
                    st.error(f"Lỗi: {e}")

        # Thông số sạt lở (có thể dùng giá trị tự động)
        slope = st.number_input(
            "Độ dốc (°)", 
            min_value=0.0, 
            value=st.session_state.get("auto_slope", 0.0), 
            format="%.2f"
        )
        elev = st.number_input(
            "Độ cao (m)", 
            min_value=0.0, 
            value=st.session_state.get("auto_elev", 0.0),
            format="%.2f"
        )
        dist_river = st.number_input(
            "Khoảng cách đến sông (m)", 
            min_value=0.0, 
            value=100.0, 
            step=10.0,
            format="%.1f"
        )
        # Giá trị mặc định/cố định
        rain_mean_year = 1750
        soil_type = st.selectbox("Loại đất", soil_labels)
        hours = st.selectbox("Khung giờ dự báo mưa", ["Tức thì", 1, 3, 6, 12, 24])

        if st.button("🔍 Dự đoán Nguy cơ"):
            try:
                # --- 1. Xử lý tọa độ ---
                if mode == "Nhập địa chỉ":
                    lat, lon, full_addr = get_coordinates_from_osm(address)
                    
                    # Tự động lấy độ cao và độ dốc (nếu chưa lấy)
                    if elev == 0.0 or slope == 0.0:
                        try:
                            elev_auto, slope_auto = get_value_at_latlon(lat, lon)
                            elev = elev_auto
                            slope = slope_auto
                            st.session_state["auto_elev"] = elev_auto
                            st.session_state["auto_slope"] = slope_auto
                        except Exception:
                            # Không tìm thấy DEM, dùng giá trị mặc định đã nhập
                            pass 

                elif lat and lon:
                    full_addr = f"Tọa độ ({lat:.5f}, {lon:.5f})"
                else:
                    raise ValueError("Chưa nhập đủ tọa độ.")

                # --- 2. Xử lý dự báo mưa & ngập ---
                forecast_json = get_forecast(lat, lon)
                
                if hours == "Tức thì":
                    # Mưa hiện tại (1h gần nhất)
                    rain_amount = get_rain_last_hour(lat, lon)
                    total_rain = rain_amount
                    effective, flood_status = compute_flood_status_from_rain(rain_amount, 1)
                else:
                    # Mưa dự báo N giờ tới
                    h = int(hours)
                    total_rain = sum_rain_for_hours(lat, lon, h, forecast_json)
                    effective, flood_status = compute_flood_status_from_rain(total_rain, h)
                
                # Điều chỉnh nguy cơ ngập lụt ở khu vực núi (độ cao cao/độ dốc lớn)
                if elev > df["elevation"].mean() or slope > 10:
                    if flood_status == "Ngập cao":
                        flood_status = "Ngập thấp"
                    elif flood_status == "Ngập thấp":
                        flood_status = "Không ngập"

                # --- 3. Dự đoán sạt lở ---
                # Luôn dùng mưa 24h cho mô hình sạt lở
                rain_24h = sum_rain_for_hours(lat, lon, 24, forecast_json) 
                prob, label = predict_landslide(slope, elev, rain_mean_year, soil_type, dist_river, rain_24h)

                # Lưu kết quả
                st.session_state["result"] = {
                    "hours": hours,
                    "total_rain": total_rain,
                    "flood_status": flood_status,
                    "label": label,
                    "prob": prob,
                    "lat": lat,
                    "lon": lon,
                    "full_addr": full_addr,
                    "elev": elev,
                    "slope": slope,
                }

            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")

    with col2:
        st.header("Kết quả & Vị trí")
        
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
                    <li><b>Type 1</b> – Đất feralit (chủ yếu ở vùng đồi núi thấp, dốc vừa)</li>
                    <li><b>Type 2</b> – Đất mùn núi cao (vùng núi cao, lạnh)</li>
                    <li><b>Type 3</b> – Đất phù sa (vùng đồng bằng ven sông)</li>
                    <li><b>Type 4</b> – Đất xám bạc màu (vùng đồi, trung du)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        if "result" in st.session_state:
            res = st.session_state["result"]
            
            # Chọn màu cho nguy cơ sạt lở
            if res["label"] == "Không sạt lở":
                color = "🟢"
            elif res["label"] == "Nguy cơ thấp":
                color = "🟠"
            else:
                color = "🔴"
            
            # Text cho lượng mưa
            if res["hours"] == "Tức thì":
                rain_text = f"🌧 **Mưa hiện tại:** {res['total_rain']:.1f} mm/h"
            else:
                rain_text = f"🌧 **Tổng lượng mưa {res['hours']}h tới:** {res['total_rain']:.1f} mm"
            
            # Hiển thị kết quả
            st.markdown(
                f"""
                ### 📌 Thông tin Địa điểm
                - **Địa chỉ/Tọa độ:** {res["full_addr"]}
                - **Độ cao:** {res["elev"]:.2f} m | **Độ dốc:** {res["slope"]:.2f}°
                
                ### 🚨 Kết quả Dự báo
                {rain_text}  
                - **Nguy cơ Ngập lụt:** **{res["flood_status"]}** - **Nguy cơ Sạt lở:** **{res["label"]}** ({res["prob"]*100:.1f}%) {color}
            """
            )
            
            # Hiển thị bản đồ Folium
            m = folium.Map(location=[res["lat"], res["lon"]], zoom_start=14)
            folium.Marker(
                [res["lat"], res["lon"]], 
                popup=f"Sạt lở: {res['label']}", 
                tooltip=res["full_addr"],
                icon=folium.Icon(color="red" if res["label"] in ["Nguy cơ thấp", "Nguy cơ cao"] else "green", icon="cloud-download")
            ).add_to(m)
            st_folium(m, width=700, height=500)

# --- TAB 2: Bản đồ DEM ---
with tab2:
    st.header("🗺️ Bản đồ Địa hình Số (DEM)")
    st.markdown("Sử dụng bản đồ này để trực quan hóa địa hình và tự động lấy **Độ cao** và **Độ dốc** tại điểm bạn click.")
    
    # Lấy thông tin DEM đầu tiên để đặt vị trí trung tâm mặc định
    first = dem_infos[0]
    b = first["bounds"]
    # Chuyển đổi tọa độ trung tâm từ CRS của DEM về WGS84 (EPSG:4326)
    transformer = Transformer.from_crs(first["crs"], "EPSG:4326", always_xy=True)
    center_lon, center_lat = transformer.transform((b.left+b.right)/2, (b.top+b.bottom)/2)

    # Nếu có marker trước đó → lấy nó làm tâm bản đồ
    if "clicked_info" in st.session_state:
        last_lat, last_lon, _, _ = st.session_state["clicked_info"]
        start_center = [last_lat, last_lon]
    else:
        start_center = [center_lat, center_lon]   # Tâm mặc định ban đầu
    
    m2 = leafmap.Map(
        center=start_center,
        zoom=12 if "clicked_info" in st.session_state else 9,
        draw_control=False,
        measure_control=True
    )
    m2.add_basemap("OpenTopoMap")

    # Thêm DEM + SLOPE của các tỉnh
    for info in dem_infos:
        name = info["province"]
        # Thêm lớp DEM (Elevation) với bảng màu địa hình
        m2.add_raster(info["dem_path"], layer_name=f"{name} - Độ cao (Elevation)", opacity=0.6, colormap="terrain")
        # Thêm lớp SLOPE (Độ dốc) với bảng màu Đỏ-Vàng-Xanh (nguy hiểm)
        m2.add_raster(info["slope_path"], layer_name=f"{name} - Độ dốc (Slope)", opacity=0.6, colormap="RdYlGn_r")

    # 🔥 Nếu đã click trước đó, thêm marker TRƯỚC khi render map
    if "clicked_info" in st.session_state:
        lat, lon, elev, slope = st.session_state["clicked_info"]
        folium.Marker(
            [lat, lon],
            popup=f"Độ cao: {elev:.2f} m<br>Độ dốc: {slope:.2f}°",
            tooltip="Điểm đã chọn",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m2)

    # Render map và chờ click
    click = st_folium(m2, height=600, width=900)

    # Xử lý click mới
    if click and "last_clicked" in click and click["last_clicked"]:
        lat = click["last_clicked"]["lat"]
        lon = click["last_clicked"]["lng"]

        try:
            # Lấy giá trị DEM/SLOPE tại điểm click
            elev, slope = get_value_at_latlon(lat, lon)
            
            # Lưu và Rerun để cập nhật marker trên bản đồ
            st.session_state["clicked_info"] = (lat, lon, elev, slope)
            st.rerun() 
        except Exception as e:
            st.warning(f"Không tìm thấy DEM bao phủ vị trí này. Vui lòng chọn trong khu vực {', '.join(provinces)}.")

    if "clicked_info" in st.session_state:
        lat, lon, elev, slope = st.session_state["clicked_info"]
        st.markdown("---")
        st.markdown(f"""
        ### 📍 Thông tin Điểm đã chọn
        - **Vĩ độ:** **{lat:.5f}**
        - **Kinh độ:** **{lon:.5f}**
        - **Độ cao (từ DEM):** **{elev:.2f} m**
        - **Độ dốc (từ SLOPE):** **{slope:.2f} °**
        """)
        st.info("Bạn có thể copy các giá trị này và dán vào Tab **'Dự báo Sạt lở & Ngập lụt'**.")

# --- TAB 3: Báo cáo sạt lở ---
with tab3:
    st.header("📝 Báo cáo vụ sạt lở tại thực địa")
    st.markdown("Thông tin của bạn sẽ giúp chúng tôi cập nhật và cải thiện độ chính xác của mô hình.")

    st.markdown("Hãy cung cấp thông tin chi tiết nhất có thể:")

    colA, colB = st.columns(2)

    with colA:
        report_address = st.text_input("📍 Địa điểm xảy ra sạt lở")
        report_lat = st.number_input("Vĩ độ (nếu biết)", format="%.6f", key="report_lat")
        report_lon = st.number_input("Kinh độ (nếu biết)", format="%.6f", key="report_lon")

        # Dùng `st.selectbox` để chuẩn hóa dữ liệu đầu vào
        soil_type_report = st.selectbox(
            "Loại đất (quan sát/ước tính)", 
            ["Không rõ"] + soil_labels, 
            key="report_soil"
        )

    with colB:
        severity = st.selectbox(
            "Mức độ thiệt hại",
            [
                "Nhẹ – chỉ sạt vài điểm nhỏ",
                "Trung bình – cản trở giao thông",
                "Nặng – sạt lớn, chôn lấp tài sản",
                "Rất nặng – nguy hiểm đến tính mạng",
            ],
            key="report_severity"
        )

        causes = st.multiselect(
            "Nguyên nhân quan sát được",
            [
                "Mưa lớn kéo dài",
                "Đất bão hòa nước",
                "Gần khu vực sông suối",
                "Hoạt động xây dựng/cắt xẻ sườn dốc",
                "Động đất/rung chấn",
                "Không rõ",
            ],
            key="report_causes"
        )

        dist_river_report = st.number_input(
            "Khoảng cách ước tính đến sông gần nhất (m)", 
            min_value=0.0, 
            max_value=1000.0, 
            value=100.0,
            step=10.0,
            key="report_dist_river"
        )

    notes = st.text_area("Ghi chú bổ sung (tùy chọn)", key="report_notes")

    if st.button("📥 Gửi Báo cáo Sạt lở"):
        # Ở đây bạn sẽ thêm logic để lưu dữ liệu (ví dụ: vào database/file)
        
        # Tạo một dictionary để chứa dữ liệu báo cáo
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "address": report_address,
            "lat": report_lat,
            "lon": report_lon,
            "soil_type": soil_type_report,
            "severity": severity,
            "causes": ", ".join(causes),
            "dist_river": dist_river_report,
            "notes": notes
        }
        
        st.success("✅ Cảm ơn bạn đã cung cấp thông tin! Báo cáo của bạn đã được ghi nhận.")


