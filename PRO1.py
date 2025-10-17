import pandas as pd
import requests
import threading
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
from tkinter import ttk, messagebox
import tkintermapview
import time

# =========================
# 1. Đọc dữ liệu CSV & train model
# =========================
df = pd.read_csv("Book1.csv")

le = LabelEncoder()
df["soil_type"] = le.fit_transform(df["soil_type"])
soil_types = list(df["soil_type"].unique())
soil_labels = list(le.inverse_transform(soil_types))

X = df[["slope", "elevation", "rain_mean_year", "soil_type", "dist_river", "rain_forecast_24h"]]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# =========================
# 2. API OpenWeather + OSM
# =========================
API_KEY = "2d4a3206becec3a48aa294ad6c759160"  # thay API key của bạn nếu cần

def get_coordinates_from_osm(address):
    """
    Lấy kinh độ & vĩ độ từ địa chỉ qua OpenStreetMap (Nominatim)
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1, "addressdetails": 1}
    headers = {"User-Agent": "LandslidePredictor/1.0 (contact: example@example.com)"}

    res = requests.get(url, params=params, headers=headers, timeout=10)
    res.raise_for_status()
    data = res.json()
    if not data:
        raise ValueError("Không tìm thấy địa chỉ trên bản đồ.")
    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    display_name = data[0].get("display_name", "")
    return lat, lon, display_name

def get_current_weather(lat, lon):
    """
    Lấy thời tiết hiện tại (dùng cho 1h)
    """
    url_current = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    resp = requests.get(url_current, timeout=10)
    resp.raise_for_status()
    return resp.json()

def get_forecast(lat, lon):
    """
    Lấy forecast 3h-block từ OpenWeather (dùng cho 3h/6h)
    """
    url_forecast = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    resp = requests.get(url_forecast, timeout=10)
    resp.raise_for_status()
    return resp.json()

def sum_rain_for_hours(lat, lon, hours, current_json=None, forecast_json=None):
    """
    Tính tổng lượng mưa dự kiến trong 'hours' giờ tới.
    - nếu hours == 1: ưu tiên current_json["rain"]["1h"]
    - nếu hours in (3,6): cộng các block 3h từ forecast_json tương ứng
    Trả về total_rain_mm (float).
    """
    total = 0.0
    # lấy current nếu chưa truyền
    if current_json is None:
        try:
            current_json = get_current_weather(lat, lon)
        except Exception:
            current_json = {}

    # 1 giờ: lấy current["rain"]["1h"] nếu có, ngược lại khai thác forecast 3h block và scale xuống 1h
    if hours == 1:
        total = float(current_json.get("rain", {}).get("1h", 0.0) or 0.0)
        # fallback: nếu không có current rain, dùng forecast đầu tiên và chia cho 3
        if total == 0.0:
            if forecast_json is None:
                try:
                    forecast_json = get_forecast(lat, lon)
                except Exception:
                    forecast_json = {}
            first_block = forecast_json.get("list", [])
            if first_block:
                block = first_block[0]
                r3 = float(block.get("rain", {}).get("3h", 0.0) or 0.0)
                total = r3 / 3.0  # xấp xỉ 1h
    else:
        # cần forecast
        if forecast_json is None:
            forecast_json = get_forecast(lat, lon)
        blocks_needed = (hours + 2) // 3  # 3h per block; 3->1, 6->2
        for block in forecast_json.get("list", [])[:blocks_needed]:
            total += float(block.get("rain", {}).get("3h", 0.0) or 0.0)

    return total

def compute_flood_status_from_rain(total_rain_mm, hours, drainage_rate_mm_per_hour=50.0):
    """
    Tính trạng thái ngập:
    - khả năng thoát = drainage_rate_mm_per_hour * hours
    - effective = max(0, total_rain_mm - capacity)
    - phân loại effective: <=50: Không ngập, <=100: Ngập thấp, >100: Ngập cao
    """
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

def predict_landslide_using_rain(lat, lon, slope, elevation, rain_mean_year, soil_type, dist_river, rain_24h):
    # mã hóa soil_type (nếu unseen sẽ ném lỗi; giữ như cũ)
    soil_encoded = le.transform([soil_type])[0]
    new_point = pd.DataFrame([{
        "slope": slope,
        "elevation": elevation,
        "rain_mean_year": rain_mean_year,
        "soil_type": soil_encoded,
        "dist_river": dist_river,
        "rain_forecast_24h": rain_24h
    }])
    probs = model.predict_proba(new_point)[0]
    # giả sử nhãn dương ở index 1 như cũ
    prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
    label = "Nguy cơ cao" if prob > 0.6 else "Nguy cơ trung bình" if prob > 0.3 else "Nguy cơ thấp"
    return prob, label

# =========================
# 3. GUI với Tkinter
# =========================
root = tk.Tk()
root.title("Dự báo nguy cơ sạt lở + Ngập lụt")
root.geometry("1200x720")

frame_left = tk.Frame(root, width=420, bg="white")
frame_left.pack(side="left", fill="y")
frame_right = tk.Frame(root)
frame_right.pack(side="right", fill="both", expand=True)

# ======================
# Ô nhập liệu
# ======================
tk.Label(frame_left, text="Thông tin vị trí", font=("Arial", 10, "bold")).pack(anchor="w", padx=5, pady=5)
tk.Label(frame_left, text="Địa chỉ:").pack(anchor="w", padx=5)
entry_address = tk.Entry(frame_left, width=45)
entry_address.pack(anchor="w", padx=5, pady=2)

tk.Label(frame_left, text="").pack(pady=2)
tk.Label(frame_left, text="Địa hình", font=("Arial", 10, "bold")).pack(anchor="w", padx=5, pady=5)
tk.Label(frame_left, text="Độ dốc (%):").pack(anchor="w", padx=5)
entry_slope = tk.Entry(frame_left); entry_slope.pack(anchor="w", padx=5, pady=2)
tk.Label(frame_left, text="Độ cao (m):").pack(anchor="w", padx=5)
entry_elev = tk.Entry(frame_left); entry_elev.pack(anchor="w", padx=5, pady=2)
tk.Label(frame_left, text="Khoảng cách đến sông (km):").pack(anchor="w", padx=5)
entry_river = tk.Entry(frame_left); entry_river.pack(anchor="w", padx=5, pady=2)

tk.Label(frame_left, text="").pack(pady=2)
tk.Label(frame_left, text="Khí hậu", font=("Arial", 10, "bold")).pack(anchor="w", padx=5, pady=5)
tk.Label(frame_left, text="Mưa trung bình năm (mm):").pack(anchor="w", padx=5)
entry_rain = tk.Entry(frame_left); entry_rain.pack(anchor="w", padx=5, pady=2)

tk.Label(frame_left, text="Loại đất:").pack(anchor="w", padx=5)
combo_soil = ttk.Combobox(frame_left, values=soil_labels, state="readonly")
if soil_labels:
    combo_soil.current(0)
combo_soil.pack(anchor="w", padx=5, pady=2)

# Combobox chọn khung thời gian ngập
tk.Label(frame_left, text="Dự báo ngập lụt trong:").pack(anchor="w", padx=5, pady=(8,0))
combo_hours = ttk.Combobox(frame_left, values=["1 giờ", "3 giờ", "6 giờ"], state="readonly", width=10)
combo_hours.current(0)
combo_hours.pack(anchor="w", padx=5, pady=2)

result_text = tk.StringVar()
result_label = tk.Label(frame_left, textvariable=result_text, font=("Arial", 10, "bold"), justify="left")
result_label.pack(anchor="w", padx=5, pady=10)

current_marker = [None]

# ======================
# Dự đoán (threaded)
# ======================
def on_predict():
    def run_prediction():
        try:
            address = entry_address.get().strip()
            if not address:
                raise ValueError("Vui lòng nhập địa chỉ cụ thể!")

            # Lấy tọa độ từ OSM
            lat, lon, full_addr = get_coordinates_from_osm(address)
            time.sleep(1)  # tránh limit OSM

            # Lấy dữ liệu current + forecast một lần để dùng ngắn gọn
            try:
                current_json = get_current_weather(lat, lon)
            except Exception:
                current_json = {}
            try:
                forecast_json = get_forecast(lat, lon)
            except Exception:
                forecast_json = {}

            # Xác định hours từ combobox
            hours_text = combo_hours.get()
            hours = int(hours_text.split()[0]) if hours_text else 1

            # Tính tổng mưa trong khung hours
            total_rain = sum_rain_for_hours(lat, lon, hours, current_json=current_json, forecast_json=forecast_json)

            # Tính effective rain sau khi trừ khả năng thoát nước (50 mm/h)
            effective_rain, flood_status = compute_flood_status_from_rain(total_rain, hours, drainage_rate_mm_per_hour=50.0)

            # Lấy rain_24h dùng cho model (dùng forecast_json)
            rain_24h = 0.0
            for block in forecast_json.get("list", [])[:8]:
                rain_24h += float(block.get("rain", {}).get("3h", 0.0) or 0.0)

            # Lấy các input khác cho model
            slope = float(entry_slope.get())
            elevation = float(entry_elev.get())
            rain_mean_year = float(entry_rain.get())
            soil_type = combo_soil.get()
            dist_river = float(entry_river.get())

            prob, landslide_label = predict_landslide_using_rain(lat, lon, slope, elevation, rain_mean_year, soil_type, dist_river, rain_24h)

            prob_percent = f"{prob * 100:.2f}%"
            color = "green" if landslide_label == "Nguy cơ thấp" else "orange" if landslide_label == "Nguy cơ trung bình" else "red"

            # Hiển thị kết quả: bao gồm total_rain trong hours, effective_rain, flood_status, và sạt lở
            root.after(0, lambda: (
                result_text.set(
                    f"🌧 Tổng mưa dự kiến {hours}h tới: {total_rain:.1f} mm\n"
                    f"🚨 Dự báo ngập: {flood_status}\n\n"
                    f"⛰ Xác suất sạt lở: {prob_percent}\n"
                    f"→ {landslide_label}"
                ),
                result_label.config(fg=color),
                update_map(lat, lon, landslide_label),
                history_table.insert("", "end", values=(address, f"{hours}h", prob_percent, flood_status))
            ))

        except Exception as e:
            root.after(0, lambda: messagebox.showerror("Lỗi", f"Dữ liệu nhập không hợp lệ hoặc lỗi API:\n{e}"))

    threading.Thread(target=run_prediction, daemon=True).start()

def update_map(lat, lon, label):
    if current_marker[0] is not None:
        try:
            current_marker[0].delete()
        except Exception:
            pass
    map_widget.set_position(lat, lon)
    map_widget.set_zoom(11)
    current_marker[0] = map_widget.set_marker(lat, lon, text=label)

tk.Button(frame_left, text="Dự đoán", command=on_predict).pack(pady=8)

# ======================
# Bảng lịch sử
# ======================
frame_history = tk.LabelFrame(frame_left, text="Lịch sử dự đoán")
frame_history.pack(fill="both", expand=True, padx=5, pady=5)

history_table = ttk.Treeview(frame_history, columns=("addr", "hours", "prob", "flood"), show="headings", height=6)
for col, text, w in [("addr", "Địa chỉ", 200), ("hours", "Khung giờ", 80), ("prob", "Xác suất sạt lở", 110), ("flood", "Ngập", 100)]:
    history_table.heading(col, text=text)
    history_table.column(col, width=w, anchor="center")
history_table.pack(fill="both", expand=True, padx=5, pady=5)

# ======================
# Bản đồ
# ======================
map_widget = tkintermapview.TkinterMapView(frame_right, width=820, height=720, corner_radius=0)
map_widget.pack(fill="both", expand=True)
map_widget.set_zoom(8)
map_widget.set_position(21.0285, 105.8542)

root.mainloop()
