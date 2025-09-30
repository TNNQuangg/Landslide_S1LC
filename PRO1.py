import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
from tkinter import ttk, messagebox
import tkintermapview

# =========================
# 1. Đọc dữ liệu CSV & train model
# =========================
df = pd.read_csv("Book1.csv")

# Encode soil_type
le = LabelEncoder()
df["soil_type"] = le.fit_transform(df["soil_type"])
soil_types = list(df["soil_type"].unique())
soil_labels = list(le.inverse_transform(soil_types))

# Train/test split
X = df[["slope", "elevation", "rain_mean_year", "soil_type", "dist_river", "rain_forecast_24h"]]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# =========================
# 2. Hàm lấy dự báo mưa
# =========================
def get_forecast_rain(api_key, lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()

    rain_24h = 0
    for block in data["list"][:8]:
        if "rain" in block and "3h" in block["rain"]:
            rain_24h += block["rain"]["3h"]

    return rain_24h

# =========================
# 3. Hàm dự đoán
# =========================
def predict_landslide(api_key, lat, lon, slope, elevation, rain_mean_year, soil_type, dist_river):
    rain_forecast_24h = get_forecast_rain(api_key, lat, lon)

    soil_encoded = le.transform([soil_type])[0]
    new_point = pd.DataFrame([{
        "slope": slope,
        "elevation": elevation,
        "rain_mean_year": rain_mean_year,
        "soil_type": soil_encoded,
        "dist_river": dist_river,
        "rain_forecast_24h": rain_forecast_24h
    }])

    prob = model.predict_proba(new_point)[0, 1]
    label = "Nguy cơ cao" if prob > 0.6 else "Nguy cơ trung bình" if prob > 0.3 else "Nguy cơ thấp"

    return rain_forecast_24h, prob, label

# =========================
# 4. GUI với Tkinter + Map + Lịch sử
# =========================
API_KEY = "2d4a3206becec3a48aa294ad6c759160"  # thay API key của bạn

# Tạo cửa sổ chính
root = tk.Tk()
root.title("Dự báo nguy cơ sạt lở")
root.geometry("1200x700")

# Khung bên trái
frame_left = tk.Frame(root, width=400, bg="white")
frame_left.pack(side="left", fill="y")

# Khung bên phải
frame_right = tk.Frame(root)
frame_right.pack(side="right", fill="both", expand=True)

# ======================
# Các ô nhập liệu
# ======================
tk.Label(frame_left, text="Thông tin vị trí", font=("Arial", 10, "bold")).pack(anchor="w", padx=5, pady=5)

tk.Label(frame_left, text="Vĩ độ (lat):").pack(anchor="w", padx=5)
entry_lat = tk.Entry(frame_left)
entry_lat.pack(anchor="w", padx=5, pady=2)

tk.Label(frame_left, text="Kinh độ (lon):").pack(anchor="w", padx=5)
entry_lon = tk.Entry(frame_left)
entry_lon.pack(anchor="w", padx=5, pady=2)

tk.Label(frame_left, text="").pack(pady=2)  # cách ra 1 dòng
tk.Label(frame_left, text="Địa hình", font=("Arial", 10, "bold")).pack(anchor="w", padx=5, pady=5)
tk.Label(frame_left, text="Độ dốc (%):").pack(anchor="w", padx=5)
entry_slope = tk.Entry(frame_left)
entry_slope.pack(anchor="w", padx=5, pady=2)

tk.Label(frame_left, text="Độ cao (m):").pack(anchor="w", padx=5)
entry_elev = tk.Entry(frame_left)
entry_elev.pack(anchor="w", padx=5, pady=2)

tk.Label(frame_left, text="Khoảng cách đến sông (km):").pack(anchor="w", padx=5)
entry_river = tk.Entry(frame_left)
entry_river.pack(anchor="w", padx=5, pady=2)

tk.Label(frame_left, text="").pack(pady=2)  # cách ra 1 dòng
tk.Label(frame_left, text="Khí hậu", font=("Arial", 10, "bold")).pack(anchor="w", padx=5, pady=5)
tk.Label(frame_left, text="Mưa trung bình năm (mm):").pack(anchor="w", padx=5)
entry_rain = tk.Entry(frame_left)
entry_rain.pack(anchor="w", padx=5, pady=2)

tk.Label(frame_left, text="Loại đất:").pack(anchor="w", padx=5)
combo_soil = ttk.Combobox(frame_left, values=soil_labels, state="readonly")
combo_soil.current(0)
combo_soil.pack(anchor="w", padx=5, pady=2)

# Kết quả
result_text = tk.StringVar()
result_label = tk.Label(frame_left, textvariable=result_text, font=("Arial", 10, "bold"))
result_label.pack(anchor="w", padx=5, pady=10)

# Hàm dự đoán
def on_predict():
    try:
        lat = float(entry_lat.get())
        lon = float(entry_lon.get())
        slope = float(entry_slope.get())
        elevation = float(entry_elev.get())
        rain_mean_year = float(entry_rain.get())
        soil_type = combo_soil.get()
        dist_river = float(entry_river.get())

        rain_24h, prob, label = predict_landslide(
            API_KEY, lat, lon, slope, elevation, rain_mean_year, soil_type, dist_river
        )

        # đổi màu theo nguy cơ
        if label == "Nguy cơ thấp":
            color = "green"
        elif label == "Nguy cơ trung bình":
            color = "orange"
        else:
            color = "red"

        result_text.set(
            f"🌧 Mưa dự báo 24h: {rain_24h:.1f} mm\n"
            f"Xác suất sạt lở: {prob*100:}%\n"
            f"→ {label}"
        )
        result_label.config(fg=color)

        # cập nhật bản đồ
        map_widget.set_position(lat, lon)
        map_widget.set_zoom(10)
        map_widget.set_marker(lat, lon, text=label)

        # thêm vào bảng lịch sử
        history_table.insert("", "end", values=(f"{lat:.4f}", f"{lon:.4f}", f"{prob*100:}%"))

    except Exception as e:
        messagebox.showerror("Lỗi", f"Dữ liệu nhập không hợp lệ:\n{e}")

# Nút dự đoán
tk.Button(frame_left, text="Dự đoán", command=on_predict).pack(pady=5)

# ======================
# Bảng lịch sử
# ======================
frame_history = tk.LabelFrame(frame_left, text="Lịch sử dự đoán")
frame_history.pack(fill="both", expand=True, padx=5, pady=5)

history_table = ttk.Treeview(frame_history, columns=("lat", "lon", "prob"), show="headings", height=6)
history_table.heading("lat", text="Vĩ độ")
history_table.heading("lon", text="Kinh độ")
history_table.heading("prob", text="Xác suất")

history_table.column("lat", width=80, anchor="center")
history_table.column("lon", width=80, anchor="center")
history_table.column("prob", width=100, anchor="center")

history_table.pack(fill="both", expand=True, padx=5, pady=5)

# ======================
# Bản đồ
# ======================
map_widget = tkintermapview.TkinterMapView(frame_right, width=800, height=700, corner_radius=0)
map_widget.pack(fill="both", expand=True)
map_widget.set_zoom(8)
map_widget.set_position(21.0285, 105.8542)  # Hà Nội mặc định

# Chạy GUI
root.mainloop()
