# Landslide S1LC

Ứng dụng Streamlit dự báo nguy cơ sạt lở/ngập lụt và hiển thị DEM/SLOPE cho một số tỉnh miền núi phía Bắc.

## Cài đặt môi trường riêng

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Nếu PowerShell chặn activate:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Cấu hình API thời tiết

Tạo biến môi trường:

```powershell
$env:OPENWEATHER_API_KEY="your_openweather_key"
```

Hoặc thêm vào `.streamlit/secrets.toml`:

```toml
OPENWEATHER_API_KEY = "your_openweather_key"
```

## Train model

```powershell
python train_model.py --data Book1.csv --model-out models/landslide_pipeline.joblib --metrics-out models/model_metrics.json
```
## Chạy app

```powershell
streamlit run PRO1.py
```
