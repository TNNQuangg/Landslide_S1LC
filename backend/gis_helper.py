import streamlit as st
import rasterio
import numpy as np
from pathlib import Path
from pyproj import Transformer
import gdown

# 4. GIS
BASE_DIR = Path(__file__).resolve().parent.parent
DAT_D = BASE_DIR / "data"

DEM_F = {
    "Tuyen Quang_DEM.tif": "1g2TTXaV4Ce3-ztXxPxQr327Rqz-S-XwC",
    "Tuyen Quang_DEM_SLOPE.tif": "1E8G9DHq8nf02MjySzXwZ8GHn8UeHEYna",
    "Yen Bai_DEM.tif": "1OSquH03dGdfrMVvoCmt4eMVFlKL6mZZO",
    "Yen Bai_DEM_SLOPE.tif": "1ITsZmNHz-TjVcOvH2QPD6Wp13kUDEsov",
    "Ha Giang_DEM.tif": "1Fh7X7DJNpZ2qvOgcrDm-Vf_YomCprgqK",
    "Ha Giang_DEM_SLOPE.tif": "16AGmHaPIhYui0hqurG2bOSHWdSC2m2vG",
    "Lao Cai_DEM.tif": "1Cl_3pDOUN4xJXr2-OroZPs6mbJF--oBm",
    "Lao Cai_DEM_SLOPE.tif": "1IjctcAWGzeINTqkh1nOCVF4aWkXoAyLF"
}

@st.cache_resource
def chk_dems():
    for f, i in DEM_F.items():
        p = DAT_D / f
        if not p.exists():
            url = f"https://drive.google.com/uc?id={i}"
            gdown.download(url, str(p), quiet=False, use_cookies=False)
    return True

chk_dems()

@st.cache_resource
def get_ras(p):
    return rasterio.open(p)

@st.cache_resource
def get_dem_meta():
    res = []
    provs = ["Lao Cai", "Yen Bai", "Ha Giang", "Tuyen Quang"]
    for pr in provs:
        dp = DAT_D / f"{pr}_DEM.tif"
        sp = DAT_D / f"{pr}_DEM_SLOPE.tif"
        if dp.exists() and sp.exists():
            with rasterio.open(dp) as s:
                tf = Transformer.from_crs("EPSG:4326", s.crs, always_xy=True)
                res.append({
                    "prv": pr, "d_p": str(dp), "s_p": str(sp),
                    "crs": s.crs, "bnd": s.bounds, "nd": s.nodata, "tf": tf
                })
    return res

d_meta = get_dem_meta()

@st.cache_data(ttl=3600)
def get_val(lat, lon):
    lat = round(lat, 5)
    lon = round(lon, 5)
    for m in d_meta:
        b = m["bnd"]
        tf = m["tf"]
        x, y = tf.transform(lon, lat)

        if not (b.left <= x <= b.right and b.bottom <= y <= b.top): continue

        ds = get_ras(m["d_p"])
        ss = get_ras(m["s_p"])

        try:
            eg = ds.sample([(x, y)])
            e = float(list(eg)[0][0])
            if np.isnan(e) or e == m["nd"]: continue
        except: continue

        try:
            sg = ss.sample([(x, y)])
            sl = float(list(sg)[0][0])
            if np.isnan(sl) or sl == ss.nodata: continue
        except: continue

        return e, sl
    raise ValueError("Lỗi DEM")
