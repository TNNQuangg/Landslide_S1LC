import streamlit as st
import folium
from streamlit_folium import st_folium
import leafmap.foliumap as leafmap
import sys
from pathlib import Path

# Add root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.api_helper import get_coords, get_fc, get_rain_1h, calc_rain
from backend.flood_helper import check_flood
from backend.model_helper import pred_ls, s_lbls
from backend.gis_helper import get_val

# 1. Config
st.set_page_config(page_title="Dự báo Sạt lở", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    div[data-testid="stStatusWidget"] { display: none !important; }
    button [data-testid="stLoader"] { display: none !important; }
    [data-testid="stSidebar"][aria-expanded="true"] [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebar"][aria-expanded="true"] [data-testid="collapsedControl"] { display: none; }
    body { font-family: 'Inter', sans-serif; background-color: #ffffff; color: #333333; }
    .stApp { background-color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #eaeaea; }
    .stTextInput input, .stSelectbox div[data-baseweb="select"] { border-radius: 4px; }
    .stButton>button { background-color: #2c3e50; color: #ffffff; border: none; border-radius: 6px; padding: 0.5rem 1rem; font-weight: 500; transition: background-color 0.3s; }
    .stButton>button:hover { background-color: #34495e; color: white; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { padding-top: 10px; padding-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("Hệ thống Dự báo")

# 2. UI Variables
DEF_R_YR = 1750.0

def fmt_soil(lbl):
    s = str(lbl).lower()
    if "1" in s: return "Type 1"
    if "2" in s: return "Type 2"
    if "3" in s: return "Type 3"
    if "4" in s: return "Type 4"
    return lbl

with st.sidebar:
    st.header("Input")

    if "a_el" not in st.session_state: st.session_state.a_el = 0.0
    if "a_sl" not in st.session_state: st.session_state.a_sl = 0.0
    if "a_k" not in st.session_state: st.session_state.a_k = 0

    adr = st.text_input("Địa chỉ:", key=f"inp_{st.session_state.a_k}")

    if st.button("Cập nhật DEM", use_container_width=True):
        try:
            if not adr.strip(): st.stop()
            lt, ln, _ = get_coords(adr)
            el, sl = get_val(lt, ln)
            st.session_state.a_el = el
            st.session_state.a_sl = sl
            st.session_state["c_inf"] = (lt, ln, el, sl)
            st.session_state["l_adr"] = adr
            
            if "res" in st.session_state: del st.session_state["res"]
            st.success(f"Cao: {el:.1f}m | Dốc: {sl:.1f}°")
        except Exception as e: st.error(f"Lỗi: {e}")

    st.markdown("---")
    
    slp = st.number_input("Độ dốc", min_value=0.0, value=st.session_state.a_sl, format="%.1f")
    elv = st.number_input("Độ cao", min_value=0.0, value=st.session_state.a_el, format="%.1f")
    d_riv = st.number_input("Cách sông", min_value=0.0, value=100.0, step=10.0)
    soil = st.selectbox("Đất", s_lbls, format_func=fmt_soil)
    hrs = st.selectbox("Giờ", ["Tức thì", "1h tới", "3h tới", "6h tới", "12h tới", "24h tới"])

    btn_pre = st.button("Dự báo", type="primary", use_container_width=True)

# 3. Logic
if btn_pre:
    with st.spinner("Đang chạy..."):
        try:
            if adr.strip():
                lt, ln, f_adr = get_coords(adr)
                if st.session_state.get("l_adr") != adr:
                    elv, slp = get_val(lt, ln)
                    st.session_state.a_el = elv
                    st.session_state.a_sl = slp
                    st.session_state["c_inf"] = (lt, ln, elv, slp)
                    st.session_state["l_adr"] = adr
            elif "c_inf" in st.session_state:
                lt, ln, _, _ = st.session_state["c_inf"]
                f_adr = "Tự chọn"
            else:
                st.sidebar.warning("Nhập địa chỉ/chọn bản đồ.")
                st.stop()

            fc = get_fc(lt, ln)
            if hrs == "Tức thì":
                t_r = get_rain_1h(lt, ln)
                ef, f_sts = check_flood(t_r, 1, slp, soil)
            else:
                h_n = int(hrs.replace("h tới", "").strip())
                t_r = calc_rain(lt, ln, h_n, fc)
                ef, f_sts = check_flood(t_r, h_n, slp, soil)

            r_24 = calc_rain(lt, ln, 24, fc) 
            prb, lbl = pred_ls(slp, elv, DEF_R_YR, soil, d_riv, r_24)

            st.session_state["res"] = {
                "hrs": hrs, "t_r": t_r, "f_sts": f_sts,
                "lbl": lbl, "prb": prb, "lt": lt, "ln": ln,
                "f_adr": f_adr, "elv": elv, "slp": slp
            }
        except Exception as e:
            st.sidebar.error(f"Lỗi: {e}")

# 4. UI Layout
t1, t2 = st.tabs(["Bản đồ", "Báo cáo"])

with t1:
    st.markdown("### Bản đồ")
    if "res" in st.session_state:
        r = st.session_state["res"]
        col = "#2ecc71" if r["lbl"] == "An toàn" else "#f39c12" if "Trung bình" in r["lbl"] else "#e74c3c"
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"**Vị trí:** {r['f_adr']}")
            st.markdown(f"Cao: {r['elv']:.1f}m | Dốc: {r['slp']:.1f}° | Mưa: {r['t_r']:.1f}mm")
        with c2:
            st.markdown(f'''
                <div style="background:{col}; color:white; padding:10px; border-radius:6px; text-align:center;">
                    <h4>{r["lbl"]} ({r["prb"]*100:.1f}%)</h4>
                    <p>Ngập: {r['f_sts']}</p>
                </div>
            ''', unsafe_allow_html=True)
    elif "c_inf" in st.session_state:
        clt, cln, cel, csl = st.session_state["c_inf"]
        c_i1, c_i2, c_i3, c_i4 = st.columns(4)
        c_i1.metric("Lat", f"{clt:.5f}")
        c_i2.metric("Lon", f"{cln:.5f}")
        c_i3.metric("Cao", f"{cel:.1f}m")
        c_i4.metric("Dốc", f"{csl:.1f}°")

    d_cen = [22.3392, 103.8448] 
    
    if "res" in st.session_state:
        s_cen = [st.session_state["res"]["lt"], st.session_state["res"]["ln"]]
        zm = 14
    elif "c_inf" in st.session_state:
        s_cen = [st.session_state["c_inf"][0], st.session_state["c_inf"][1]]
        zm = 12
    else:
        s_cen = d_cen
        zm = 10

    m = leafmap.Map(location=s_cen, zoom_start=zm, draw_control=False, measure_control=True)
    m.add_basemap("CartoDB.Positron")

    if "res" in st.session_state:
        r = st.session_state["res"]
        i_col = "red" if "Rủi ro" in r["lbl"] else "green"
        folium.Marker(
            [r["lt"], r["ln"]], popup=r['lbl'], tooltip=r["f_adr"],
            icon=folium.Icon(color=i_col, icon="info-sign")
        ).add_to(m)
    elif "c_inf" in st.session_state:
        clt, cln, cel, csl = st.session_state["c_inf"]
        folium.Marker(
            [clt, cln], popup=f"Cao: {cel:.1f}m | Dốc: {csl:.1f}°", tooltip="Chọn",
            icon=folium.Icon(color="black", icon="info-sign")
        ).add_to(m)

    clk = st_folium(m, height=500, width=900, returned_objects=["last_clicked"])

    if clk and "last_clicked" in clk and clk["last_clicked"]:
        l_c, l_n = clk["last_clicked"]["lat"], clk["last_clicked"]["lng"]
        try:
            e_c, s_c = get_val(l_c, l_n)
            st.session_state["c_inf"] = (l_c, l_n, e_c, s_c)
            st.session_state.a_el = e_c
            st.session_state.a_sl = s_c
            st.session_state.a_k += 1
            st.session_state["l_adr"] = ""
            
            if "res" in st.session_state: del st.session_state["res"]
            st.rerun() 
        except: st.toast("Lỗi DEM.")

with t2:
    st.markdown("### Báo cáo")
    rc1, rc2 = st.columns(2)
    with rc1:
        r_adr = st.text_input("Địa điểm")
        r_lt = st.number_input("Lat", format="%.5f")
        r_ln = st.number_input("Lon", format="%.5f")
        r_s = st.selectbox("Đất", ["Không rõ"] + s_lbls, format_func=fmt_soil)
    
    with rc2:
        sev = st.selectbox("Mức độ", ["Nhẹ", "Trung bình", "Nặng"])
        cau = st.multiselect("Nguyên nhân", ["Mưa", "Đất", "Sông"])
        d_r = st.number_input("Cách sông", value=100.0)

    note = st.text_area("Ghi chú")

    if st.button("Gửi"):
        st.success("Xong!")
