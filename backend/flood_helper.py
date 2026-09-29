# 2. Ngập
def check_flood(rain, h, slp, soil):
    sf = 1.0
    s_str = str(soil).lower()
    if "1" in s_str: sf = 1.0
    elif "2" in s_str: sf = 1.2
    elif "3" in s_str: sf = 1.5
    elif "4" in s_str: sf = 0.8
        
    base = 10.0 * sf
    slp_f = 1 + (slp / 10) 
    
    rate = base * slp_f
    ef_r = max(0.0, rain - (rate * h))
    
    if ef_r <= 30: return ef_r, "Không ngập"
    elif ef_r <= 80: return ef_r, "Ngập nhẹ"
    return ef_r, "Ngập nặng"
