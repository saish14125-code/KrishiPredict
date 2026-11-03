import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import time
from streamlit.testing.v1 import AppTest
from pathlib import Path
R=str(Path(__file__).resolve().parents[1])+"/"
def summary(at, name):
    ch=len(at.get("plotly_chart")); md=len(at.markdown); df=len(at.dataframe)
    print(f"{name}: charts={ch} markdown={md} dataframes={df} exc={len(at.exception)} err={len(at.error)} warn={len(at.warning)}")
    for e in at.exception: print("  EXC", e.value[:500])
    for e in at.error: print("  ERR", e.value[:300])
for p in ["1_Agricultural_Analytics","2_Crop_Yield_Prediction","3_District_Intelligence","4_Rainfall_Intelligence","5_Crop_Comparison","6_Model_Performance","7_About"]:
    at=AppTest.from_file(R+f"pages/{p}.py",default_timeout=120).run(); summary(at,p)
# app.py with navigation (Overview)
at=AppTest.from_file(R+"app.py",default_timeout=120).run(); summary(at,"app.py (Overview)")
# Overview filter interaction
at.multiselect(key="ov_s").set_value(["Punjab"]).run(); summary(at,"Overview state=Punjab")
at.multiselect(key="ov_c").set_value(["Rice","Wheat"]).run(); summary(at,"Overview Punjab+Rice/Wheat")
at.multiselect(key="ov_c").set_value(["Coconut"]).run(); summary(at,"Overview Punjab+Coconut (edge)")
at.multiselect(key="ov_s").set_value(["Chandigarh"]).run(); at.multiselect(key="ov_c").set_value(["Apple"]).run(); summary(at,"Overview Chandigarh+Apple (likely empty)")
# Prediction click
at=AppTest.from_file(R+"pages/2_Crop_Yield_Prediction.py",default_timeout=120).run()
at.button[0].click().run(); summary(at,"Predict (defaults)")
print([m.value[:160] for m in at.markdown if "kpi" in m.value][:3])
at.number_input[0].set_value(2030).run(); at.button[0].click().run(); summary(at,"Predict year 2030")
print([w.value[:150] for w in at.warning])
at.number_input[1].set_value(0.0).run(); at.button[0].click().run(); summary(at,"Predict area=0")
