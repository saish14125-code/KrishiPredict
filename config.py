"""Central configuration for KrishiPredict (paths, constants, domain rules)."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = BASE_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

# ---- Original (raw) datasets: optional. If absent, the master files below are used. ----
RAW_CROP = DATA_DIR / "crop_production.csv"
RAW_CROP_CLEAN = DATA_DIR / "crop_clean.csv"
RAW_ANNUAL_RAIN = DATA_DIR / "rainfall.csv"
RAW_DAILY_RAIN = DATA_DIR / "rainfall_districtwise_daily_imd.csv"

# ---- Previously generated cleaned datasets ----
MASTER_CROP = DATA_DIR / "krishipredict_crop_master.csv"
MASTER_ANNUAL_RAIN = DATA_DIR / "krishipredict_annual_rainfall.csv"
MASTER_DAILY_RAIN = DATA_DIR / "krishipredict_current_daily_rainfall.csv"

# ---- Outputs of the preprocessing pipeline ----
PROC_CROP = PROCESSED_DIR / "crop_processed.csv"
PROC_RAIN_ANNUAL = PROCESSED_DIR / "rainfall_annual_processed.csv"
PROC_RAIN_DAILY = PROCESSED_DIR / "rainfall_current_processed.csv"
QUALITY_REPORT = PROCESSED_DIR / "data_quality_report.csv"

# ---- Model artefacts ----
MODEL_FILE = MODELS_DIR / "best_model.joblib"
METRICS_FILE = MODELS_DIR / "model_metrics.json"
COMPARISON_FILE = MODELS_DIR / "model_comparison.csv"
TEST_PRED_FILE = MODELS_DIR / "test_predictions.csv"
IMPORTANCE_FILE = MODELS_DIR / "feature_importance.csv"
CROP_ERROR_FILE = MODELS_DIR / "crop_error_table.csv"

# ---- Domain rules ----
# Crops whose "Production" is NOT recorded in tonnes in the Indian official statistics.
# Evidence in the data: median "yield" of Coconut is ~8,140 per ha (impossible as t/ha, consistent
# with nuts/ha); Jute median 9.8 and Mesta 5.0 (impossible as t/ha for fibre, consistent with bales/ha).
# Source of the unit convention: Ministry of Agriculture & Farmers Welfare (DES) - ASSUMPTION, please verify.
NON_TONNE_CROPS = {
    "Coconut": "nuts",
    "Cotton(Lint)": "bales",
    "Jute": "bales",
    "Mesta": "bales",
    "Jute & Mesta": "bales",
}
MIN_VALID_AREA_HA = 1.0       # records below 1 ha give unstable yields (rounded production)
OUTLIER_IQR_K = 3.0           # far-out upper fence on log(yield) per crop (Tukey, k=3)

# ---- Chronological split ----
TRAIN_END_YEAR = 2017         # fit on 1997-2017 for model comparison
VAL_END_YEAR = 2019           # validation 2018-2019 (model selection)
TEST_START_YEAR = 2020        # held-out test 2020-2022 (never used for selection)
RANDOM_STATE = 42

# ---- Rainfall: states that map 1:1 to an IMD meteorological subdivision ----
STATE_TO_SUBDIVISION = {
    "Punjab": "PUNJAB",
    "Bihar": "BIHAR",
    "Kerala": "KERALA",
    "Tamil Nadu": "TAMIL NADU",
    "Himachal Pradesh": "HIMACHAL PRADESH",
    "Jharkhand": "JHARKHAND",
    "Chhattisgarh": "CHHATTISGARH",
    "Telangana": "TELANGANA",
    "Uttarakhand": "UTTARAKHAND",
    "Arunachal Pradesh": "ARUNACHAL PRADESH",
    "Odisha": "ORISSA",
    "Jammu And Kashmir": "JAMMU & KASHMIR",
}

RAIN_CATEGORY_LABELS = {
    "LE": "Large Excess", "E": "Excess", "N": "Normal",
    "D": "Deficient", "LD": "Large Deficient", "NR": "No Rain", "ND": "No Data",
}
RAIN_STATE_FIXES = {"CHHATISGARH": "CHHATTISGARH", "TAMILNADU": "TAMIL NADU"}

# ---- Theme ----
GREEN_DARK = "#14532d"
GREEN = "#166534"
GREEN_MID = "#22c55e"
GREEN_LIGHT = "#dcfce7"
BG = "#f3f7f3"
PALETTE = ["#166534", "#22c55e", "#ca8a04", "#0e7490", "#b45309", "#65a30d",
           "#7c3aed", "#be123c", "#475569", "#0d9488"]
