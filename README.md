# KrishiPredict
**India's Agricultural Intelligence and Crop Yield Prediction System** - a Streamlit dashboard with data analytics, rainfall monitoring and yield-prediction models.

## 1. Problem statement
Crop statistics and weather data come as large, separate, inconsistent tables. KrishiPredict cleans them, analyses production/yield/rainfall, and provides a transparent, honestly evaluated yield-prediction baseline.

## 2. Features
Overview KPIs and filters | Agricultural Analytics (production, yield, seasons, generated insights) | Crop Yield Prediction | District Intelligence | Rainfall Intelligence (historical + current + valid state-level link) | Crop Comparison with CSV export | Model Performance | About + data-quality report.

## 3. Technology stack
Python 3.10+, Pandas, NumPy, Scikit-learn, XGBoost (optional), Plotly, Streamlit, Joblib.

## 4. Datasets (in `data/`)
| File | Content |
|---|---|
| `krishipredict_crop_master.csv` | 402,342 rows; 34 states/UTs; 740 districts; 115 crops; 1997-2022 |
| `krishipredict_annual_rainfall.csv` | 4,090 rows; 36 IMD subdivisions; 1901-2015; annual mm |
| `krishipredict_current_daily_rainfall.csv` | 54,556 rows; 728 districts; 2026-08-19 to 2026-11-02; daily actual, daily normal, IMD category |

The original files `crop_production.csv`, `crop_clean.csv`, `rainfall.csv`, `rainfall_districtwise_daily_imd.csv` were **not supplied**. If you place `crop_production.csv` (columns State_Name, District_Name, Crop_Year, Season, Crop, Area, Production) in `data/`, the pipeline cleans it from scratch; the raw-file path was only tested on a synthetic file derived from the master data. The other three originals are not used.

Important data findings (see `processed/data_quality_report.csv`):
- **Units are not labelled.** Tonnes / hectares are assumed. Coconut (~8,100 per ha) is clearly not tonnes (nuts); Cotton(Lint), Jute, Mesta are bales by official convention (assumption - please verify). These crops are excluded from tonnage totals and from the ML model.
- 3,969 records have zero production with positive area (treated as "not reported"), 129 have area < 1 ha, 410 are extreme high-yield outliers (e.g. a 223 t/ha rice record). They are **flagged, not deleted**, and excluded only from yield statistics and ML.
- Reporting coverage grows from 9,433 records (1997) to 18,703 (2022), so raw yearly totals partly reflect coverage.
- Daily rainfall: 'ND' rows (630) have a placeholder 0 - set to missing; spelling variants (CHHATISGARH, TAMILNADU) merged; 2026-10-31 is absent; 3 districts have no data at all. There are no weekly/cumulative columns; weekly/monthly values are computed from non-overlapping daily values.

## 5. Rules enforced
2026 rainfall is never merged with 1997-2022 crops. Subdivisions are never treated as districts. Rainfall is not a model feature. The only rainfall-crop view is a state-level exploratory correlation (12 states that equal one subdivision, 1997-2015).

## 6. Folder structure
```
KrishiPredict/
  app.py                 Overview + navigation
  pages/                 1_Agricultural_Analytics ... 7_About
  src/                   data_loader, preprocessing, eda, feature_engineering, model_training,
                         prediction, rainfall_analysis, visualizations, utils
  data/  processed/  models/  assets/  tests/smoke_test.py
  run_preprocessing.py   train_model.py   config.py   requirements.txt
```

## 7. Installation (Windows PowerShell)
```powershell
cd KrishiPredict
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 8. Run
```powershell
python run_preprocessing.py      # optional: rebuilds processed/ (the app does it automatically if missing)
python train_model.py            # optional: retrain (about 1-3 min); a trained model is included
streamlit run app.py
```
`python train_model.py --quick` is a 12% smoke test; `--no-xgb` skips XGBoost.

## 9. Machine-learning methodology
- Target: yield = production / area (t/ha); models learn log(1+yield), metrics are on the t/ha scale.
- Inputs: state, district, crop, season, year, area (+ crop-state and crop-district keys). **Production is never an input.**
- Linear Regression (one-hot), Random Forest and XGBoost (target-encoded categories) inside one sklearn Pipeline.
- Chronological split: train 1997-2017, validation 2018-2019 (model selection), test 2020-2022 (untouched). The selected model is refit on 1997-2019 and re-scored on 2020-2022.
- Baselines: crop median and crop-district median.

## 10. Results (held-out test 2020-2022, 52,984 records; produced by `train_model.py`)
| Model | MAE t/ha | RMSE t/ha | R² |
|---|---|---|---|
| Baseline: crop median | 2.316 | 7.203 | 0.695 |
| Baseline: crop-district median | 1.374 | 4.845 | 0.862 |
| Linear Regression | 1.252 | 4.575 | 0.877 |
| Random Forest | 1.156 | 4.411 | 0.886 |
| XGBoost | 1.148 | 4.646 | 0.873 |
| **XGBoost, deployed (refit to 2019)** | **1.032** | **4.484** | **0.882** |

XGBoost was chosen on validation RMSE. On the test years Random Forest has a slightly lower RMSE, and all ML models beat the strongest simple baseline by only a modest margin (clearest on MAE). R² is variance explained, not accuracy, and is driven largely by which crop it is. Permutation importance: Crop and State dominate; Year contributes about nothing.

## 11. Limitations
Assumed units; no weather input; the model cannot extrapolate to years beyond its data; coverage changes over time; district names/boundaries change; no district map is drawn (no boundary data bundled); estimates are not agricultural advice.

## 12. Future work
Gridded district rainfall aligned to crop years, satellite indices, boundary maps, per-crop models, quantile prediction intervals.

## 13. Tests
`python tests/smoke_test.py` runs every page headlessly, applies filters and exercises the prediction form.
