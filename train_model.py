"""Train the yield models:  python train_model.py   (add --quick for a fast smoke test)"""
from src.model_training import run_training
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--no-xgb", action="store_true")
a = ap.parse_args(); run_training(a.quick, a.no_xgb)
