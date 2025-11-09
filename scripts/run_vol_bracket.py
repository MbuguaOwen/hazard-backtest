import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.engine.bracket_backtest import run_from_config

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vol_bracket.yaml")
    args = ap.parse_args()
    run_from_config(args.config)

