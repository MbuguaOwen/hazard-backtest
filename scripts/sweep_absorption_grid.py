import copy, json
from pathlib import Path
import yaml

# Ensure repo root on sys.path for direct invocation
import sys, pathlib as _pl
ROOT = _pl.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.bracket_backtest import run_from_config


def sweep(cfg_path: str,
          z_list=(2.5,3.0,3.5),
          retest_list=(0.30,0.35,0.40),
          tp_list=(4.0,5.0,6.0),
          eps_list=(0.08,)):
    base = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    outdir = Path("results/sweeps"); outdir.mkdir(parents=True, exist_ok=True)
    for z in z_list:
        for r in retest_list:
            for tp in tp_list:
                for eps in eps_list:
                    cfg = copy.deepcopy(base)
                    strat = cfg.setdefault("strategy", {})
                    strat.setdefault("absorption", {})["z_delta_thr"] = float(z)
                    ec = strat.setdefault("entry_confirm", {})
                    ec["retest_atr"] = float(r)
                    ec["epsilon_atr"] = float(eps)
                    strat["tp_mult"] = float(tp)
                    # ensure modes
                    ec["mode"] = "absorption_reversal"
                    strat.setdefault("direction", {}).setdefault("mode", "absorption")
                    # write temp config
                    tmp = outdir / f"cfg_z{z}_r{r}_tp{tp}_eps{eps}.yaml"
                    yaml.safe_dump(cfg, open(tmp, "w", encoding="utf-8"))
                    print(f"[Sweep] Running {tmp}")
                    run_from_config(str(tmp))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vol_bracket.yaml")
    args = ap.parse_args()
    sweep(args.config)
