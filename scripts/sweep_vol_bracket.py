import itertools, yaml, copy, subprocess, json, os, argparse, hashlib

# Small, fast grid. Adjust as needed.
# Defaults; can be overridden via CLI args below
grid = {
    "k_breakout": [1.25, 1.50],
    "sl_mult": [1.50, 1.75],
    "tp_mult": [4.0],
    "be_frac": [0.60],
    "trail_mult": [None],
    "entry_confirm.epsilon_atr": [0.05, 0.075],
    "entry_confirm.retest_atr": [0.10, 0.20],
    "entry_filters.range_expansion_mult": [1.0],
}

# Training (in-sample) and OOS months, using glob character sets for 2025.
TRAIN_PATTERN = "data/ticks/BTCUSDT/BTCUSDT-*-2025-0[56].csv"  # May–Jun
OOS_PATTERN = "data/ticks/BTCUSDT/BTCUSDT-*-2025-0[78].csv"    # Jul–Aug

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/vol_bracket.yaml")
ap.add_argument("--k_breakout", nargs="+", type=float)
ap.add_argument("--sl_mult", nargs="+", type=float)
ap.add_argument("--tp_mult", nargs="+", type=float)
ap.add_argument("--be_frac", nargs="+", type=float)
ap.add_argument("--trail_mult", nargs="+")
ap.add_argument("--drift_M", nargs="+", type=int)
ap.add_argument("--drift_eps_atr", nargs="+", type=float)
ap.add_argument("--cvd_z_thr", nargs="+", type=float)
args = ap.parse_args()

# Apply CLI overrides to grid
def _coerce_trail(vals):
    out = []
    for v in vals:
        if isinstance(v, str) and v.lower() in ("none", "null"):
            out.append(None)
        else:
            try:
                out.append(float(v))
            except Exception:
                out.append(v)
    return out

if args.k_breakout: grid["k_breakout"] = args.k_breakout
if args.sl_mult: grid["sl_mult"] = args.sl_mult
if args.tp_mult: grid["tp_mult"] = args.tp_mult
if args.be_frac: grid["be_frac"] = args.be_frac
if args.trail_mult: grid["trail_mult"] = _coerce_trail(args.trail_mult)
if args.drift_M: grid["direction.drift.M_min"] = args.drift_M
if args.drift_eps_atr: grid["direction.drift.eps_atr"] = args.drift_eps_atr
if args.cvd_z_thr: grid["direction.cvd.z_thr"] = args.cvd_z_thr

os.makedirs("results/sweeps", exist_ok=True)

base = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
keys = list(grid.keys())
rows = []

def run_cfg(cfg_dict):
    """Run the backtest with a temporary YAML and parse the final JSON line."""
    tag_vals = []
    for k in keys:
        tag_vals.append(f"{k}-{str(_get_nested(cfg_dict.get('strategy', {}), k, 'NA'))}")
    tag = "_".join(tag_vals).replace(" ", "")
    # Avoid very long filenames on Windows by hashing the tag
    digest = hashlib.sha1(tag.encode('utf-8')).hexdigest()[:12]
    fname = f"cfg_{digest}.yaml"
    path = os.path.join("results", "sweeps", fname)
    with open(path, "w", encoding="utf-8") as f:
        # Embed the full tag for traceability
        cfg_to_write = copy.deepcopy(cfg_dict)
        cfg_to_write.setdefault("sweep_tag", tag)
        yaml.safe_dump(cfg_to_write, f)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    out = subprocess.check_output(["python", "scripts/run_vol_bracket.py", "--config", path], text=True, env=env)
    # Parse the last JSON object from output (skip progress lines)
    js = None
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                js = json.loads(line)
                break
            except Exception:
                continue
    if js is None:
        raise RuntimeError(f"Failed to parse metrics JSON for tag={tag}")
    js["tag"] = tag
    return js

def _set_nested(d, path, value):
    cur = d
    parts = path.split('.')
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def _get_nested(d, path, default=None):
    cur = d
    for p in path.split('.'):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

for vals in itertools.product(*[grid[k] for k in keys]):
    cfg = copy.deepcopy(base)
    # set parameters
    for k, v in zip(keys, vals):
        if '.' in k:
            _set_nested(cfg.setdefault("strategy", {}), k, v)
        else:
            cfg.setdefault("strategy", {})[k] = v
    # ensure costs present (defaults if missing)
    cfg["strategy"].setdefault("fee_bps", 1.0)
    cfg["strategy"].setdefault("slip_bps", 0.5)
    # enforce retest_second_break mode and ATR recalc at entry, filter on
    cfg["strategy"].setdefault("entry_confirm", {})["mode"] = "retest_second_break"
    cfg["strategy"]["entry_recalc_atr"] = True
    cfg["strategy"].setdefault("entry_filters", {}).setdefault("range_expansion_mult", 1.0)

    # In-sample: restrict to May–Jun
    cfg["files"]["ohlcv"] = TRAIN_PATTERN
    train_js = run_cfg(cfg)

    # Pick best later; store training metrics and params
    rec = {
        "params": {k: _get_nested(cfg.get("strategy", {}), k) for k in keys},
        "train": {k: train_js.get(k) for k in ["sum_pnl", "p_win", "max_dd", "n_trades", "sharpe", "alerts"]},
        "tag": train_js["tag"],
    }
    rows.append(rec)

# Persist training sweep summary
with open("results/sweeps/summary.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)
print("Wrote results/sweeps/summary.json")

# Select the best by in-sample sum_pnl (you can refine for plateau manually)
best = max(rows, key=lambda r: (r["train"]["sum_pnl"], -r["train"]["max_dd"])) if rows else None
if best is None:
    raise SystemExit("No sweep rows produced")

# Run OOS for best params (Jul–Aug)
best_cfg = copy.deepcopy(base)
for k, v in best["params"].items():
    if '.' in k:
        _set_nested(best_cfg.setdefault("strategy", {}), k, v)
    else:
        best_cfg.setdefault("strategy", {})[k] = v
best_cfg["strategy"].setdefault("fee_bps", 1.0)
best_cfg["strategy"].setdefault("slip_bps", 0.5)
best_cfg["files"]["ohlcv"] = OOS_PATTERN

oos_js = run_cfg(best_cfg)

best_out = {
    "params": best["params"],
    "train": best["train"],
    "oos": {k: oos_js.get(k) for k in ["sum_pnl", "p_win", "max_dd", "n_trades", "sharpe", "alerts"]},
    "tag": best["tag"],
}
with open("results/sweeps/best_oos.json", "w", encoding="utf-8") as f:
    json.dump(best_out, f, indent=2)
print("Wrote results/sweeps/best_oos.json")
