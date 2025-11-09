
# Post-Gate Excursion Study

Measures how far price moves within the hazard horizon after each alert, **independent of any entry model**.
All moves are scaled by ATR at the alert time.

## Metrics per alert
- `up_atr`  = (max high - P0) / ATR_ref
- `dn_atr`  = (P0 - min low) / ATR_ref
- `max_exc_atr` = max(up_atr, dn_atr)
- `net_move_atr` = (close_at_horizon - P0) / ATR_ref

## Outputs
- `results/excursions/excursions.csv` — per-alert excursions
- `results/excursions/exceedance.csv` — fraction exceeding thresholds
- `results/excursions/summary.json` — quantiles and exceedance map

## Run
```bash
python scripts/measure_excursions.py --config configs/excursions.yaml
```
