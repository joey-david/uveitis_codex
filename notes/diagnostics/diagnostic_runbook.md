# Diagnostics Runbook (Subagent Dispatch)

## 1) Run baseline diagnostics only (fast)

```bash
docker compose run --rm train \
  python scripts/diagnostics/dispatch_subagents.py \
  --config configs/diagnostics/deep_diagnostics.yaml
```

Outputs:
- `eval/diagnostics/data_qc/datasets_integrity.json`
- `eval/diagnostics/preproc_ablation/mask_norm_qc.json`
- `eval/diagnostics/baseline_repro/main_baseline.json`
- `eval/diagnostics/baseline_repro/vascularite_baseline.json`
- `eval/diagnostics/dispatch_status.json`
- logs: `eval/diagnostics/logs/*.log`

## 2) Run full diagnostics (includes hypothesis sweeps)

```bash
docker compose run --rm train \
  python scripts/diagnostics/dispatch_subagents.py \
  --config configs/diagnostics/deep_diagnostics.yaml \
  --run-sweeps
```

Additional outputs:
- `eval/diagnostics/hypothesis_sweeps/*.json`
- `eval/diagnostics/hypothesis_sweeps/sweep_summary.json`
- logs: `eval/diagnostics/logs/subagent_c/*.log`

## 3) Track progress

```bash
ls eval/diagnostics/logs
tail -n 80 eval/diagnostics/logs/subagent_c_dispatch.log
tail -n 80 eval/diagnostics/logs/subagent_c/C1_repro_seed1_e3.log
```

## 4) Aggregate best candidates quickly

```bash
python - <<'PY'
import json, pathlib
files = sorted(pathlib.Path('eval/diagnostics/hypothesis_sweeps').glob('C*.json'))
rows=[]
for p in files:
    o=json.loads(p.read_text())
    rows.append((o['name'], o['eval_map50_95'], o['eval_map50']))
for name,m95,m50 in sorted(rows, key=lambda x:x[1], reverse=True):
    print(f\"{name:24s} mAP50-95={m95:.4f} mAP50={m50:.4f}\")
PY
```
