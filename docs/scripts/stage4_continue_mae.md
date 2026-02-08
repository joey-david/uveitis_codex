# `stage4_continue_mae.py`

## Purpose
Optional hook to continue RETFound MAE pretraining on local data if `main_pretrain.py` exists.

## CLI
```bash
python scripts/stage4_continue_mae.py --retfound-dir ../RETFound --data-path preproc/global_1024 [--output-dir pretrain_runs/mae_uwf] [--epochs 50] [--batch-size 64] [--run]
```

## Behavior
- Without `--run`: prints command preview.
- With `--run`: executes subprocess command.
- If pretrain script is missing: prints a structured skip message.

## Functions
| Function | Description |
|---|---|
| `main()` | Resolves RETFound path, builds MAE command, optionally runs it. |
