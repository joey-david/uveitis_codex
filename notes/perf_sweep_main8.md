# Main8 (Non-Vascularite) Detector: Perf Sweep Notes

Label-space policy: `configs/active_label_space.yaml` (8 main classes, `vascularite` separate).

## Experiments

### mixed_main8_os20_y8m_1280_e20
- Train: FGADR main8 + UWF main8 oversampled 20x (symlink repeats)
- Val: UWF main8 only
- Data: `out/yolo_obb/mixed_fgadr1_uwf20_val_uwf_main8/data.yaml`
- Run: `runs/obb/runs/yolo_obb/mixed_main8_os20_y8m_1280_e20/`
- Epoch 1: mAP50(B)=0.0844, mAP50-95(B)=0.0325
- Epoch 2: mAP50(B)=0.0641, mAP50-95(B)=0.0198

