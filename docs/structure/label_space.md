# Label Space (Active Policy)

We intentionally do **not** train or infer all available uveitis classes at once, because the UWF label distribution is sparse and heavily imbalanced.

Current policy:
- Main detector predicts: FGADR-present classes + top-6 most frequent UWF-700/Uveitis classes.
- `vascularite` is handled by a dedicated model and is excluded from the main detector.

Active classes (current):
- Main detector: `exudats`, `hemorragie`, `macroanevrisme_arteriel`, `ischemie_retine`, `nodule_choroidien`, `hyalite`,
  `foyer_choroidien`, `granulome_choroidien`, `oedeme_papillaire`
- Separate model: `vascularite`

Canonical source of truth:
- `configs/active_label_space.yaml`
- `configs/main_detector_classes.txt`
