# `uveitis_pipeline.manifest`

Manifest scanning and split construction.

## Functions

| Function | Description |
|---|---|
| `_image_hw(path, read_size)` | Returns image `(width, height)` or `(0,0)` when size reads are disabled/fail. |
| `_safe_rel(path)` | Converts path to POSIX-like relative string. |
| `_scan_uwf700(root, read_size)` | Scans UWF700 structure and emits manifest rows. |
| `_scan_fgadr(root, read_size)` | Scans FGADR images and tracks mask availability in notes. |
| `_parse_deepdrid_csv(csv_path, dataset_tag, split, read_size)` | Parses DeepDRiD CSV metadata to manifest rows. |
| `_scan_deepdrid(root, read_size)` | Aggregates DeepDRiD UWF + regular subsets across split CSVs. |
| `_scan_eyepacs(root, read_size)` | Scans EyePACS train images and optional grade labels. |
| `_assign_random_splits(rows, ratios, seed)` | Patient-grouped random split assignment per dataset. |
| `build_manifests(cfg)` | Builds enabled dataset manifests and merged split dict. |
| `write_manifests(manifests, out_dir)` | Writes JSONL and CSV manifest files per dataset. |
| `write_splits(split_dict, out_path)` | Writes split JSON file. |
| `summarize_manifest(rows)` | Summarizes image count, split counts, label formats, and UWF class counts. |
