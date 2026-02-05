# `uveitis_pipeline.common`

Shared I/O and utility helpers.

## Functions

| Function | Description |
|---|---|
| `ensure_dir(path)` | Creates directory tree if missing and returns `Path`. |
| `load_yaml(path)` | Loads YAML config and returns dict (or empty dict). |
| `save_json(path, data)` | Writes JSON with indentation, creating parent dirs. |
| `save_jsonl(path, rows)` | Writes list of dict rows as JSONL. |
| `read_jsonl(path)` | Reads JSONL into list of dicts. |
| `read_image(path)` | Loads RGB image with OpenCV and raises if missing. |
| `write_image(path, image)` | Saves RGB image to disk (BGR conversion internally). |
| `draw_boxes(image, boxes, labels, color)` | Draws axis-aligned boxes + text labels on image. |
| `set_seed(seed)` | Sets Python, NumPy, and Torch seeds. |
| `parse_eye_token(name)` | Heuristic parser for eye side (`L` / `R` / `None`). |
