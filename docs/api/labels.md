# `uveitis_pipeline.labels`

Annotation harmonization and COCO building.

## Functions

| Function | Description |
|---|---|
| `_connected_component_boxes(mask, min_area)` | Extracts bounding boxes from connected components above area threshold. |
| `_parse_uwf_obb(label_path, width, height, class_map, image_path)` | Parses UWF OBB labels and converts quadrilateral points to AABB boxes. |
| `_parse_fgadr_masks(image_name, root, class_map, min_area)` | Parses FGADR class masks and converts components to AABB boxes. |
| `_project_to_global(box, crop_meta, global_meta)` | Projects crop-space box into global resized coordinates. |
| `_clip_box(box, width, height)` | Clips box to image bounds and drops invalid boxes. |
| `_to_coco_bbox(box)` | Converts `[x1,y1,x2,y2]` to COCO `[x,y,w,h]`. |
| `_intersect(a, b)` | Computes intersection box between two axis-aligned boxes. |
| `_box_area(box)` | Computes axis-aligned box area. |
| `build_coco_from_manifest(...)` | Builds COCO dict for selected split rows in global or tile mode, writes output file. |
| `summarize_coco(coco)` | Summarizes COCO image/annotation counts and box area stats. |

## Notes
- UWF source OBB is preserved in annotation field `obb` for global mode.
- Tile mode filters partial boxes with `min_tile_box_ratio`.
