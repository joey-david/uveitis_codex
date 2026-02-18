# `uveitis_pipeline.vascularite`

Vascularite utilities and a fast, training-free vesselness proxy.

## Functions

| Function | Description |
|---|---|
| `nonblack_mask(rgb, thresh=8)` | Returns a 0/1 mask for non-black pixels (masked fundus region). |
| `vessel_mask(rgb, nb=None, p=VesselMaskParams())` | Vesselness proxy mask using CLAHE + multi-scale blackhat on green channel. |
| `obb_norm_to_poly_px(obb_norm, w, h)` | Converts normalized OBB (8 floats) to a 4-point polygon in pixel coordinates. |
| `bbox_to_poly_px(bbox_xywh)` | Converts COCO bbox `[x,y,w,h]` to a 4-point polygon in pixel coordinates. |
| `point_in_any_poly(x, y, polys)` | Tests whether a point lies inside any polygon. |
| `extract_patch(rgb, x, y, size)` | Extracts a square patch centered at `(x,y)` with padding as needed. |
| `mask_to_obbs(mask, min_area=40)` | Converts a binary mask into oriented boxes using `minAreaRect`. |

## Data classes

| Class | Description |
|---|---|
| `VesselMaskParams` | Parameters for `vessel_mask` (CLAHE, blackhat scales, threshold quantile, morphology). |

