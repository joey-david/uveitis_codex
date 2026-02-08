# `uveitis_pipeline.dataset`

COCO dataset adapter for Torchvision detection models.

## Class: `CocoDetectionDataset`

### Constructor
- `__init__(coco_json, resize=None)`: loads COCO file and indexes annotations by image ID.

### Methods
| Method | Description |
|---|---|
| `__len__()` | Returns number of images. |
| `__getitem__(idx)` | Loads image + annotations and returns `(image_tensor, target_dict)`. |
| `build_class_balanced_sampler()` | Returns `WeightedRandomSampler` weighted inversely by class frequency. |

## Functions

| Function | Description |
|---|---|
| `collate_fn(batch)` | Detection dataloader collate: list of images and list of targets. |
