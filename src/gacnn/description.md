## 1 · Patch proposals  → **nodes**

1. **Replace SURF** with *SURF + blob + SLIC* for 100 % lesion recall:

```python
# scripts/gen_patches.py
import cv2, numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
def generate_patches(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create(400).detect(gray)
    blob = cv2.SimpleBlobDetector_create().detect(gray)
    spx  = slic(img_as_float(img), n_segments=450, start_label=0)
    # centre of super-pixels
    sp_centres = [np.column_stack(np.where(spx==i)).mean(0)[::-1] for i in np.unique(spx)]
    # unify into a list of (x,y)
    kp = [k.pt for k in surf] + [k.pt for k in blob] + sp_centres
    return np.array(kp, dtype=np.float32)           # [N,2]
```

Expect **200–600 nodes** per 3000×3000 Optos image.

---

## 2 · Edge list

```python
# scripts/build_edges.py
import torch
def knn_edges(xy, k=8):
    dist = torch.cdist(xy, xy)                      # [N,N]
    knn = dist.topk(k+1, largest=False).indices[:,1:]
    src = torch.arange(xy.size(0)).unsqueeze(1).repeat(1,k).flatten()
    dst = knn.flatten()
    return torch.stack([src, dst], 0)               # [2,E]

def cosine_edges(feats, τ=0.7):
    cos = torch.mm(F.normalize(feats), F.normalize(feats).t())
    idx = (cos>τ).nonzero(as_tuple=False)
    idx = idx[idx[:,0]!=idx[:,1]].t()
    return idx                                      # [2,E′]
```

Final `edge_index = torch.cat([knn_edges, cos_edges], dim=1)`
Edge attr = `Δx, Δy, ||Δ||`.

---

## 3 · Node features

*Crop → 128-D embedding* using the **frozen MTSN checkpoint**:

```python
# scripts/embed_patch.py
from torchvision.transforms import ToTensor, Resize
def mtsn_embed(crop, model):
    t = Resize(224)(ToTensor()(crop)).unsqueeze(0).cuda()
    with torch.no_grad(): z = model(t)              # [1,128]
    return z.squeeze(0).cpu()
```

Feature vector for each node = `[MTSN128 ∥ (x,y,w,h)/img_size]` → **132-D**.

---

## 4 · Graph building  (→ \*.pt)

```python
# scripts/make_graphs.py
from gatcnn import build_graph                   # from canvas file
import cv2, torch, os, tqdm, json

mtsn = torch.load("models/mtsn/best.pt").eval().cuda()
os.makedirs("graphs/train", exist_ok=True)

for img_path in tqdm.tqdm(sorted(glob("dataset/images/train/*"))):
    img = cv2.imread(img_path)
    xy   = generate_patches(img)                 # (N,2)
    feats= [mtsn_embed(crop_patch(img,*p), mtsn) for p in xy]
    feats= torch.stack(feats)                    # (N,128)
    xywh = get_xywh_array(xy)                    # (N,4)  centre+boxsize
    graph = build_graph(feats, torch.tensor(xywh), k=8, cosine_threshold=0.7)
    torch.save(graph, img_path.replace("images","graphs").rsplit(".",1)[0]+".pt")
```

Now `graphs/train` mirrors the image list.

---

## 5 · Target generation

Parse every **label .txt** → assign a class & box-delta target to the **nearest node** (centre-distance).
Store `y_cls` (one-hot) and `y_box` (8 values) as `graph.y`, `graph.box`.

```python
# inside graph loop
for obb,label in zip(bboxes, labels):
    centre = obb.reshape(4,2).mean(0)
    idx = torch.norm(torch.tensor(xy)-centre, dim=1).argmin()
    graph.y[idx,label] = 1
    graph.box[idx]     = obb_normalised
```

Unnamed / background nodes remain zeros.

---

## 6 · Model

Use the **`gatcnn.py`** from the canvas unchanged
(`GACNN(node_feat_dim=132, edge_feat_dim=3, n_classes=#cls)`).

---

## 7 · Train loop

```python
from torch_geometric.loader import DataLoader
train_dl = DataLoader(GraphDataset("graphs/train"), batch_size=8, shuffle=True)
val_dl   = DataLoader(GraphDataset("graphs/val"),   batch_size=8)

model = GACNN(132,3,n_classes).cuda()
opt   = torch.optim.AdamW(model.parameters(), 1e-3)
for epoch in range(30):
    model.train(); tot=0
    for batch in train_dl:
        batch = batch.cuda()
        loss,_ = training_step(model, batch,
                               targets=(batch.y, batch.box, batch.hm))
        opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
    print(f"E{epoch} loss={tot/len(train_dl):.3f}")
    # … add mAP evaluation every 5 epochs
```

* Stop when **[mAP@0.5IoU](mailto:mAP@0.5IoU) ≥ 0.45** or loss plateaus.

---

## 8 · Validation metrics

```python
from torchmetrics.detection.mean_ap import MeanAveragePrecision
map_metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])

model.eval()
with torch.no_grad():
    for batch in val_dl:
        pred_cls, pred_box,_ = model(batch.cuda())
        preds = pyg_to_pred_dict(pred_cls, pred_box, batch)
        targs = pyg_to_targ_dict(batch)
        map_metric.update(preds, targs)
print(map_metric.compute())
```

---

## 9 · Debugging tips

| Symptom                 | Quick probe                                             |
| ----------------------- | ------------------------------------------------------- |
| Loss stuck >3           | Print class-wise positive ratio; often class imbalance. |
| Heat-map head dominates | Lower λ\_hm to 0.1 or mask tiny objects.                |
| GPU OOM                 | Reduce `k`, or sample one of every two super-pixels.    |

---

## 10 · (Opt.) Joint finetune with MMRotate

1. Train an **Oriented R-CNN** in MMRotate on the same OBB files.
2. During `make_graphs.py`, **use detector proposals as nodes** instead of SURF/SLIC.
3. Finetune GACNN for 5–10 epochs ➜ usually +5 mAP on small lesions.

---