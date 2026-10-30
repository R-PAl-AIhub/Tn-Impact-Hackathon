# Training

The detector was fine-tuned with Ultralytics on the public [Construction-PPE dataset](https://docs.ultralytics.com/datasets/detect/construction-ppe/).

| | |
|---|---|
| Starting checkpoint | `yolo26n.pt` (Ultralytics YOLO26 nano) |
| Library | Ultralytics 8.4.10, PyTorch 2.10, CUDA 12.8 |
| Hardware | NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB (logged in `train.ipynb`) |
| Dataset | 1,132 train / 143 val / 141 test images, 11 classes |
| Settings | 100 epochs, batch 8, image size 640, default augmentation |
| Training time | 64 minutes (3,825 s) |
| Kept weights | epoch 78, highest val mAP@0.5:0.95 → [`../backend/best.pt`](../backend/best.pt) |

Classes: `helmet` `gloves` `vest` `boots` `goggles` `none` `Person` `no_helmet` `no_goggle` `no_gloves` `no_boots`. The app uses `helmet` (0), `vest` (2), `Person` (6) and `no_helmet` (7).

## Files

| File | Source |
|---|---|
| `train.ipynb` | The training notebook. It contains a shorter 20-epoch run (batch 16) and inference checks. |
| `train_args.yaml` | Exact arguments of the 100-epoch run that produced `best.pt` |
| `results.csv` | Per-epoch losses and metrics of that run |
| `plots/` | Ultralytics plots of that run: PR curve, F1 curve, confusion matrices, training curves |
| `construction-ppe.yaml` | Dataset config |
| `eval_results.json` | `best.pt` re-evaluated on the val and test splits with `model.val()` |

## Results of `best.pt`

| Class | val AP@0.5 | test AP@0.5 | test precision | test recall |
|---|---:|---:|---:|---:|
| Person | 0.901 | 0.799 | 0.792 | 0.807 |
| helmet | 0.784 | 0.922 | 0.838 | 0.911 |
| vest | 0.830 | 0.856 | 0.780 | 0.837 |
| no_helmet | 0.218 | 0.216 | 0.371 | 0.200 |
| **All 11 classes (mAP)** | **0.536** | **0.507** | | |

The val numbers match the training run's own PR curve (`plots/BoxPR_curve.png`). The test split was not used during training.

## Reproduce

```bash
pip install ultralytics==8.4.10
yolo detect train model=yolo26n.pt data=construction-ppe.yaml epochs=100 batch=8 imgsz=640
yolo detect val   model=../backend/best.pt data=construction-ppe.yaml split=test
```

The dataset (178 MB) downloads automatically on first use.
