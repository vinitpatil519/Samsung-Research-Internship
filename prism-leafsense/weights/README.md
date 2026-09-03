# Weights

Checkpoints are not committed (they are large binaries). Regenerate them with the training
scripts in `backend/training/`:

| File | Produced by |
| --- | --- |
| `unetpp_leaf.pt` | `python train_unet.py --epochs 3 --per-class 50 --size 160` |
| `efficientnet_plantvillage.pt` | `python train_classifier.py --epochs 4 --per-class 260 --size 160` |
| `sparse_encoder.pt` | `python train_encoder.py --per-class 40 --epochs 80` |
| `feature_cache.npz` | side output of `train_encoder.py` |
| `tsne_reference.npz` | `python build_tsne.py` |

The backend loads whatever is present and reports the rest as missing through `GET /health`.
