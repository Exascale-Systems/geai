# Data and Model File Loading Paths

This document describes where every dataset, model, and result file is loaded from during the training and evaluation pipeline.

## Overview

The system has three main file types:
1. **Data files** - Gravity measurements and density models (HDF5 format)
2. **Model checkpoints** - Trained neural network weights (PyTorch format)
3. **Train/val splits** - Dataset partition indices (NumPy format)

---

## Data Files (HDF5)

### Data Generation
**File Written:** `data/single_block_v2.h5` (492 MB, 20,000 samples)
- **Source:** `src/gen/batch.py:20`
- **Function:** `generate_batch(out_path="data/single_block_v2.h5")`
- **When:** Run `python3 src/gen/batch.py` to generate
- **Contains:** 
  - Gravity measurements (gx, gy, gz components)
  - True density models
  - Receiver locations
  - Active cell masks
  - Global mesh metadata (shape, spacing)

### Data Loading (Training)
**File Read:** `data/{ds_name}.h5`
- **Main function:** `src/data/dataset.py:173-174`
  - `stats = compute_stats(f"data/{ds_name}.h5")`
  - `ds = MasterDataset(f"data/{ds_name}.h5", components=components)`
- **Called from:** `src/train.py:29` via `data_prep(ds_name="single_block_v2", ...)`
- **Key class:** `MasterDataset` in `src/data/dataset.py`
- **Component extraction:** `src/data/dataset.py:63-75` extracts requested gravity components

### Data Loading (Evaluation)
**File Read:** Same as training - `data/single_block_v2.h5`
- **Called from:** `src/training/evaluation.py` via `data_prep()`
- **Used for:** Validating neural network predictions against true gravity data

---

## Train/Val Splits (NumPy)

### Splits File Generated
**File Written:** `splits/{split_name}.npz`
- **Source:** `src/data/dataset.py:194`
- **When:** First time running training if file doesn't exist
- **Contains:** 
  - `tr`: Training dataset indices (80% of samples)
  - `va`: Validation dataset indices (20% of samples)

### Splits File Loaded
**File Read:** `splits/{split_name}.npz`
- **Source:** `src/data/dataset.py:182-183`
  ```python
  if load_splits and Path(f"splits/{split_name}.npz").exists():
      splits = np.load(f"splits/{split_name}.npz")
      tr_indices, va_indices = splits["tr"], splits["va"]
  ```
- **Split name:** Defaults to `"single_block_v2"` in `src/train.py:24`
- **Purpose:** Ensures train/val split is consistent across runs

---

## Model Checkpoints (PyTorch)

### Models Saved During Training
**Files Written:**
1. `checkpoints/best.pt` - Best model on validation set
   - **Source:** `src/training/engine.py:106`
   - **When:** When validation loss improves
   
2. `checkpoints/final.pt` - Final model after all epochs
   - **Source:** `src/training/engine.py:113`
   - **When:** After training completes

### Model Loaded During Evaluation
**File Read:** `checkpoints/final.pt`
- **Source:** `src/training/evaluation.py:32-33`
  ```python
  if Path(f"checkpoints/final.pt").exists():
      state = torch.load(f"checkpoints/final.pt", map_location=device)
      model.load_state_dict(state.get("model", state))
  ```
- **Called from:** `src/training/evaluation.py:563-564` via `load_model(model_name="single_block_500")`
- **Network:** `GravInvNet` from `src/modeling/networks.py`
- **Input channels:** Determined by `len(components)` (default: 3)

**Note:** If `checkpoints/final.pt` doesn't exist, the network loads with random weights (untrained).

---

## Complete Pipeline

### Training Pipeline
```
src/train.py
  ↓
src/data/dataset.py:data_prep()
  ├─ Read: data/single_block_v2.h5
  ├─ Read/Create: splits/single_block_v2.npz
  └─ Returns: train_loader, val_loader, stats
  ↓
src/modeling/networks.py:GravInvNet() [created with random weights]
  ↓
src/training/engine.py:train_model()
  ↓
Write: checkpoints/best.pt [if val loss improves]
Write: checkpoints/final.pt [at end of training]
```

### Evaluation Pipeline
```
src/eval.py
  ↓
src/training/evaluation.py:_eval()
  ├─ Read: checkpoints/final.pt
  ├─ Read: data/single_block_v2.h5 [for val set]
  ├─ Read: splits/single_block_v2.npz [for val indices]
  └─ Predicts and visualizes results
```

---

## File Locations Summary

| Purpose | File Path | Format | When Created/Used |
|---------|-----------|--------|-------------------|
| Training data | `data/single_block_v2.h5` | HDF5 | Gen with `src/gen/batch.py` |
| Train/val split | `splits/single_block_v2.npz` | NumPy | Auto-created first training run |
| Best model | `checkpoints/best.pt` | PyTorch | During training (if val improves) |
| Final model | `checkpoints/final.pt` | PyTorch | End of training |
| Simulation code | `src/gen/simulation/` | Python | Data generation |
| Data loading | `src/data/dataset.py` | Python | Training & evaluation |
| Network | `src/modeling/networks.py` | Python | Always loaded |
| Training loop | `src/training/engine.py` | Python | Training |
| Evaluation | `src/training/evaluation.py` | Python | Evaluation |

---

## Key Variables

**Training (`src/train.py`)**
- `ds_name`: `"single_block_v2"` (dataset name for `data/{ds_name}.h5`)
- `split_name`: `"single_block_v2"` (split name for `splits/{split_name}.npz`)
- `components`: `("gx", "gy", "gz")` (determines input channels)

**Evaluation (`src/eval.py`)**
- `components`: Must match training components
- Model loaded from `checkpoints/final.pt`

**Data Preparation (`src/data/dataset.py`)**
- `stats`: Computed from data file for normalization
- `component_indices`: Maps component names to indices in interleaved data
