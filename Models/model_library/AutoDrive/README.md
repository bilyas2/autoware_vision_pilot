## AutoDrive — end-to-end longitudinal + lateral cues

AutoDrive is a compact end-to-end network for **driver-assistance style** outputs from a front camera: it predicts **distance to the closest in-path object (CIPO)**, **road curvature** (related to steering demand), and a **binary CIPO presence** signal. It is designed to sit alongside classical pipelines (for example AutoSpeed + homography-based distance) and to consume the same style of **wide front view** inputs after a fixed geometric crop.

The network shares its backbone architecture with the **AutoSpeed** detector (same width/depth variant) and adds a small **temporal head** that fuses **previous** and **current** frame features. Training typically **warm-starts the backbone** from a trained AutoSpeed checkpoint, then learns the head (and optionally the full stack) on labelled sequences.

### What AutoDrive outputs (inference)

- **Normalized distance** `d_norm` in `[0, 1]`  
  Mapped to meters as:  
  `d = 150 × (1 - d_norm)`  
  *(capped at 150 m)*

- **Curvature** in `1/m`  
  *(Scaled internally during training; see `CURV_SCALE` in `Models/data_utils/load_data_auto_drive.py`)*

- **CIPO logit**  
  Converted to probability using the sigmoid function  
  A common validation threshold is **0.65** for class 1

Input resolution after the standard ZOD / training preprocessing is **1024 × 512** (2:1), RGB, with **ImageNet** mean/std normalisation.

### Demo / explainer

A packaged demo video link can be added here when the public release is ready. Until then, use the [tutorial](tutorial.ipynb) and the inference scripts under `Models/inference/`.

## Get started

Follow the steps in **[tutorial.ipynb](tutorial.ipynb)**. Use the same **2:1** aspect ratio as training (1024 × 512) for best results.

### Where the code lives

| Topic | Location |
|--------|-----------|
| Network definition | `Models/model_components/autodrive/autodrive_network.py` |
| Data + crop / labels | `Models/data_utils/load_data_auto_drive.py` |
| Training entry | `Models/training/train_auto_drive.py` |
| Example inference (video) | `Models/inference/autodrive_curvature_video.py` |
| Benchmark vs classical | `Models/inference/classical_vs_autodrive_benchmark.py` |
| Side-by-side comparison video | `Models/inference/comparison_video.py` |

### Performance and datasets

Training and evaluation are tied to your **ZOD** (or compatible) label layout: curvature, `distance_to_in_path_object`, and `cipo_detected` per frame. Report numbers in your release notes when you freeze a checkpoint; this README intentionally stays version-agnostic.

## Model weights (release placeholder)

**Planned public artifact:** `AutoDrive.pt` (or `AutoDrive.pth`) — single-file weights aligned with the release tag.

Until the official link is published:

- Use checkpoints produced by this repo, for example  
  `{zod_root}/training/autodrive/run002/checkpoints/AutoDrive_best.pth`
- The training script saves a dict with a `"model"` key containing `state_dict`-compatible weights.

When you publish the final file, update this section with:

- **PyTorch** — link to `AutoDrive.pt`
- **Optional ONNX / TensorRT** — add links here if you export them

## Model variant

There is a single **AutoDrive** architecture in-tree: **1024 × 512** RGB, **two-frame** input (previous + current), backbone compatible with **AutoSpeed** `.pt` for weight transfer.
