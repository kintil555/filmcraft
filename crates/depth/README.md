# filmcraft-depth

Monocular depth for FilmCraft (layer L2): a `DepthMap` (relative inverse depth per pixel, 1 = near,
0 = far), the `Estimator` trait, and — with the `depth-anything` feature — **Depth Anything V2
Small** inference on the CPU with [candle](https://github.com/huggingface/candle) (pure Rust).

The planar tracker (`filmcraft_render::track`) uses the map to tell the tracked plane from things in
front of or behind it: the matches' depth is fitted with one plane (inverse depth is affine across
a plane) and matches off it are left out of the homography fit. Tracking Motion ▸ **Depth
Detection** (0–100 %) sets how far off the plane's depth a match may be (45 % of the depth range at
0, 8 % at 100). Without a model, the tracker still rejects off-plane matches from their parallax.

## Enabling it

The weights are **not** in the repository. Build the engine / app with the feature and point the
environment at a folder holding two safetensors files:

```text
cargo run -p filmcraft --features depth-anything
FILMCRAFT_DEPTH_MODEL=/path/to/depth-models   # folder with the two files below
```

| File | What | Where it comes from |
|---|---|---|
| `dinov2_vits14.safetensors` | DINOv2 ViT-S/14 backbone | Hugging Face `lmz/candle-dino-v2` |
| `depth_anything_v2_vits.safetensors` | Depth Anything V2 Small DPT head | Hugging Face `jeroenvlek/depth-anything-v2-safetensors` |

Licences: Depth Anything V2 **Small** and DINOv2 are Apache-2.0 (the larger Depth Anything V2
models are CC BY-NC and must not be used here); check each model card before use. These are the
files the candle `depth_anything_v2` example loads; the file names and repositories could not be
checked from the environment this was written in.

Inference runs once per tracked frame on the CPU at 518 × 518 (about a second a frame on a
desktop CPU, more on slow machines). If the folder is missing or the model fails to load, tracking
carries on without depth and the status line says so.
