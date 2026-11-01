//! Depth Anything V2 Small on the CPU (candle). Architecture: L. Yang et al., "Depth Anything V2"
//! (2024), a DINOv2 ViT-S/14 backbone with a DPT head; both Apache-2.0.

use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::depth_anything_v2::{DepthAnythingV2, DepthAnythingV2Config};
use candle_transformers::models::dinov2;

use crate::{DepthError, DepthMap, Estimator};

/// The DINOv2 ViT-S/14 backbone weights (safetensors) expected in the model folder.
pub const DINO_FILE: &str = "dinov2_vits14.safetensors";
/// The Depth Anything V2 Small head weights (safetensors) expected in the model folder.
pub const HEAD_FILE: &str = "depth_anything_v2_vits.safetensors";

/// Side of the square picture the network works on (a multiple of the 14 px patch).
const SIZE: usize = 518;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

fn model_err(e: impl std::fmt::Display) -> DepthError {
    DepthError::Model(e.to_string())
}

/// Depth Anything V2 Small, loaded from a folder holding [`DINO_FILE`] and [`HEAD_FILE`].
pub struct DepthAnything {
    model: DepthAnythingV2,
    device: Device,
}

impl DepthAnything {
    /// Load the model from `dir`. Reads the files into memory (no memory mapping).
    pub fn load(dir: &Path) -> Result<Self, DepthError> {
        let device = Device::Cpu;
        let read = |name: &str| {
            let path = dir.join(name);
            std::fs::read(&path).map_err(|e| DepthError::Model(format!("{}: {e}", path.display())))
        };
        let vb = VarBuilder::from_buffered_safetensors(read(DINO_FILE)?, DType::F32, &device).map_err(model_err)?;
        let backbone = dinov2::vit_small(vb).map_err(model_err)?;
        let vb = VarBuilder::from_buffered_safetensors(read(HEAD_FILE)?, DType::F32, &device).map_err(model_err)?;
        let model = DepthAnythingV2::new(Arc::new(backbone), DepthAnythingV2Config::vit_small(), vb).map_err(model_err)?;
        Ok(Self { model, device })
    }
}

/// `rgba` resampled (bilinear) to a normalised `3 × SIZE × SIZE` planar float picture.
fn prepare(rgba: &[u8], w: usize, h: usize) -> Vec<f32> {
    let plane = SIZE * SIZE;
    let mut chw = vec![0.0f32; 3 * plane];
    let (kx, ky) = (w as f64 / SIZE as f64, h as f64 / SIZE as f64);
    for y in 0..SIZE {
        let sy = ((y as f64 + 0.5) * ky - 0.5).clamp(0.0, (h - 1) as f64);
        let (y0, fy) = (sy.floor() as usize, sy - sy.floor());
        let y1 = (y0 + 1).min(h - 1);
        for x in 0..SIZE {
            let sx = ((x as f64 + 0.5) * kx - 0.5).clamp(0.0, (w - 1) as f64);
            let (x0, fx) = (sx.floor() as usize, sx - sx.floor());
            let x1 = (x0 + 1).min(w - 1);
            for (c, (mean, std)) in MEAN.iter().zip(STD.iter()).enumerate() {
                let px = |xx: usize, yy: usize| rgba.get((yy * w + xx) * 4 + c).copied().unwrap_or(0) as f64 / 255.0;
                let top = px(x0, y0) + (px(x1, y0) - px(x0, y0)) * fx;
                let bot = px(x0, y1) + (px(x1, y1) - px(x0, y1)) * fx;
                let v = top + (bot - top) * fy;
                if let Some(slot) = chw.get_mut(c * plane + y * SIZE + x) {
                    *slot = ((v as f32) - mean) / std;
                }
            }
        }
    }
    chw
}

impl Estimator for DepthAnything {
    fn estimate(&self, rgba: &[u8], w: usize, h: usize) -> Result<DepthMap, DepthError> {
        if w == 0 || h == 0 || w.checked_mul(h).and_then(|n| n.checked_mul(4)) != Some(rgba.len()) {
            return Err(DepthError::Input(format!("{w}×{h} picture over {} bytes", rgba.len())));
        }
        let input = Tensor::from_vec(prepare(rgba, w, h), (1, 3, SIZE, SIZE), &self.device).map_err(model_err)?;
        let out = self.model.forward(&input).map_err(model_err)?;
        let flat = out.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(model_err)?;
        if flat.len() != SIZE * SIZE {
            return Err(DepthError::Model(format!("unexpected output of {} values", flat.len())));
        }
        DepthMap::new(SIZE, SIZE, flat)?.normalized().resized(w, h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_model_folder_is_an_error_not_a_panic() {
        assert!(DepthAnything::load(Path::new("/nonexistent/filmcraft-depth")).is_err());
    }

    #[test]
    fn preparing_a_picture_fills_all_three_planes() {
        let rgba = vec![128u8; 5 * 3 * 4];
        let chw = prepare(&rgba, 5, 3);
        assert_eq!(chw.len(), 3 * SIZE * SIZE);
        assert!(chw.iter().all(|v| v.is_finite()));
        let expect = (128.0 / 255.0 - MEAN[0]) / STD[0];
        assert!((chw[0] - expect).abs() < 1e-4);
    }
}
