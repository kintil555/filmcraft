//! Monocular depth (layer L2, no UI): a [`DepthMap`] (one relative-depth value per pixel, 1 = near,
//! 0 = far) and the [`Estimator`] that produces it from a picture.
//!
//! The planar tracker (`filmcraft_render::track`) uses a depth map to tell the tracked plane from
//! what moves in front of or behind it. With the `depth-anything` feature, [`DepthAnything`] runs
//! Depth Anything V2 Small (Apache-2.0) on the CPU; the weights are not part of the repository
//! (see the crate README for where to get them and where to put them).

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

#[cfg(feature = "depth-anything")]
mod anything;
mod map;

#[cfg(feature = "depth-anything")]
pub use anything::{DINO_FILE, DepthAnything, HEAD_FILE};
pub use map::DepthMap;

/// Why a depth map could not be made.
#[derive(Debug, thiserror::Error)]
pub enum DepthError {
    /// The picture or the map size is unusable (zero, too large, or a buffer of the wrong length).
    #[error("bad depth input: {0}")]
    Input(String),
    /// The depth model could not be loaded or run.
    #[error("depth model: {0}")]
    Model(String),
}

/// Something that estimates relative depth from a picture.
pub trait Estimator {
    /// Depth of the `w`×`h` 8-bit RGBA picture `rgba`, as a map of the same size (1 = near).
    fn estimate(&self, rgba: &[u8], w: usize, h: usize) -> Result<DepthMap, DepthError>;
}
