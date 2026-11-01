//! [`DepthMap`]: relative depth per pixel.

use crate::DepthError;

/// Most pixels a map may hold (caps allocations sized by input).
const MAX_PIXELS: usize = 1 << 28;

/// Relative depth per pixel, row-major. Larger = nearer (inverse depth, the output convention of
/// monocular depth models); the scale is relative, not metric.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthMap {
    w: usize,
    h: usize,
    data: Vec<f32>,
}

impl DepthMap {
    /// A `w`×`h` map over `data` (which must hold exactly `w * h` values).
    pub fn new(w: usize, h: usize, data: Vec<f32>) -> Result<Self, DepthError> {
        match w.checked_mul(h) {
            Some(n) if n > 0 && n <= MAX_PIXELS && n == data.len() => Ok(Self { w, h, data }),
            _ => Err(DepthError::Input(format!("{w}×{h} map over {} values", data.len()))),
        }
    }

    pub fn width(&self) -> usize {
        self.w
    }

    pub fn height(&self) -> usize {
        self.h
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// The value at integer pixel `(x, y)`, clamped to the map.
    fn at(&self, x: i64, y: i64) -> f32 {
        let x = x.clamp(0, self.w as i64 - 1) as usize;
        let y = y.clamp(0, self.h as i64 - 1) as usize;
        self.data.get(y * self.w + x).copied().unwrap_or(0.0)
    }

    /// Bilinear sample at continuous pixel coordinates (pixel centres at integers), clamped to
    /// the map. A non-finite coordinate reads 0.
    pub fn sample(&self, x: f64, y: f64) -> f32 {
        if !x.is_finite() || !y.is_finite() {
            return 0.0;
        }
        let (x, y) = (x.clamp(-1.0e9, 1.0e9), y.clamp(-1.0e9, 1.0e9));
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = ((x - x0) as f32, (y - y0) as f32);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let top = self.at(x0, y0) + (self.at(x0 + 1, y0) - self.at(x0, y0)) * fx;
        let bot = self.at(x0, y0 + 1) + (self.at(x0 + 1, y0 + 1) - self.at(x0, y0 + 1)) * fx;
        top + (bot - top) * fy
    }

    /// This map resampled (bilinear) to `w`×`h`.
    pub fn resized(&self, w: usize, h: usize) -> Result<DepthMap, DepthError> {
        let n = w.checked_mul(h).filter(|n| *n > 0 && *n <= MAX_PIXELS).ok_or_else(|| DepthError::Input(format!("resize to {w}×{h}")))?;
        let (kx, ky) = (self.w as f64 / w as f64, self.h as f64 / h as f64);
        let mut data = Vec::with_capacity(n);
        for y in 0..h {
            for x in 0..w {
                data.push(self.sample((x as f64 + 0.5) * kx - 0.5, (y as f64 + 0.5) * ky - 0.5));
            }
        }
        DepthMap::new(w, h, data)
    }

    /// The map stretched to 0..=1 (a flat map becomes all 0.5; non-finite values become 0).
    pub fn normalized(&self) -> DepthMap {
        let finite = self.data.iter().copied().filter(|v| v.is_finite());
        let (lo, hi) = finite.fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| (lo.min(v), hi.max(v)));
        let span = hi - lo;
        let data = self
            .data
            .iter()
            .map(|&v| match (v.is_finite(), span.is_finite() && span > 1e-12) {
                (true, true) => (v - lo) / span,
                (true, false) => 0.5,
                _ => 0.0,
            })
            .collect();
        DepthMap { w: self.w, h: self.h, data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_sizes() {
        assert!(DepthMap::new(0, 4, vec![]).is_err());
        assert!(DepthMap::new(2, 2, vec![0.0; 3]).is_err());
        assert!(DepthMap::new(usize::MAX, 2, vec![0.0; 4]).is_err());
        assert!(DepthMap::new(2, 2, vec![0.0; 4]).is_ok());
    }

    #[test]
    fn samples_bilinearly_and_clamps() {
        let m = DepthMap::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]).expect("map");
        assert!((m.sample(0.5, 0.0) - 0.5).abs() < 1e-6);
        assert!((m.sample(0.5, 0.5) - 0.5).abs() < 1e-6);
        assert_eq!(m.sample(-50.0, -50.0), 0.0);
        assert_eq!(m.sample(f64::NAN, 1.0), 0.0);
        assert_eq!(m.sample(f64::INFINITY, 0.0), 0.0);
    }

    #[test]
    fn resizes_a_ramp() {
        let m = DepthMap::new(4, 1, vec![0.0, 1.0, 2.0, 3.0]).expect("map");
        let r = m.resized(8, 2).expect("resized");
        assert_eq!((r.width(), r.height()), (8, 2));
        assert!(r.data().windows(2).take(7).all(|p| p[1] >= p[0]));
        assert!(m.resized(0, 3).is_err());
    }

    #[test]
    fn normalizes_to_the_unit_range() {
        let m = DepthMap::new(2, 2, vec![2.0, 4.0, f32::NAN, 6.0]).expect("map");
        let n = m.normalized();
        assert_eq!(n.data(), &[0.0, 0.5, 0.0, 1.0]);
        let flat = DepthMap::new(2, 1, vec![3.0, 3.0]).expect("map").normalized();
        assert_eq!(flat.data(), &[0.5, 0.5]);
    }
}
