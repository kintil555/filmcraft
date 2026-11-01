//! Planar tracking: follows a flat surface (a wall, a screen, a sign) as a **homography**, so the
//! perspective change of the plane is kept, not only its shift, turn and size.
//!
//! Same building blocks as the similarity tracker in the parent module (Shi–Tomasi features,
//! forward–backward checked pyramidal Lucas–Kanade, drift correction against an anchor frame,
//! coasting through short losses), with the fit replaced by RANSAC over minimal four-point
//! samples and a normalised-DLT least-squares refit on the inliers ([`Homography::fit`]).
//! Plausibility is judged on the region itself: the warped polygon must keep its orientation and
//! its area within 0.5×–2× per frame, which rejects flips and runaway fits.

use super::{ANCHOR_FB_MAX, FB_MAX, MAX_COAST, MAX_FEATURES, Prepared, Rng, features, lk, lk_from};
use filmcraft_geom::{Homography, Vec2};
use rayon::prelude::*;

/// Inlier threshold of the homography fit, in pixels (looser than the similarity fit: a plane in
/// perspective is not perfectly modelled by sparse, sub-pixel features).
const THRESHOLD: f64 = 1.5;
const ITERS: usize = 400;
/// Fewest inliers a frame-to-frame fit needs.
const MIN_INLIERS: usize = 8;
/// Per-frame decay of the extrapolated motion while coasting.
const COAST_DAMPING: f64 = 0.85;

/// One planar tracking step's result.
#[derive(Clone, Debug)]
pub struct PlanarStep {
    /// Maps the previous frame's pixels to the next frame's.
    pub homography: Homography,
    pub features: usize,
    pub tracked: usize,
    pub inliers: usize,
    /// The motion was extrapolated (the frame-to-frame match failed).
    pub coasting: bool,
    /// The motion was corrected against the anchor frame (drift removed).
    pub anchored: bool,
}

/// Signed area of a polygon (shoelace).
fn area(poly: &[Vec2]) -> f64 {
    let n = poly.len();
    (0..n).map(|i| poly.get(i).zip(poly.get((i + 1) % n)).map_or(0.0, |(a, b)| a.x * b.y - b.x * a.y)).sum::<f64>() / 2.0
}

/// `region` warped by `h`; `None` when any point is lost.
fn warp_all(h: &Homography, region: &[Vec2]) -> Option<Vec<Vec2>> {
    region.iter().map(|p| h.apply(*p)).collect()
}

/// A per-frame motion that keeps the region's orientation and 0.5×–2× of its area.
fn plausible(h: &Homography, region: &[Vec2]) -> bool {
    let Some(w) = warp_all(h, region) else { return false };
    let (a0, a1) = (area(region), area(&w));
    a0.abs() > 1e-9 && a1.is_finite() && a1 * a0 > 0.0 && (0.5..2.0).contains(&(a1 / a0))
}

/// `h` with its motion scaled by `k` (0 = identity, 1 = unchanged).
fn damp(h: &Homography, k: f64) -> Homography {
    let n = h.normalized().0;
    let id = Homography::IDENTITY.0;
    Homography(std::array::from_fn(|i| id[i] + (n[i] - id[i]) * k))
}

/// RANSAC (four-point samples) + least-squares refit. Returns (homography, inlier count).
pub fn fit_robust(src: &[Vec2], dst: &[Vec2]) -> Option<(Homography, usize)> {
    let n = src.len();
    if n < 4 || dst.len() != n {
        return None;
    }
    let inliers_of = |h: &Homography| -> Vec<usize> {
        (0..n).filter(|&i| matches!((src.get(i).and_then(|p| h.apply(*p)), dst.get(i)), (Some(q), Some(d)) if (q - *d).length() < THRESHOLD)).collect()
    };
    let pick = |idx: &[usize]| -> (Vec<Vec2>, Vec<Vec2>) { idx.iter().filter_map(|&i| src.get(i).zip(dst.get(i))).map(|(s, d)| (*s, *d)).unzip() };
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ n as u64);
    let mut best: Vec<usize> = Vec::new();
    for _ in 0..ITERS {
        let mut idx: Vec<usize> = Vec::with_capacity(4);
        let mut guard = 0;
        while idx.len() < 4 && guard < 64 {
            guard += 1;
            let i = rng.below(n);
            let far = idx.iter().all(|&j| src.get(i).zip(src.get(j)).is_some_and(|(a, b)| (*a - *b).length() >= 4.0));
            if far {
                idx.push(i);
            }
        }
        if idx.len() < 4 {
            continue;
        }
        let (s, d) = pick(&idx);
        let Some(h) = Homography::fit(&s, &d) else { continue };
        let inl = inliers_of(&h);
        if inl.len() > best.len() {
            best = inl;
            if best.len() == n {
                break;
            }
        }
    }
    if best.len() < 4 {
        return None;
    }
    for _ in 0..2 {
        let (s, d) = pick(&best);
        let h = Homography::fit(&s, &d)?;
        let inl = inliers_of(&h);
        if inl.len() < best.len() {
            break;
        }
        best = inl;
    }
    let (s, d) = pick(&best);
    Homography::fit(&s, &d).map(|h| (h, best.len()))
}

/// Frame-to-frame planar step from `a` to `b` over the picture inside `region`.
fn step(a: &Prepared, b: &Prepared, region: &[Vec2]) -> Option<(Homography, usize, usize, usize)> {
    let feats = features(a.base(), region, MAX_FEATURES, 5.0);
    if feats.len() < MIN_INLIERS {
        return None;
    }
    let pairs: Vec<(Vec2, Vec2)> = feats
        .par_iter()
        .filter_map(|&p| {
            let q = lk(&a.levels, &b.levels, p)?;
            let back = lk(&b.levels, &a.levels, q)?;
            let fb = ((back[0] - p[0]).powi(2) + (back[1] - p[1]).powi(2)).sqrt();
            (fb < FB_MAX).then(|| (Vec2::new(p[0] as f64, p[1] as f64), Vec2::new(q[0] as f64, q[1] as f64)))
        })
        .collect();
    let (src, dst): (Vec<Vec2>, Vec<Vec2>) = pairs.into_iter().unzip();
    let (h, inliers) = fit_robust(&src, &dst)?;
    (inliers >= MIN_INLIERS).then_some((h, feats.len(), src.len(), inliers))
}

/// Stateful planar tracker: [`step`] frame to frame with anchor drift correction and coasting
/// through short losses (see the similarity `Tracker` in the parent module).
pub struct PlanarTracker {
    prev: Prepared,
    anchor: Prepared,
    anchor_feats: Vec<[f32; 2]>,
    /// Maps anchor pixels to `prev` pixels.
    anchor_to_prev: Homography,
    velocity: Homography,
    coasted: usize,
}

impl PlanarTracker {
    /// Start on `first`, with `region` (polygon in `first`'s pixels) as the plane.
    pub fn new(first: Prepared, region: &[Vec2]) -> Self {
        let anchor_feats = features(first.base(), region, MAX_FEATURES, 5.0);
        Self { anchor: first.clone(), prev: first, anchor_feats, anchor_to_prev: Homography::IDENTITY, velocity: Homography::IDENTITY, coasted: 0 }
    }

    /// Advance to `next`; `region` is the plane's polygon in the previous frame's pixels. The
    /// returned homography maps previous-frame pixels to `next`'s. `None` once the track is lost.
    pub fn advance(&mut self, next: Prepared, region: &[Vec2]) -> Option<PlanarStep> {
        let ff = step(&self.prev, &next, region).filter(|(h, ..)| plausible(h, region));
        let (mut st, coasting) = match ff {
            Some((h, n_feat, tracked, inliers)) => {
                self.velocity = h;
                self.coasted = 0;
                (PlanarStep { homography: h, features: n_feat, tracked, inliers, coasting: false, anchored: false }, false)
            }
            None => {
                self.coasted += 1;
                if self.coasted > MAX_COAST {
                    return None;
                }
                self.velocity = damp(&self.velocity, COAST_DAMPING);
                (PlanarStep { homography: self.velocity, features: 0, tracked: 0, inliers: 0, coasting: true, anchored: false }, true)
            }
        };
        let pred = st.homography.then_apply(&self.anchor_to_prev);
        let mut anchor_to_next = pred;
        let mut renew = !coasting;
        if let Some((t_abs, inl)) = self.register_anchor(&next, &pred) {
            let refined = self.anchor_to_prev.inverse().map(|inv| t_abs.then_apply(&inv)).filter(|r| plausible(r, region));
            if let (Some(r), true) = (refined, inl >= (self.anchor_feats.len() / 3).max(MIN_INLIERS)) {
                st = PlanarStep { homography: r, inliers: inl, anchored: true, ..st };
                anchor_to_next = t_abs;
                renew = inl * 2 < self.anchor_feats.len();
                if coasting {
                    // the anchor re-acquired the plane
                    self.coasted = 0;
                    self.velocity = r;
                    st.coasting = false;
                }
            }
        }
        let region_next = warp_all(&st.homography, region)?;
        self.anchor_to_prev = anchor_to_next;
        self.prev = next;
        if renew {
            let feats = features(self.prev.base(), &region_next, MAX_FEATURES, 5.0);
            if feats.len() >= MIN_INLIERS {
                self.anchor = self.prev.clone();
                self.anchor_feats = feats;
                self.anchor_to_prev = Homography::IDENTITY;
            }
        }
        Some(st)
    }

    /// Track the anchor features straight to `next`, starting from `pred` (anchor → next).
    fn register_anchor(&self, next: &Prepared, pred: &Homography) -> Option<(Homography, usize)> {
        if self.anchor_feats.len() < MIN_INLIERS {
            return None;
        }
        let pairs: Vec<(Vec2, Vec2)> = self
            .anchor_feats
            .par_iter()
            .filter_map(|&p| {
                let g = pred.apply(Vec2::new(p[0] as f64, p[1] as f64))?;
                let init = [g.x as f32 - p[0], g.y as f32 - p[1]];
                let q = lk_from(&self.anchor.levels, &next.levels, p, init)?;
                let back = lk_from(&next.levels, &self.anchor.levels, q, [p[0] - q[0], p[1] - q[1]])?;
                let fb = ((back[0] - p[0]).powi(2) + (back[1] - p[1]).powi(2)).sqrt();
                (fb < ANCHOR_FB_MAX).then(|| (Vec2::new(p[0] as f64, p[1] as f64), Vec2::new(q[0] as f64, q[1] as f64)))
            })
            .collect();
        let (src, dst): (Vec<Vec2>, Vec<Vec2>) = pairs.into_iter().unzip();
        fit_robust(&src, &dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::track::Gray;

    /// A deterministic texture: smooth blobs + fine detail.
    fn texture(u: f64, v: f64) -> f32 {
        let s = (u * 0.21).sin() * (v * 0.17).cos() + 0.5 * ((u + v) * 0.43).sin() + 0.35 * ((u * 0.9 - v * 0.6).sin() * (v * 0.75).cos());
        let (iu, iv) = ((u / 6.0).floor() as i64, (v / 6.0).floor() as i64);
        let mut x = (iu.wrapping_mul(73_856_093) ^ iv.wrapping_mul(19_349_663)) as u64;
        x ^= x >> 13;
        x = x.wrapping_mul(0x5bd1_e995);
        let h = (x >> 40) as f64 / (1u64 << 24) as f64;
        (0.5 + 0.22 * s + 0.12 * h).clamp(0.0, 1.0) as f32
    }

    const PLANE: (f64, f64, f64, f64) = (60.0, 40.0, 260.0, 200.0);

    /// The textured plane (a rectangle in plane coordinates) seen through `pose`, over a static
    /// differently-textured background.
    fn frame(w: usize, h: usize, pose: &Homography) -> Gray {
        let inv = pose.inverse().expect("pose invertible");
        let px = (0..w * h)
            .map(|i| {
                let p = Vec2::new((i % w) as f64, (i / w) as f64);
                match inv.apply(p) {
                    Some(o) if o.x >= PLANE.0 && o.x <= PLANE.2 && o.y >= PLANE.1 && o.y <= PLANE.3 => texture(o.x, o.y),
                    _ => 0.25 + 0.1 * texture(p.x * 0.5 + 900.0, p.y * 0.5),
                }
            })
            .collect();
        Gray::new(w, h, px)
    }

    fn pose(t: f64) -> Homography {
        Homography([1.0 + 0.004 * t, 0.006 * t, 2.5 * t, -0.004 * t, 1.0 - 0.002 * t, 1.2 * t, 0.00002 * t, -0.00001 * t, 1.0])
    }

    fn quad() -> Vec<Vec2> {
        [(90.0, 70.0), (230.0, 70.0), (230.0, 170.0), (90.0, 170.0)].iter().map(|&(x, y)| Vec2::new(x, y)).collect()
    }

    #[test]
    fn tracks_a_plane_through_perspective_change() {
        let (w, h) = (320, 240);
        let mut region = quad();
        let truth = |k: f64| -> Vec<Vec2> { quad().iter().filter_map(|p| pose(k).apply(*p)).collect() };
        let mut tracker = PlanarTracker::new(Prepared::new(frame(w, h, &pose(0.0))), &region);
        let mut worst = 0.0f64;
        for k in 1..=8 {
            let st = tracker.advance(Prepared::new(frame(w, h, &pose(k as f64))), &region).expect("tracked");
            region = warp_all(&st.homography, &region).expect("warp");
            for (a, b) in region.iter().zip(truth(k as f64)) {
                worst = worst.max((*a - b).length());
            }
        }
        assert!(worst < 2.0, "worst corner error {worst} px");
    }

    #[test]
    fn robust_fit_rejects_outliers() {
        let t = pose(5.0);
        let (mut src, mut dst) = (Vec::new(), Vec::new());
        for i in 0..40 {
            let p = Vec2::new(40.0 + (i * 37 % 200) as f64, 30.0 + (i * 61 % 150) as f64);
            src.push(p);
            let q = t.apply(p).expect("q");
            dst.push(if i % 4 == 0 { q + Vec2::new(25.0, -17.0) } else { q });
        }
        let (h, inl) = fit_robust(&src, &dst).expect("fit");
        assert_eq!(inl, 30);
        for p in &src {
            assert!((h.apply(*p).expect("a") - t.apply(*p).expect("b")).length() < 1e-5);
        }
    }

    #[test]
    fn rejects_flips_and_runaway_areas() {
        let r = quad();
        assert!(plausible(&Homography::IDENTITY, &r));
        let flip = Homography([-1.0, 0.0, 400.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(!plausible(&flip, &r));
        let big = Homography([3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(!plausible(&big, &r));
    }

    #[test]
    fn featureless_frames_end_in_a_lost_track() {
        let (w, h) = (320, 240);
        let region = quad();
        let mut tracker = PlanarTracker::new(Prepared::new(frame(w, h, &pose(0.0))), &region);
        let mut lost = false;
        for _ in 0..=MAX_COAST + 1 {
            if tracker.advance(Prepared::new(Gray::new(w, h, vec![0.5; w * h])), &region).is_none() {
                lost = true;
                break;
            }
        }
        assert!(lost);
    }
}
