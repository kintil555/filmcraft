//! Planar tracking: follows a flat surface (a wall, a screen, a sign) as a **homography**, so the
//! perspective change of the plane is kept, not only its shift, turn and size.
//!
//! Same building blocks as the similarity tracker in the parent module (Shi–Tomasi features,
//! forward–backward checked pyramidal Lucas–Kanade, drift correction against an anchor frame,
//! coasting through short losses), with the fit replaced by RANSAC over minimal four-point
//! samples and a normalised-DLT least-squares refit on the inliers ([`Homography::fit`]).
//! Plausibility is judged on the region itself: the warped polygon must keep its orientation and
//! its area within 0.5×–2× per frame, which rejects flips and runaway fits.
//!
//! Three things keep the plane honest where the plain fit is not:
//!
//! * **Depth (parallax) detection.** Features that do not move like the plane (a pole in front of
//!   a wall, a person walking past a sign) sit at another depth and show up as parallax: their
//!   residual against the fitted homography. [`PlanarOptions::depth_strength`] sets how strictly
//!   such off-plane features are rejected from the fit (0 = tolerant, 1 = only the plane counts),
//!   and every step reports the per-feature parallax as [`DepthSample`]s.
//! * **Degrees of freedom follow the coverage.** When the visible features cover only a small part
//!   of the region (it is leaving the canvas, or the rest is blank or occluded), a full homography
//!   is unconstrained and its far corners fly off. The fit then blends down to a similarity, and
//!   to a plain shift when the features are even sparser.
//! * **Leaving the canvas is not an error.** A region that is mostly outside the frame coasts on
//!   its (slowly decaying) motion for up to [`MAX_COAST_OFF_CANVAS`] frames and is picked up again
//!   when it comes back, instead of ending the track.

use super::{ANCHOR_FB_MAX, FB_MAX, MAX_COAST, MAX_FEATURES, Prepared, Rng, features, lk, lk_from};
use filmcraft_geom::{Homography, Vec2};
use filmcraft_project::TrackMethod;
use rayon::prelude::*;

/// Inlier threshold of the homography fit, in pixels (looser than the similarity fit: a plane in
/// perspective is not perfectly modelled by sparse, sub-pixel features).
const THRESHOLD: f64 = 1.5;
const ITERS: usize = 400;
/// Fewest inliers a frame-to-frame fit needs.
const MIN_INLIERS: usize = 8;
/// Per-frame decay of the extrapolated motion while coasting.
const COAST_DAMPING: f64 = 0.85;
/// Frames a region that is mostly off the canvas keeps coasting before the track is given up.
pub const MAX_COAST_OFF_CANVAS: usize = 90;
/// Slower decay while off the canvas: a sign sliding out of frame keeps sliding.
const COAST_DAMPING_OFF_CANVAS: f64 = 0.95;
/// Below this visible share of the region's bounding box the region counts as off the canvas.
const OFF_CANVAS_VISIBLE: f64 = 0.5;
/// Inlier threshold (px) at depth strength 0 (tolerant) and 1 (only the plane itself counts).
const THRESHOLD_LOOSE: f64 = 3.0;
const THRESHOLD_TIGHT: f64 = 0.6;
/// Share of the region's bounding box the inliers must span for a full homography, an affine-like
/// similarity, and anything beyond a plain shift.
const COVERAGE_FULL: f64 = 0.35;
const COVERAGE_SIMILARITY: f64 = 0.15;
const COVERAGE_SHIFT: f64 = 0.04;
/// Inliers whose spread has a minor/major variance ratio below this are treated as collinear.
const MIN_SPREAD_RATIO: f64 = 0.01;
/// Residual (px) at which a feature counts as half-way off the plane in [`DepthSample::parallax`].
const PARALLAX_HALF: f64 = 1.0;

/// Tuning of the planar tracker.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanarOptions {
    /// How strictly features at another depth than the plane are rejected, 0..=1.
    pub depth_strength: f64,
}

impl Default for PlanarOptions {
    fn default() -> Self {
        Self { depth_strength: 0.6 }
    }
}

impl PlanarOptions {
    /// Inlier threshold in pixels for this depth strength (a NaN strength reads as the default).
    pub fn threshold(&self) -> f64 {
        let s = if self.depth_strength.is_finite() { self.depth_strength.clamp(0.0, 1.0) } else { Self::default().depth_strength };
        THRESHOLD_LOOSE + (THRESHOLD_TIGHT - THRESHOLD_LOOSE) * s
    }
}

/// Relative depth of one tracked feature: how far it sits off the plane, from its parallax.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthSample {
    /// Where the feature was, in the previous frame's pixels.
    pub pos: Vec2,
    /// 0 = on the plane, towards 1 = far off it (`r / (r + 1 px)` of the fit residual `r`).
    pub parallax: f32,
}

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
    /// The region is mostly outside the canvas (being followed on its extrapolated motion).
    pub off_canvas: bool,
    /// Per-feature parallax against the fitted plane (empty while coasting).
    pub depth: Vec<DepthSample>,
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

/// Bounding box of `pts` (NaN coordinates are ignored).
fn bbox(pts: &[Vec2]) -> Option<(Vec2, Vec2)> {
    let first = *pts.first()?;
    Some(pts.iter().fold((first, first), |(lo, hi), p| (Vec2::new(lo.x.min(p.x), lo.y.min(p.y)), Vec2::new(hi.x.max(p.x), hi.y.max(p.y)))))
}

fn bbox_area((lo, hi): (Vec2, Vec2)) -> f64 {
    (hi.x - lo.x).max(0.0) * (hi.y - lo.y).max(0.0)
}

/// Minor/major variance ratio of the point cloud (0 = collinear or too few points).
fn spread_ratio(pts: &[Vec2]) -> f64 {
    let n = pts.len() as f64;
    if pts.len() < 3 {
        return 0.0;
    }
    let (mx, my) = (pts.iter().map(|p| p.x).sum::<f64>() / n, pts.iter().map(|p| p.y).sum::<f64>() / n);
    let (mut cxx, mut cxy, mut cyy) = (0.0, 0.0, 0.0);
    for p in pts {
        let (dx, dy) = (p.x - mx, p.y - my);
        cxx += dx * dx;
        cxy += dx * dy;
        cyy += dy * dy;
    }
    let half_tr = (cxx + cyy) / 2.0;
    let disc = (((cxx - cyy) / 2.0).powi(2) + cxy * cxy).sqrt();
    let (major, minor) = (half_tr + disc, half_tr - disc);
    if major.is_finite() && major > 1e-12 { (minor / major).max(0.0) } else { 0.0 }
}

/// Share of `region`'s bounding box that the points span (0 when they are nearly collinear).
fn coverage(points: &[Vec2], region: &[Vec2]) -> f64 {
    let (Some(p), Some(r)) = (bbox(points), bbox(region)) else { return 0.0 };
    let region_area = bbox_area(r);
    if !region_area.is_finite() || region_area <= 1e-9 || spread_ratio(points) < MIN_SPREAD_RATIO {
        return 0.0;
    }
    (bbox_area(p) / region_area).clamp(0.0, 1.0)
}

/// Share of `region`'s bounding box inside the `w`×`h` canvas (0 for a degenerate region).
fn visible_fraction(region: &[Vec2], w: usize, h: usize) -> f64 {
    let Some((lo, hi)) = bbox(region) else { return 0.0 };
    let area = bbox_area((lo, hi));
    if !area.is_finite() || area <= 1e-9 {
        return 0.0;
    }
    let clipped = (Vec2::new(lo.x.max(0.0), lo.y.max(0.0)), Vec2::new(hi.x.min(w as f64), hi.y.min(h as f64)));
    (bbox_area(clipped) / area).clamp(0.0, 1.0)
}

/// Element-wise blend of two (normalised) homographies, `t` = 0 → `a`, 1 → `b`.
fn blend(a: &Homography, b: &Homography, t: f64) -> Homography {
    let (a, b) = (a.normalized().0, b.normalized().0);
    Homography(std::array::from_fn(|i| match (a.get(i), b.get(i)) {
        (Some(x), Some(y)) => x + (y - x) * t,
        _ => 0.0,
    }))
}

/// RANSAC (four-point samples) + least-squares refit. Returns (homography, inlier count).
pub fn fit_robust(src: &[Vec2], dst: &[Vec2]) -> Option<(Homography, usize)> {
    fit_robust_with(src, dst, THRESHOLD).map(|(h, inl)| (h, inl.len()))
}

/// [`fit_robust`] with an explicit inlier `threshold` (px); returns the inlier indices.
fn fit_robust_with(src: &[Vec2], dst: &[Vec2], threshold: f64) -> Option<(Homography, Vec<usize>)> {
    let n = src.len();
    if n < 4 || dst.len() != n || !threshold.is_finite() || threshold <= 0.0 {
        return None;
    }
    let inliers_of = |h: &Homography| -> Vec<usize> {
        (0..n).filter(|&i| matches!((src.get(i).and_then(|p| h.apply(*p)), dst.get(i)), (Some(q), Some(d)) if (q - *d).length() < threshold)).collect()
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
    Homography::fit(&s, &d).map(|h| (h, best))
}

/// Motion `src → dst` of the plane `region`, spending only the degrees of freedom the matches pin
/// down: a full homography when the inliers span enough of the region, a similarity (blended in
/// gradually) when they cover less, a plain shift when they are sparser still. Returns the motion
/// and its inlier count.
fn fit_model(src: &[Vec2], dst: &[Vec2], region: &[Vec2], threshold: f64) -> Option<(Homography, usize)> {
    let (h, inl) = fit_robust_with(src, dst, threshold)?;
    let (s, d): (Vec<Vec2>, Vec<Vec2>) = inl.iter().filter_map(|&i| src.get(i).zip(dst.get(i))).map(|(a, b)| (*a, *b)).unzip();
    let cov = coverage(&s, region);
    if cov >= COVERAGE_FULL {
        return Some((h, inl.len()));
    }
    let method = if cov >= COVERAGE_SHIFT { TrackMethod::PositionScaleRotation } else { TrackMethod::Position };
    let Some((affine, _)) = super::fit_robust(method, &s, &d).or_else(|| super::fit_robust(TrackMethod::Position, &s, &d)) else {
        return Some((h, inl.len()));
    };
    let low = Homography::from_affine(&affine);
    let model = if cov >= COVERAGE_SIMILARITY { blend(&low, &h, (cov - COVERAGE_SIMILARITY) / (COVERAGE_FULL - COVERAGE_SIMILARITY)) } else { low };
    Some((model, inl.len()))
}

/// Per-feature parallax of the matches `src → dst` against the plane motion `h`.
fn depth_samples(h: &Homography, src: &[Vec2], dst: &[Vec2]) -> Vec<DepthSample> {
    src.iter()
        .zip(dst)
        .filter_map(|(s, d)| {
            let r = (h.apply(*s)? - *d).length();
            r.is_finite().then_some(DepthSample { pos: *s, parallax: (r / (r + PARALLAX_HALF)) as f32 })
        })
        .collect()
}

/// A frame-to-frame fit.
struct FrameFit {
    h: Homography,
    features: usize,
    tracked: usize,
    inliers: usize,
    depth: Vec<DepthSample>,
}

/// Frame-to-frame planar step from `a` to `b` over the picture inside `region`.
fn step(a: &Prepared, b: &Prepared, region: &[Vec2], threshold: f64) -> Option<FrameFit> {
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
    // a strict depth setting that leaves too few plane features falls back to the tolerant fit
    let (h, inliers) =
        fit_model(&src, &dst, region, threshold).filter(|(_, n)| *n >= MIN_INLIERS).or_else(|| fit_model(&src, &dst, region, THRESHOLD_LOOSE))?;
    (inliers >= MIN_INLIERS).then(|| FrameFit { h, features: feats.len(), tracked: src.len(), inliers, depth: depth_samples(&h, &src, &dst) })
}

/// Stateful planar tracker: [`step`] frame to frame with anchor drift correction and coasting
/// through short losses (see the similarity `Tracker` in the parent module).
pub struct PlanarTracker {
    prev: Prepared,
    anchor: Prepared,
    anchor_feats: Vec<[f32; 2]>,
    /// The plane's polygon in the anchor frame's pixels (what the anchor fit's coverage is judged on).
    anchor_region: Vec<Vec2>,
    /// Maps anchor pixels to `prev` pixels.
    anchor_to_prev: Homography,
    velocity: Homography,
    coasted: usize,
    /// The previous step saw the region mostly off the canvas.
    was_off_canvas: bool,
    /// Inlier threshold (px) from the depth strength.
    threshold: f64,
}

impl PlanarTracker {
    /// Start on `first`, with `region` (polygon in `first`'s pixels) as the plane.
    pub fn new(first: Prepared, region: &[Vec2]) -> Self {
        Self::with_options(first, region, PlanarOptions::default())
    }

    /// [`PlanarTracker::new`] with explicit [`PlanarOptions`].
    pub fn with_options(first: Prepared, region: &[Vec2], options: PlanarOptions) -> Self {
        let anchor_feats = features(first.base(), region, MAX_FEATURES, 5.0);
        Self {
            anchor: first.clone(),
            prev: first,
            anchor_feats,
            anchor_region: region.to_vec(),
            anchor_to_prev: Homography::IDENTITY,
            velocity: Homography::IDENTITY,
            coasted: 0,
            was_off_canvas: false,
            threshold: options.threshold(),
        }
    }

    /// Advance to `next`; `region` is the plane's polygon in the previous frame's pixels. The
    /// returned homography maps previous-frame pixels to `next`'s. `None` once the track is lost.
    pub fn advance(&mut self, next: Prepared, region: &[Vec2]) -> Option<PlanarStep> {
        let (fw, fh) = (next.base().w, next.base().h);
        // a region that is (about to be) mostly outside the canvas has nothing to track: coast
        // on its motion, and take it up again when it comes back
        let predicted = warp_all(&self.velocity, region);
        let off_canvas = visible_fraction(predicted.as_deref().unwrap_or(region), fw, fh) < OFF_CANVAS_VISIBLE;
        let (max_coast, damping) = if off_canvas { (MAX_COAST_OFF_CANVAS, COAST_DAMPING_OFF_CANVAS) } else { (MAX_COAST, COAST_DAMPING) };
        if self.was_off_canvas && !off_canvas {
            // back on the canvas: a few frames to find the plane again
            self.coasted = self.coasted.min(MAX_COAST / 2);
        }
        self.was_off_canvas = off_canvas;
        let ff = step(&self.prev, &next, region, self.threshold).filter(|f| plausible(&f.h, region));
        let (mut st, coasting) = match ff {
            Some(f) => {
                self.velocity = f.h;
                self.coasted = 0;
                (
                    PlanarStep {
                        homography: f.h,
                        features: f.features,
                        tracked: f.tracked,
                        inliers: f.inliers,
                        coasting: false,
                        anchored: false,
                        off_canvas,
                        depth: f.depth,
                    },
                    false,
                )
            }
            None => {
                self.coasted += 1;
                if self.coasted > max_coast {
                    return None;
                }
                self.velocity = damp(&self.velocity, damping);
                (
                    PlanarStep {
                        homography: self.velocity,
                        features: 0,
                        tracked: 0,
                        inliers: 0,
                        coasting: true,
                        anchored: false,
                        off_canvas,
                        depth: Vec::new(),
                    },
                    true,
                )
            }
        };
        let pred = st.homography.then_apply(&self.anchor_to_prev);
        let mut anchor_to_next = pred;
        let mut renew = !coasting;
        let expected = self.expected_in_frame(&pred, fw, fh);
        if let Some((t_abs, inl)) = self.register_anchor(&next, &pred) {
            let refined = self.anchor_to_prev.inverse().map(|inv| t_abs.then_apply(&inv)).filter(|r| plausible(r, region));
            if let (Some(r), true) = (refined, inl >= (expected / 3).max(MIN_INLIERS)) {
                st = PlanarStep { homography: r, inliers: inl, anchored: true, ..st };
                anchor_to_next = t_abs;
                renew = inl * 2 < expected;
                if coasting {
                    // the anchor re-acquired the plane
                    self.coasted = 0;
                    self.velocity = r;
                    st.coasting = false;
                }
            }
        }
        let region_next = match warp_all(&st.homography, region) {
            Some(r) => r,
            None => {
                // the fit sends a corner to infinity (a plane edge-on, or leaving the canvas):
                // coast on the last believable motion instead of ending the track
                self.velocity = damp(&self.velocity, damping);
                st = PlanarStep { homography: self.velocity, coasting: true, anchored: false, ..st };
                anchor_to_next = st.homography.then_apply(&self.anchor_to_prev);
                renew = false;
                warp_all(&st.homography, region)?
            }
        };
        self.anchor_to_prev = anchor_to_next;
        self.prev = next;
        if renew {
            let feats = features(self.prev.base(), &region_next, MAX_FEATURES, 5.0);
            if feats.len() >= MIN_INLIERS {
                self.anchor = self.prev.clone();
                self.anchor_feats = feats;
                self.anchor_region = region_next;
                self.anchor_to_prev = Homography::IDENTITY;
            }
        }
        Some(st)
    }

    /// How many anchor features `pred` (anchor → next) puts inside the `w`×`h` canvas.
    fn expected_in_frame(&self, pred: &Homography, w: usize, h: usize) -> usize {
        const EDGE: f64 = 8.0;
        let (fw, fh) = (w as f64, h as f64);
        self.anchor_feats
            .iter()
            .filter(|p| pred.apply(Vec2::new(p[0] as f64, p[1] as f64)).is_some_and(|q| q.x >= EDGE && q.y >= EDGE && q.x < fw - EDGE && q.y < fh - EDGE))
            .count()
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
        fit_model(&src, &dst, &self.anchor_region, self.threshold)
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

    /// Pure shift to the right, `9 px` per frame.
    fn slide(t: f64) -> Homography {
        Homography([1.0, 0.0, 9.0 * t, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    }

    /// Regression: a plane sliding out of the canvas used to end the track (too few features, then
    /// the extrapolation ran out after `MAX_COAST` frames). It now coasts off the canvas.
    #[test]
    fn leaving_the_canvas_keeps_the_track() {
        let (w, h) = (320, 240);
        let mut region = quad();
        let truth = |k: f64| -> Vec<Vec2> { quad().iter().filter_map(|p| slide(k).apply(*p)).collect() };
        let mut tracker = PlanarTracker::new(Prepared::new(frame(w, h, &slide(0.0))), &region);
        let mut worst = 0.0f64;
        let mut went_off = false;
        for k in 1..=45 {
            let st = tracker.advance(Prepared::new(frame(w, h, &slide(k as f64))), &region).expect("track survives leaving the canvas");
            region = warp_all(&st.homography, &region).expect("region stays finite");
            went_off |= st.off_canvas;
            if k <= 10 {
                for (a, b) in region.iter().zip(truth(k as f64)) {
                    worst = worst.max((*a - b).length());
                }
            }
        }
        assert!(worst < 2.0, "worst corner error {worst} px while fully visible");
        assert!(went_off, "the region ended up off the canvas");
    }

    /// With a cluster of features covering a sliver of the region, the far corners must follow the
    /// shift instead of being extrapolated by a free homography.
    #[test]
    fn sparse_coverage_falls_back_to_a_shift() {
        let region = quad();
        let (mut src, mut dst) = (Vec::new(), Vec::new());
        for i in 0..16 {
            let p = Vec2::new(100.0 + 5.0 * (i % 4) as f64, 80.0 + 5.0 * (i / 4) as f64);
            let noise = Vec2::new(0.25 * (i as f64 * 1.7).sin(), 0.25 * (i as f64 * 2.3).cos());
            src.push(p);
            dst.push(p + Vec2::new(5.0, 3.0) + noise);
        }
        assert!(coverage(&src, &region) < COVERAGE_SHIFT);
        let (h, inl) = fit_model(&src, &dst, &region, THRESHOLD).expect("fit");
        assert!(inl >= 12);
        for c in &region {
            let moved = h.apply(*c).expect("corner");
            assert!((moved - (*c + Vec2::new(5.0, 3.0))).length() < 0.6, "corner {c:?} went to {moved:?}");
        }
    }

    #[test]
    fn collinear_features_count_as_no_coverage() {
        let line: Vec<Vec2> = (0..12).map(|i| Vec2::new(90.0 + 10.0 * i as f64, 70.0 + 8.0 * i as f64)).collect();
        assert_eq!(coverage(&line, &quad()), 0.0);
    }

    #[test]
    fn parallax_marks_features_off_the_plane() {
        let t = pose(5.0);
        let (mut src, mut dst) = (Vec::new(), Vec::new());
        for i in 0..20 {
            let p = Vec2::new(40.0 + (i * 37 % 200) as f64, 30.0 + (i * 61 % 150) as f64);
            src.push(p);
            let q = t.apply(p).expect("q");
            dst.push(if i >= 15 { q + Vec2::new(6.0, 0.0) } else { q });
        }
        let depth = depth_samples(&t, &src, &dst);
        assert_eq!(depth.len(), 20);
        assert!(depth.iter().take(15).all(|d| d.parallax < 0.05));
        assert!(depth.iter().skip(15).all(|d| d.parallax > 0.8));
    }

    #[test]
    fn depth_strength_tightens_the_inlier_threshold() {
        let at = |s: f64| PlanarOptions { depth_strength: s }.threshold();
        assert!(at(1.0) < at(0.6) && at(0.6) < at(0.0));
        assert_eq!(at(f64::NAN), at(PlanarOptions::default().depth_strength));
        assert_eq!(at(9.0), at(1.0));
        assert_eq!(at(-3.0), at(0.0));
    }

    #[test]
    fn visible_fraction_counts_the_part_on_the_canvas() {
        let r = |x0: f64, x1: f64| vec![Vec2::new(x0, 10.0), Vec2::new(x1, 10.0), Vec2::new(x1, 50.0), Vec2::new(x0, 50.0)];
        assert!((visible_fraction(&r(10.0, 50.0), 100, 100) - 1.0).abs() < 1e-9);
        assert!((visible_fraction(&r(80.0, 120.0), 100, 100) - 0.5).abs() < 1e-9);
        assert_eq!(visible_fraction(&r(200.0, 240.0), 100, 100), 0.0);
        assert_eq!(visible_fraction(&[], 100, 100), 0.0);
    }
}
