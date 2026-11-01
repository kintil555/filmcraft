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
use filmcraft_depth::DepthMap;
use filmcraft_geom::{Homography, Vec2};
use filmcraft_project::TrackMethod;
use rayon::prelude::*;

/// Inlier threshold of the homography fit, in pixels (looser than the similarity fit: a plane in
/// perspective is not perfectly modelled by sparse, sub-pixel features).
#[cfg(test)]
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
/// How far (fraction of the normalised 0..1 neural depth range) a feature may sit off the plane's
/// depth at depth strength 0 (tolerant) and 1 (strict).
const DEPTH_TOL_LOOSE: f64 = 0.45;
const DEPTH_TOL_TIGHT: f64 = 0.08;
/// Fewest matches a depth plane is fitted through.
const MIN_DEPTH_POINTS: usize = 6;
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
    /// The depth strength as a number in 0..=1 (a NaN strength reads as the default).
    fn strength(&self) -> f64 {
        if self.depth_strength.is_finite() { self.depth_strength.clamp(0.0, 1.0) } else { Self::default().depth_strength }
    }

    /// Inlier threshold in pixels for this depth strength.
    pub fn threshold(&self) -> f64 {
        THRESHOLD_LOOSE + (THRESHOLD_TIGHT - THRESHOLD_LOOSE) * self.strength()
    }

    /// How far (fraction of the neural depth range) a feature may sit off the plane's depth.
    pub fn depth_tolerance(&self) -> f64 {
        DEPTH_TOL_LOOSE + (DEPTH_TOL_TIGHT - DEPTH_TOL_LOOSE) * self.strength()
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
#[cfg(test)]
pub fn fit_robust(src: &[Vec2], dst: &[Vec2]) -> Option<(Homography, usize)> {
    fit_robust_with(src, dst, THRESHOLD).map(|(h, inl)| (h, inl.len()))
}

/// `fit_robust` with an explicit inlier `threshold` (px); returns the inlier indices.
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

/// Neural depth as an affine function of the picture position, `z = a·(x - mx) + b·(y - my) + c`:
/// inverse depth is affine across a plane, which is what makes a slanted wall a single plane here.
struct DepthPlane {
    mx: f64,
    my: f64,
    a: f64,
    b: f64,
    c: f64,
}

impl DepthPlane {
    /// Least-squares plane through `(x, y, z)` points; `None` for too few or collinear points.
    fn fit(pts: &[(f64, f64, f64)]) -> Option<Self> {
        if pts.len() < MIN_DEPTH_POINTS {
            return None;
        }
        let n = pts.len() as f64;
        let (mx, my, c) = (pts.iter().map(|p| p.0).sum::<f64>() / n, pts.iter().map(|p| p.1).sum::<f64>() / n, pts.iter().map(|p| p.2).sum::<f64>() / n);
        let (mut sxx, mut sxy, mut syy, mut sxz, mut syz) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for &(x, y, z) in pts {
            let (dx, dy, dz) = (x - mx, y - my, z - c);
            sxx += dx * dx;
            sxy += dx * dy;
            syy += dy * dy;
            sxz += dx * dz;
            syz += dy * dz;
        }
        let det = sxx * syy - sxy * sxy;
        if !det.is_finite() || sxx <= 1e-12 || syy <= 1e-12 || det <= 1e-6 * sxx * syy {
            return None;
        }
        let (a, b) = ((sxz * syy - syz * sxy) / det, (syz * sxx - sxz * sxy) / det);
        (a.is_finite() && b.is_finite()).then_some(Self { mx, my, a, b, c })
    }

    fn at(&self, x: f64, y: f64) -> f64 {
        self.a * (x - self.mx) + self.b * (y - self.my) + self.c
    }
}

/// Which of the matches `src` lie on one depth plane of `depth` (within `tol` of the 0..1 depth
/// range), refitting the plane on the survivors. `None` when the depth cannot decide (too few
/// matches, or a degenerate layout), in which case no match is rejected.
fn depth_keep(src: &[Vec2], depth: &DepthMap, tol: f64) -> Option<Vec<bool>> {
    let z: Vec<f64> = src.iter().map(|p| depth.sample(p.x, p.y) as f64).collect();
    let mut keep = vec![true; src.len()];
    for cut in [tol * 3.0, tol * 1.5, tol] {
        let pts: Vec<(f64, f64, f64)> = src.iter().zip(&z).zip(&keep).filter(|(_, k)| **k).map(|((p, z), _)| (p.x, p.y, *z)).collect();
        let plane = DepthPlane::fit(&pts)?;
        for ((p, z), k) in src.iter().zip(&z).zip(keep.iter_mut()) {
            *k = (z - plane.at(p.x, p.y)).abs() <= cut;
        }
    }
    Some(keep)
}

/// A frame-to-frame fit.
struct FrameFit {
    h: Homography,
    features: usize,
    tracked: usize,
    inliers: usize,
    depth: Vec<DepthSample>,
}

/// Frame-to-frame planar step from `a` to `b` over the picture inside `region`. `depth` is the
/// neural depth of `a`: matches off the plane's depth are left out of the fit.
fn step(a: &Prepared, b: &Prepared, region: &[Vec2], options: &PlanarOptions, depth: Option<&DepthMap>) -> Option<FrameFit> {
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
    let (all_src, all_dst): (Vec<Vec2>, Vec<Vec2>) = pairs.into_iter().unzip();
    let (src, dst): (Vec<Vec2>, Vec<Vec2>) = match depth.and_then(|d| depth_keep(&all_src, d, options.depth_tolerance())) {
        Some(keep) if keep.iter().filter(|k| **k).count() >= MIN_INLIERS => {
            all_src.iter().zip(&all_dst).zip(&keep).filter(|(_, k)| **k).map(|((s, d), _)| (*s, *d)).unzip()
        }
        _ => (all_src.clone(), all_dst.clone()),
    };
    let threshold = options.threshold();
    // a strict depth setting that leaves too few plane features falls back to the tolerant fit
    let (h, inliers) =
        fit_model(&src, &dst, region, threshold).filter(|(_, n)| *n >= MIN_INLIERS).or_else(|| fit_model(&src, &dst, region, THRESHOLD_LOOSE))?;
    (inliers >= MIN_INLIERS).then(|| FrameFit { h, features: feats.len(), tracked: src.len(), inliers, depth: depth_samples(&h, &all_src, &all_dst) })
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
    options: PlanarOptions,
    /// Neural depth of `prev`, when a depth model is running.
    prev_depth: Option<DepthMap>,
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
            options,
            prev_depth: None,
        }
    }

    /// Give the neural depth of the frame the tracker is on (call right after construction).
    pub fn set_depth(&mut self, depth: Option<DepthMap>) {
        self.prev_depth = depth;
    }

    /// Advance to `next`; `region` is the plane's polygon in the previous frame's pixels. The
    /// returned homography maps previous-frame pixels to `next`'s. `None` once the track is lost.
    pub fn advance(&mut self, next: Prepared, region: &[Vec2]) -> Option<PlanarStep> {
        self.advance_with_depth(next, region, None)
    }

    /// [`PlanarTracker::advance`] that also takes the neural depth of `next` (kept for the step
    /// after this one, where it tells the plane's features from the rest).
    pub fn advance_with_depth(&mut self, next: Prepared, region: &[Vec2], next_depth: Option<DepthMap>) -> Option<PlanarStep> {
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
        let (st, coasting) = self.frame_motion(&next, region, off_canvas, (max_coast, damping))?;
        let (mut st, mut anchor_to_next, mut renew) = self.refine(&next, region, st, coasting, (fw, fh));
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
        self.prev_depth = next_depth;
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

    /// Frame-to-frame motion to `next`, or the damped last motion when the match fails. `limits`
    /// is (most frames to coast, damping). Returns the step and whether it is coasting; `None` once
    /// the coasting has run out.
    fn frame_motion(&mut self, next: &Prepared, region: &[Vec2], off_canvas: bool, limits: (usize, f64)) -> Option<(PlanarStep, bool)> {
        let (max_coast, damping) = limits;
        let ff = step(&self.prev, next, region, &self.options, self.prev_depth.as_ref()).filter(|f| plausible(&f.h, region));
        match ff {
            Some(f) => {
                self.velocity = f.h;
                self.coasted = 0;
                let st = PlanarStep {
                    homography: f.h,
                    features: f.features,
                    tracked: f.tracked,
                    inliers: f.inliers,
                    coasting: false,
                    anchored: false,
                    off_canvas,
                    depth: f.depth,
                };
                Some((st, false))
            }
            None => {
                self.coasted += 1;
                if self.coasted > max_coast {
                    return None;
                }
                self.velocity = damp(&self.velocity, damping);
                let st = PlanarStep {
                    homography: self.velocity,
                    features: 0,
                    tracked: 0,
                    inliers: 0,
                    coasting: true,
                    anchored: false,
                    off_canvas,
                    depth: Vec::new(),
                };
                Some((st, true))
            }
        }
    }

    /// Drift correction of `st` against the anchor frame. Returns the (possibly refined) step, the
    /// anchor → next motion, and whether the anchor should be renewed on this frame.
    fn refine(&mut self, next: &Prepared, region: &[Vec2], st: PlanarStep, coasting: bool, (fw, fh): (usize, usize)) -> (PlanarStep, Homography, bool) {
        let pred = st.homography.then_apply(&self.anchor_to_prev);
        let (mut st, mut anchor_to_next, mut renew) = (st, pred, !coasting);
        let expected = self.expected_in_frame(&pred, fw, fh);
        if let Some((t_abs, inl)) = self.register_anchor(next, &pred) {
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
        (st, anchor_to_next, renew)
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
        fit_model(&src, &dst, &self.anchor_region, self.options.threshold())
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

    #[test]
    fn depth_keep_rejects_features_at_another_depth() {
        // a slanted plane (inverse depth affine in x) with a near object in front of it
        let (w, h) = (320usize, 240usize);
        let px: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                if (100.0..160.0).contains(&x) && (50.0..100.0).contains(&y) { 0.95 } else { 0.5 + 0.001 * x }
            })
            .collect();
        let depth = DepthMap::new(w, h, px).expect("map");
        let src: Vec<Vec2> = (0..7).flat_map(|i| (0..6).map(move |j| Vec2::new(30.0 + 40.0 * i as f64, 30.0 + 30.0 * j as f64))).collect();
        let tol = PlanarOptions { depth_strength: 1.0 }.depth_tolerance();
        let keep = depth_keep(&src, &depth, tol).expect("decides");
        for (p, k) in src.iter().zip(&keep) {
            let in_front = (100.0..160.0).contains(&p.x) && (50.0..100.0).contains(&p.y);
            assert_eq!(*k, !in_front, "feature at {p:?}");
        }
    }

    #[test]
    fn depth_cannot_decide_without_enough_matches() {
        let depth = DepthMap::new(8, 8, vec![0.5; 64]).expect("map");
        let few = [Vec2::new(1.0, 1.0), Vec2::new(5.0, 2.0), Vec2::new(3.0, 6.0)];
        assert!(depth_keep(&few, &depth, 0.1).is_none());
        let line: Vec<Vec2> = (0..8).map(|i| Vec2::new(i as f64, i as f64)).collect();
        assert!(depth_keep(&line, &depth, 0.1).is_none(), "collinear matches pin no plane");
    }

    #[test]
    fn depth_strength_tightens_the_depth_tolerance() {
        let at = |s: f64| PlanarOptions { depth_strength: s }.depth_tolerance();
        assert!(at(1.0) < at(0.6) && at(0.6) < at(0.0));
        assert_eq!(at(f64::NAN), at(PlanarOptions::default().depth_strength));
    }
}
