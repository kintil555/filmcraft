//! Projective transforms (3×3 homographies): planar tracking and Corner Pin math.

use crate::{Affine, Vec2};
use serde::{Deserialize, Serialize};

/// A projective transform, row-major `[h0 h1 h2; h3 h4 h5; h6 h7 h8]`, mapping
/// (x, y) → ((h0 x + h1 y + h2) / w, (h3 x + h4 y + h5) / w) with w = h6 x + h7 y + h8.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Homography(pub [f64; 9]);

impl Default for Homography {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Smallest |w| a point may have after projection (guards the divide).
const MIN_W: f64 = 1e-9;

/// `x` is NaN or no bigger than `eps` in magnitude.
fn small(x: f64, eps: f64) -> bool {
    x.is_nan() || x.abs() <= eps
}

fn mul3(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut o = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            o[r * 3 + c] = (0..3).map(|k| a[r * 3 + k] * b[k * 3 + c]).sum();
        }
    }
    o
}

/// Solve the `N`×`N` system `a x = b` (Gaussian elimination, partial pivoting). `None` when singular.
fn solve<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        let piv = (col..N).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if small(a[piv][col], 1e-12) {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in col + 1..N {
            let k = a[r][col] / a[col][col];
            for c in col..N {
                a[r][c] -= k * a[col][c];
            }
            b[r] -= k * b[col];
        }
    }
    let mut x = [0.0; N];
    for r in (0..N).rev() {
        let s: f64 = (r + 1..N).map(|c| a[r][c] * x[c]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

impl Homography {
    pub const IDENTITY: Homography = Homography([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

    pub fn from_affine(a: &Affine) -> Self {
        Homography([a.a, a.c, a.e, a.b, a.d, a.f, 0.0, 0.0, 1.0])
    }

    pub fn is_finite(&self) -> bool {
        self.0.iter().all(|v| v.is_finite())
    }

    /// The image of `p`; `None` when it goes to infinity (or behind the camera plane's horizon).
    pub fn apply(&self, p: Vec2) -> Option<Vec2> {
        let h = &self.0;
        let w = h[6] * p.x + h[7] * p.y + h[8];
        if small(w, MIN_W) {
            return None;
        }
        let q = Vec2::new((h[0] * p.x + h[1] * p.y + h[2]) / w, (h[3] * p.x + h[4] * p.y + h[5]) / w);
        (q.x.is_finite() && q.y.is_finite()).then_some(q)
    }

    /// `self ∘ o`: apply `o` first, then `self` (same convention as [`Affine::then_apply`]).
    pub fn then_apply(&self, o: &Homography) -> Homography {
        Homography(mul3(&self.0, &o.0))
    }

    pub fn inverse(&self) -> Option<Homography> {
        let m = &self.0;
        let c = [
            m[4] * m[8] - m[5] * m[7],
            m[2] * m[7] - m[1] * m[8],
            m[1] * m[5] - m[2] * m[4],
            m[5] * m[6] - m[3] * m[8],
            m[0] * m[8] - m[2] * m[6],
            m[2] * m[3] - m[0] * m[5],
            m[3] * m[7] - m[4] * m[6],
            m[1] * m[6] - m[0] * m[7],
            m[0] * m[4] - m[1] * m[3],
        ];
        let det = m[0] * c[0] + m[1] * c[3] + m[2] * c[6];
        if small(det, 1e-15) {
            return None;
        }
        let inv = Homography(c.map(|v| v / det));
        inv.is_finite().then_some(inv)
    }

    /// Scaled so that `h8 = 1` when that is possible (a stable form for storing and comparing).
    pub fn normalized(&self) -> Homography {
        let s = self.0[8];
        if s.abs() > 1e-12 { Homography(self.0.map(|v| v / s)) } else { *self }
    }

    /// The affine map that best matches this one around `p` (its Jacobian there).
    pub fn linearize_at(&self, p: Vec2) -> Option<Affine> {
        let h = &self.0;
        let w = h[6] * p.x + h[7] * p.y + h[8];
        if small(w, MIN_W) {
            return None;
        }
        let q = self.apply(p)?;
        let (ux, uy) = (h[0] * p.x + h[1] * p.y + h[2], h[3] * p.x + h[4] * p.y + h[5]);
        let (a, c) = ((h[0] - h[6] * ux / w) / w, (h[1] - h[7] * ux / w) / w);
        let (b, d) = ((h[3] - h[6] * uy / w) / w, (h[4] - h[7] * uy / w) / w);
        let lin = Affine { a, b, c, d, e: 0.0, f: 0.0 };
        let t = q - lin.apply(p);
        let m = Affine { e: t.x, f: t.y, ..lin };
        [m.a, m.b, m.c, m.d, m.e, m.f].iter().all(|v| v.is_finite()).then_some(m)
    }

    /// Least-squares homography mapping `src[i]` → `dst[i]` (normalised direct linear transform;
    /// exact for four points in general position). `None` for fewer than four points, mismatched
    /// lengths, non-finite input or a degenerate configuration (e.g. collinear points).
    pub fn fit(src: &[Vec2], dst: &[Vec2]) -> Option<Homography> {
        let n = src.len();
        if n < 4 || dst.len() != n || src.iter().chain(dst).any(|p| !(p.x.is_finite() && p.y.is_finite())) {
            return None;
        }
        let (ts, td) = (Self::normalizer(src)?, Self::normalizer(dst)?);
        // normal equations of the 2n × 8 system (h8 = 1)
        let mut ata = [[0.0f64; 8]; 8];
        let mut atb = [0.0f64; 8];
        for (s, d) in src.iter().zip(dst) {
            let (s, d) = (ts.apply(*s)?, td.apply(*d)?);
            let rows = [([s.x, s.y, 1.0, 0.0, 0.0, 0.0, -d.x * s.x, -d.x * s.y], d.x), ([0.0, 0.0, 0.0, s.x, s.y, 1.0, -d.y * s.x, -d.y * s.y], d.y)];
            for (row, rhs) in rows {
                for i in 0..8 {
                    for j in 0..8 {
                        ata[i][j] += row[i] * row[j];
                    }
                    atb[i] += row[i] * rhs;
                }
            }
        }
        let h = solve(ata, atb)?;
        let hn = Homography([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0]);
        let full = td.inverse()?.then_apply(&hn).then_apply(&ts);
        let full = full.normalized();
        full.is_finite().then_some(full)
    }

    /// Hartley normalisation: centroid to the origin, mean distance √2.
    fn normalizer(pts: &[Vec2]) -> Option<Homography> {
        let inv = 1.0 / pts.len().max(1) as f64;
        let c = pts.iter().fold(Vec2::ZERO, |a, p| a + *p) * inv;
        let mean = pts.iter().map(|p| (*p - c).length()).sum::<f64>() * inv;
        if small(mean, 1e-9) {
            return None;
        }
        let s = std::f64::consts::SQRT_2 / mean;
        Some(Homography([s, 0.0, -s * c.x, 0.0, s, -s * c.y, 0.0, 0.0, 1.0]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn truth() -> Homography {
        Homography([1.1, 0.12, 30.0, -0.08, 0.95, -12.0, 0.0004, -0.0002, 1.0])
    }

    fn grid() -> Vec<Vec2> {
        (0..12).map(|i| Vec2::new(40.0 + (i % 4) as f64 * 90.0 + (i * 7 % 5) as f64, 30.0 + (i / 4) as f64 * 70.0 + (i * 3 % 4) as f64)).collect()
    }

    #[test]
    fn fit_recovers_a_projective_map_exactly() {
        let src = grid();
        let dst: Vec<Vec2> = src.iter().filter_map(|p| truth().apply(*p)).collect();
        let h = Homography::fit(&src, &dst).expect("fit");
        for p in &src {
            let (a, b) = (h.apply(*p).expect("a"), truth().apply(*p).expect("b"));
            assert!((a - b).length() < 1e-6, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn four_points_are_enough_and_three_are_not() {
        let src = [Vec2::new(0.0, 0.0), Vec2::new(100.0, 0.0), Vec2::new(100.0, 80.0), Vec2::new(0.0, 80.0)];
        let dst = [Vec2::new(10.0, 5.0), Vec2::new(120.0, 0.0), Vec2::new(90.0, 85.0), Vec2::new(0.0, 70.0)];
        let h = Homography::fit(&src, &dst).expect("fit");
        for (s, d) in src.iter().zip(&dst) {
            assert!((h.apply(*s).expect("p") - *d).length() < 1e-6);
        }
        assert!(Homography::fit(&src[..3], &dst[..3]).is_none());
    }

    #[test]
    fn fit_refuses_degenerate_and_hostile_input() {
        let line: Vec<Vec2> = (0..6).map(|i| Vec2::new(i as f64 * 10.0, i as f64 * 5.0)).collect();
        assert!(Homography::fit(&line, &line).is_none());
        let same = [Vec2::new(3.0, 3.0); 5];
        assert!(Homography::fit(&same, &same).is_none());
        let mut bad = grid();
        bad[2] = Vec2::new(f64::NAN, 1.0);
        assert!(Homography::fit(&bad, &grid()).is_none());
        assert!(Homography::fit(&grid(), &grid()[..5]).is_none());
    }

    #[test]
    fn inverse_and_composition_round_trip() {
        let h = truth();
        let id = h.then_apply(&h.inverse().expect("inv"));
        for p in grid() {
            assert!((id.apply(p).expect("p") - p).length() < 1e-6);
        }
        assert!(Homography([0.0; 9]).inverse().is_none());
    }

    #[test]
    fn apply_rejects_points_at_infinity() {
        let h = Homography([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
        assert!(h.apply(Vec2::new(0.0, 5.0)).is_none());
        assert!(h.apply(Vec2::new(2.0, 5.0)).is_some());
    }

    #[test]
    fn affine_round_trip_and_linearisation() {
        let a = Affine::motion(Vec2::new(10.0, -4.0), Vec2::new(1.5, 1.5), 20.0, Vec2::new(3.0, 2.0));
        let h = Homography::from_affine(&a);
        let p = Vec2::new(17.0, 9.0);
        assert!((h.apply(p).expect("h") - a.apply(p)).length() < 1e-9);
        let lin = h.linearize_at(p).expect("lin");
        assert!((lin.apply(Vec2::new(40.0, 60.0)) - a.apply(Vec2::new(40.0, 60.0))).length() < 1e-9);
        // a true projective map: the linearisation agrees near the point
        let t = truth();
        let l = t.linearize_at(p).expect("lin");
        let q = p + Vec2::new(0.5, -0.5);
        assert!((l.apply(q) - t.apply(q).expect("t")).length() < 1e-3);
    }
}
