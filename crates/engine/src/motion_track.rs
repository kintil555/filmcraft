//! Tracking Motion: `tracking.*` commands.
//!
//! The **Tracking Motion** effect (Effects ▸ Tracking) marks a region of a clip and follows it:
//!
//! * the region is the effect's first mask (`Tracking Region`), so everything the mask tracker
//!   already does applies: the Track buttons (`masks.track`: Lucas–Kanade + RANSAC, see
//!   `filmcraft_render::track`) write the region's path as keyframes, frame by frame;
//! * `tracking.setup` creates that region, **Auto** (the most trackable patch of the frame,
//!   found with the tracker's own feature detector) or **Manual** (a centred box to move and
//!   resize on the monitor);
//! * `tracking.attach` links another clip (text, image, shape…) to the track: the region's motion
//!   (position, and rotation / scale per the effect's *Follow* choice) is baked into the follower's
//!   Motion keyframes, relative to where the follower sits at the playhead when it is attached.
//!   The links live in the effect's `links` text parameter, so a new track (see
//!   [`on_track_finished`]) or `tracking.update` refreshes every follower;
//! * `tracking.detach` unlinks a clip and gives it back its own static Motion.
//!
//! Every command is one undo step.

use filmcraft_geom::Vec2;
use filmcraft_project::{ClipId, Mask, MaskPath, ParamValue, TrackItem, TrackMethod};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, clip_p, has_seq, str_p, time_p, u64_p};
use crate::masks::MaskSel;
use crate::{EngineError, Result, Session};

/// Effect id (see `filmcraft_project::effect`).
pub const EFFECT: &str = "tracking_motion";

type Run = fn(&mut Session, &Value) -> Result<Value>;

fn spec(id: &'static str, label: &'static str, params: &'static str, run: Run) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled: has_seq, run, journal: true }
}

pub(crate) fn commands() -> Vec<CommandSpec> {
    vec![
        spec(
            "tracking.setup",
            "Set Up Tracking Region",
            r#"{"clip":id?,"effect":index?,"mode":"auto"|"manual"?,"method":"position|positionRotation|positionScaleRotation"?,"reset":bool?,"time":ticks?}"#,
            setup,
        ),
        spec("tracking.attach", "Attach Clip to Tracking", r#"{"clip":id?,"effect":index?,"follower":id,"time":ticks?}"#, attach),
        spec("tracking.detach", "Detach Clip from Tracking", r#"{"clip":id?,"effect":index?,"follower":id}"#, detach),
        spec("tracking.update", "Update Tracked Clips", r#"{"clip":id?,"effect":index?}"#, update),
    ]
}

// ---------------------------------------------------------------- links

/// A clip that follows the track. `pos` / `rot` / `scale` are its own Motion values at the moment it
/// was attached (sequence pixels, degrees, percent) and `at` the tracked clip's media time then.
#[derive(Clone, Debug, PartialEq)]
struct Link {
    clip: ClipId,
    pos: Vec2,
    rot: f64,
    scale: f64,
    at: Tick,
}

impl Link {
    fn encode(&self) -> String {
        format!("{},{},{},{},{},{}", self.clip.0, self.pos.x, self.pos.y, self.rot, self.scale, self.at.0)
    }

    fn decode(s: &str) -> Option<Link> {
        let mut it = s.split(',');
        let clip = ClipId(it.next()?.trim().parse().ok()?);
        let mut num = || it.next()?.trim().parse::<f64>().ok().filter(|v| v.is_finite());
        let (x, y, rot, scale) = (num()?, num()?, num()?, num()?);
        let at = Tick(it.next()?.trim().parse().ok()?);
        Some(Link { clip, pos: Vec2::new(x, y), rot, scale, at })
    }
}

/// The links stored in the effect's `links` text (`;`-separated). Damaged entries are dropped.
fn parse_links(text: &str) -> Vec<Link> {
    text.split(';').filter_map(Link::decode).collect()
}

fn write_links(links: &[Link]) -> String {
    links.iter().map(Link::encode).collect::<Vec<_>>().join(";")
}

fn links_of(e: &filmcraft_project::EffectInstance) -> Vec<Link> {
    match e.param("links").map(|p| &p.value) {
        Some(ParamValue::Text(t)) => parse_links(t),
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------- geometry

/// How a tracked region sits now compared with where it started.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Rel {
    /// Centroid of the region now (clip pixels).
    center: Vec2,
    /// Rotation since the start, degrees (clockwise on screen).
    rot_deg: f64,
    /// Size since the start (1 = unchanged).
    scale: f64,
}

fn vertices(path: &MaskPath) -> Vec<Vec2> {
    path.vertices.iter().map(|v| v.p).collect()
}

fn centroid(pts: &[Vec2]) -> Option<Vec2> {
    if pts.is_empty() {
        return None;
    }
    let sum = pts.iter().fold(Vec2::ZERO, |a, p| a + *p);
    Some(sum * (1.0 / pts.len() as f64))
}

/// Rotation, scale and centre of `cur` against `reference` (the same vertices, moved): the
/// closed-form 2D Procrustes fit about the centroids. `None` when the shapes don't correspond or
/// are degenerate.
fn relative(reference: &[Vec2], cur: &[Vec2]) -> Option<Rel> {
    if reference.len() != cur.len() || reference.len() < 2 {
        return None;
    }
    let (c0, c1) = (centroid(reference)?, centroid(cur)?);
    let (mut cross, mut dot, mut n0, mut n1) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (a, b) in reference.iter().zip(cur) {
        let (a, b) = (*a - c0, *b - c1);
        cross += a.x * b.y - a.y * b.x;
        dot += a.x * b.x + a.y * b.y;
        n0 += a.x * a.x + a.y * a.y;
        n1 += b.x * b.x + b.y * b.y;
    }
    if !(n0 > 1e-9 && n1.is_finite() && cross.is_finite() && dot.is_finite()) {
        return None;
    }
    Some(Rel { center: c1, rot_deg: cross.atan2(dot).to_degrees(), scale: (n1 / n0).sqrt() })
}

/// `offset` turned clockwise by `deg` (y points down) and scaled by `grow`.
fn orbit(offset: Vec2, deg: f64, grow: f64) -> Vec2 {
    let (s, c) = deg.to_radians().sin_cos();
    Vec2::new((offset.x * c - offset.y * s) * grow, (offset.x * s + offset.y * c) * grow)
}

// ---------------------------------------------------------------- planning

/// One baked Motion sample of a follower.
#[derive(Clone, Copy, Debug)]
struct Sample {
    /// Follower media (effect) time.
    mt: Tick,
    pos: Vec2,
    rot: f64,
    scale: f64,
}

struct Plan {
    tracker: ClipId,
    effect: usize,
    links: Vec<Link>,
    /// (follower, samples) per link that could be baked.
    bakes: Vec<(ClipId, Vec<Sample>)>,
}

/// Most frames baked per follower (a guard against absurd clip lengths).
const MAX_FRAMES: i64 = 100_000;

fn method_of(e: &filmcraft_project::EffectInstance) -> TrackMethod {
    match e.param("method").map(|p| &p.value) {
        Some(ParamValue::Choice(i)) => TrackMethod::ALL.get(*i as usize).copied().unwrap_or_default(),
        _ => TrackMethod::default(),
    }
}

/// The tracker's effect and its region mask.
fn region(it: &TrackItem, effect: usize) -> Result<(&filmcraft_project::EffectInstance, &Mask)> {
    let e = it.effects.get(effect).filter(|e| e.effect == EFFECT).ok_or_else(|| bad("tracking", "that effect is not Tracking Motion"))?;
    let m = e.masks.first().ok_or_else(|| bad("tracking", "set up a tracking region first"))?;
    Ok((e, m))
}

/// What baking a follower needs to know about the tracked clip.
struct Tracked<'a> {
    seq: &'a filmcraft_project::Sequence,
    clip: &'a TrackItem,
    mask: &'a Mask,
    method: TrackMethod,
    size: (u32, u32),
    par: Option<(u32, u32)>,
    /// First / last tracked media time.
    span: (Tick, Tick),
    frame: Tick,
}

impl Tracked<'_> {
    /// `link`'s follower `fol` moved along the track, one sample per frame both clips cover.
    fn bake(&self, link: &Link, fol: &TrackItem) -> Vec<Sample> {
        let mut out = Vec::new();
        let ref_pts = vertices(&self.mask.path_at(link.at));
        let Some(ref_c) = centroid(&ref_pts) else { return out };
        let to_seq = |mt: Tick| filmcraft_render::motion_matrix(self.seq, self.clip, self.size, self.par, mt);
        let ref_seq = to_seq(link.at).apply(ref_c);
        // where the follower sits relative to the region's centre: that offset turns and grows with it
        let offset = link.pos - ref_seq;
        let (t0, t1) = (self.clip.start.max(fol.start), self.clip.end().min(fol.end()));
        let n = ((t1 - t0).0 / self.frame.0.max(1)).clamp(0, MAX_FRAMES);
        for k in 0..n {
            let t = t0 + Tick(self.frame.0 * k);
            let mt = self.clip.effect_time_at(t);
            if mt < self.span.0 || mt > self.span.1 {
                continue;
            }
            let Some(rel) = relative(&ref_pts, &vertices(&self.mask.path_at(mt))) else { continue };
            let rot = if self.method == TrackMethod::Position { link.rot } else { link.rot + rel.rot_deg };
            let scale = if self.method == TrackMethod::PositionScaleRotation { link.scale * rel.scale } else { link.scale };
            let (turn, grow) = (rot - link.rot, scale / link.scale.max(1e-6));
            let pos = to_seq(mt).apply(rel.center) + orbit(offset, turn, grow);
            if [pos.x, pos.y, rot, scale].iter().all(|v| v.is_finite()) {
                out.push(Sample { mt: fol.effect_time_at(t), pos, rot, scale });
            }
        }
        out
    }
}

/// Bake every link of `links` against the tracked region (read-only).
fn plan(s: &Session, tracker: ClipId, effect: usize, links: Vec<Link>) -> Result<Plan> {
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let (_, tr) = seq.find_item(tracker).ok_or_else(|| bad("tracking", "no such clip"))?;
    let (e, mask) = region(tr, effect)?;
    let span = match (mask.path.keyframes.first(), mask.path.keyframes.last()) {
        (Some(a), Some(b)) => (a.time, b.time),
        _ => return Err(bad("tracking", "track the region first (nothing tracked yet)")),
    };
    let frame = s.sequence_rate().frame_duration();
    if frame.0 <= 0 {
        return Err(bad("tracking", "the sequence has no frame rate"));
    }
    let tracked = Tracked {
        seq,
        clip: tr,
        mask,
        method: method_of(e),
        size: filmcraft_render::source_size(&s.project, tr.item).unwrap_or((seq.settings.width, seq.settings.height)),
        par: filmcraft_render::source_par(&s.project, tr.item),
        span,
        frame,
    };
    let bakes = links.iter().filter_map(|l| seq.find_item(l.clip).map(|(_, fol)| (l.clip, tracked.bake(l, fol)))).collect();
    Ok(Plan { tracker, effect, links, bakes })
}

/// Write a [`Plan`]: the links text on the tracker and the baked Motion keyframes on the followers.
fn commit(s: &mut Session, label: &str, pl: Plan) -> Result<usize> {
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let method = {
        let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
        let (_, tr) = seq.find_item(pl.tracker).ok_or_else(|| bad("tracking", "no such clip"))?;
        method_of(region(tr, pl.effect)?.0)
    };
    let frames: usize = pl.bakes.iter().map(|(_, v)| v.len()).sum();
    s.edit(label, move |pr, _| {
        let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
        for (clip, samples) in &pl.bakes {
            let Some((_, it)) = q.find_item_mut(*clip) else { continue };
            let Some(m) = it.effect_mut("motion") else { continue };
            let home = pl.links.iter().find(|l| l.clip == *clip);
            for (pid, pick) in [("position", 0), ("rotation", 1), ("scale", 2)] {
                let Some(p) = m.param_mut(pid) else { continue };
                if (pick == 1 && method == TrackMethod::Position) || (pick == 2 && method != TrackMethod::PositionScaleRotation) {
                    // this method leaves the value alone: drop keyframes an earlier method left behind
                    if let Some(l) = home.filter(|_| !p.keyframes.is_empty()) {
                        p.keyframes.clear();
                        p.value = ParamValue::Float(if pick == 1 { l.rot } else { l.scale });
                    }
                    continue;
                }
                p.keyframes.clear();
                for sm in samples {
                    let v = match pick {
                        0 => ParamValue::Vec2(sm.pos),
                        1 => ParamValue::Float(sm.rot),
                        _ => ParamValue::Float(sm.scale),
                    };
                    p.put_keyframe(sm.mt, v);
                }
            }
        }
        let (_, tr) = q.find_item_mut(pl.tracker).ok_or(filmcraft_edit::EditError::NoItem(pl.tracker))?;
        let e = tr.effects.get_mut(pl.effect).ok_or_else(|| bad("tracking", "the effect is gone"))?;
        if let Some(p) = e.param_mut("links") {
            p.value = ParamValue::Text(write_links(&pl.links));
            p.keyframes.clear();
        }
        Ok(())
    })?;
    Ok(frames)
}

// ---------------------------------------------------------------- commands

/// The tracker clip and the index of its Tracking Motion effect.
fn target(s: &Session, p: &Value, cmd: &str) -> Result<(ClipId, usize)> {
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let has_fx = |c: ClipId| seq.find_item(c).is_some_and(|(_, it)| it.effect(EFFECT).is_some());
    let clip = clip_p(p, "clip")
        .or_else(|| s.state.selected_mask.map(|m| m.clip).filter(|c| has_fx(*c)))
        .or_else(|| s.state.selection.iter().copied().find(|c| has_fx(*c)))
        .ok_or_else(|| bad(cmd, "select a clip with Tracking Motion (or pass `clip`)"))?;
    let (_, it) = seq.find_item(clip).ok_or_else(|| bad(cmd, "no such clip"))?;
    let idx = match u64_p(p, "effect") {
        Some(n) => n as usize,
        None => it.effects.iter().position(|e| e.effect == EFFECT).ok_or_else(|| bad(cmd, "the clip has no Tracking Motion effect"))?,
    };
    if it.effects.get(idx).map(|e| e.effect.as_str()) != Some(EFFECT) {
        return Err(bad(cmd, "that effect is not Tracking Motion"));
    }
    Ok((clip, idx))
}

/// Region box as a fraction of the frame when nothing better is known.
const DEFAULT_BOX: f64 = 0.25;
/// Frames the Auto region analysis works on are shrunk to this width (speed).
const AUTO_MAX_WIDTH: f64 = 960.0;
/// Auto region: the frame is cut into this many columns and rows and the cell with the most
/// trackable features wins.
const AUTO_GRID: (usize, usize) = (4, 3);

/// A centred box covering [`DEFAULT_BOX`] of the frame.
fn centred_box((w, h): (f64, f64)) -> MaskPath {
    let (bw, bh) = (w * DEFAULT_BOX, h * DEFAULT_BOX);
    MaskPath::rect((w - bw) / 2.0, (h - bh) / 2.0, (w + bw) / 2.0, (h + bh) / 2.0)
}

/// The grid cell of the frame at media time `mt` with the most Shi–Tomasi features (ties go to
/// the cell nearest the centre), as a clip-pixel box. `None` when the frame can't be read or has
/// no texture to lock on to.
fn auto_box(s: &Session, it: &TrackItem, mt: Tick, size: (f64, f64)) -> Option<MaskPath> {
    let src = s.source(it.item)?;
    let scale = (AUTO_MAX_WIDTH / size.0.max(1.0)).min(1.0) as f32;
    let f = src.video_frame(filmcraft_media::FrameRequest { time: mt, scale }).ok()?;
    let (fw, fh) = (f.width as usize, f.height as usize);
    let gray = filmcraft_render::track::Gray::from_rgba8(fw, fh, &f.to_rgba8());
    let frame_poly = [Vec2::new(0.0, 0.0), Vec2::new(fw as f64, 0.0), Vec2::new(fw as f64, fh as f64), Vec2::new(0.0, fh as f64)];
    let feats = filmcraft_render::track::features(&gray, &frame_poly, filmcraft_render::track::MAX_FEATURES, 5.0);
    let (cols, rows) = AUTO_GRID;
    let mut count = vec![0usize; cols * rows];
    for p in &feats {
        let cx = ((p[0] / fw.max(1) as f32) * cols as f32).floor().clamp(0.0, (cols - 1) as f32) as usize;
        let cy = ((p[1] / fh.max(1) as f32) * rows as f32).floor().clamp(0.0, (rows - 1) as f32) as usize;
        if let Some(c) = count.get_mut(cy * cols + cx) {
            *c += 1;
        }
    }
    let centre = |i: usize| {
        let (cx, cy) = ((i % cols) as f64 + 0.5 - cols as f64 / 2.0, (i / cols) as f64 + 0.5 - rows as f64 / 2.0);
        cx * cx + cy * cy
    };
    let best = (0..count.len()).filter(|i| count.get(*i).copied().unwrap_or(0) >= 6).max_by(|a, b| {
        let (ca, cb) = (count.get(*a).copied().unwrap_or(0), count.get(*b).copied().unwrap_or(0));
        ca.cmp(&cb).then(centre(*b).total_cmp(&centre(*a)))
    })?;
    let (cw, ch) = (size.0 / cols as f64, size.1 / rows as f64);
    let (x0, y0) = ((best % cols) as f64 * cw, (best / cols) as f64 * ch);
    // pull the box in a little so its edge isn't on the frame border
    let (mx, my) = (cw * 0.1, ch * 0.1);
    Some(MaskPath::rect(x0 + mx, y0 + my, x0 + cw - mx, y0 + ch - my))
}

fn mode_index(v: &str) -> Option<u32> {
    match v.to_ascii_lowercase().as_str() {
        "auto" => Some(0),
        "manual" => Some(1),
        _ => None,
    }
}

fn setup(s: &mut Session, p: &Value) -> Result<Value> {
    let (clip, ei) = target(s, p, "tracking.setup")?;
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let (_, it) = seq.find_item(clip).ok_or_else(|| bad("tracking.setup", "no such clip"))?;
    let it = it.clone();
    let e = it.effects.get(ei).ok_or_else(|| bad("tracking.setup", "no such effect"))?;
    let mode = match str_p(p, "mode") {
        Some(m) => mode_index(m).ok_or_else(|| bad("tracking.setup", format!("mode `{m}`: auto or manual")))?,
        None => match e.param("mode").map(|q| &q.value) {
            Some(ParamValue::Choice(i)) => *i,
            _ => 0,
        },
    };
    let method = match str_p(p, "method") {
        Some(m) => TrackMethod::from_name(m).ok_or_else(|| bad("tracking.setup", format!("unknown method `{m}`")))?,
        None => method_of(e),
    };
    let reset = p.get("reset").and_then(Value::as_bool).unwrap_or(false);
    let keep = !e.masks.is_empty() && !reset;
    let size = filmcraft_render::source_size(&s.project, it.item).ok_or_else(|| bad("tracking.setup", "the clip has no picture"))?;
    let size = (size.0 as f64, size.1 as f64);
    let ph = time_p(s, p, "").unwrap_or(s.playhead());
    let mt = it.source_time_at(ph.clamp(it.start, it.end() - Tick(1)));
    // Auto looks at the frame under the playhead; Manual (or no readable texture) starts from a centred box
    let path = if keep {
        None
    } else if mode == 0 {
        Some(auto_box(s, &it, mt, size).unwrap_or_else(|| centred_box(size)))
    } else {
        Some(centred_box(size))
    };
    s.edit("Set Up Tracking Region", move |pr, st| {
        let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
        let (_, it) = q.find_item_mut(clip).ok_or(filmcraft_edit::EditError::NoItem(clip))?;
        let e = it.effects.get_mut(ei).ok_or_else(|| bad("tracking.setup", "the effect is gone"))?;
        if let Some(path) = path {
            let mut m = Mask::new("Tracking Region", path);
            m.track_method = method;
            e.masks.clear();
            e.masks.push(m);
        } else if let Some(m) = e.masks.first_mut() {
            m.track_method = method;
        }
        if let Some(q) = e.param_mut("mode") {
            q.value = ParamValue::Choice(mode);
        }
        if let Some(q) = e.param_mut("method") {
            q.value = ParamValue::Choice(TrackMethod::ALL.iter().position(|m| *m == method).unwrap_or(0) as u32);
        }
        st.selected_mask = Some(MaskSel { clip, effect: ei, mask: 0 });
        Ok(())
    })?;
    Ok(json!({"clip": clip.0, "effect": ei, "mask": 0, "mode": if mode == 0 { "auto" } else { "manual" }}))
}

fn attach(s: &mut Session, p: &Value) -> Result<Value> {
    let (clip, ei) = target(s, p, "tracking.attach")?;
    let follower = clip_p(p, "follower").ok_or_else(|| bad("tracking.attach", "need `follower` (the clip that should follow)"))?;
    if follower == clip {
        return Err(bad("tracking.attach", "a clip can't follow its own track"));
    }
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    let (_, tr) = seq.find_item(clip).ok_or_else(|| bad("tracking.attach", "no such clip"))?;
    let (_, fol) = seq.find_item(follower).ok_or_else(|| bad("tracking.attach", "no such follower clip"))?;
    let (e, mask) = region(tr, ei)?;
    if mask.path.keyframes.is_empty() {
        return Err(bad("tracking.attach", "track the region first (nothing tracked yet)"));
    }
    let ph = time_p(s, p, "").unwrap_or(s.playhead());
    // the reference moment: where the follower sits is "home", wherever the playhead is
    let at_tr = tr.effect_time_at(ph.clamp(tr.start, tr.end() - Tick(1)));
    let m = fol.effect("motion").ok_or_else(|| bad("tracking.attach", "the follower has no Motion"))?;
    let mt_f = fol.effect_time_at(ph.clamp(fol.start, fol.end() - Tick(1)));
    let pos = m.vec2_at("position", mt_f);
    let pos = if pos.x.is_nan() || pos.y.is_nan() { Vec2::new(seq.settings.width as f64 / 2.0, seq.settings.height as f64 / 2.0) } else { pos };
    let link = Link { clip: follower, pos, rot: m.f64_at("rotation", mt_f), scale: m.f64_at("scale", mt_f), at: at_tr };
    let mut links = links_of(e);
    links.retain(|l| l.clip != follower);
    links.push(link);
    let pl = plan(s, clip, ei, links)?;
    let frames = commit(s, "Attach to Tracking", pl)?;
    Ok(json!({"clip": clip.0, "effect": ei, "follower": follower.0, "frames": frames}))
}

fn update(s: &mut Session, p: &Value) -> Result<Value> {
    let (clip, ei) = target(s, p, "tracking.update")?;
    let links = {
        let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
        let (_, tr) = seq.find_item(clip).ok_or_else(|| bad("tracking.update", "no such clip"))?;
        links_of(region_effect(tr, ei)?)
    };
    if links.is_empty() {
        return Ok(json!({"clip": clip.0, "effect": ei, "frames": 0}));
    }
    let pl = plan(s, clip, ei, links)?;
    let frames = commit(s, "Update Tracked Clips", pl)?;
    Ok(json!({"clip": clip.0, "effect": ei, "frames": frames}))
}

fn region_effect(it: &TrackItem, effect: usize) -> Result<&filmcraft_project::EffectInstance> {
    it.effects.get(effect).filter(|e| e.effect == EFFECT).ok_or_else(|| bad("tracking", "that effect is not Tracking Motion"))
}

fn detach(s: &mut Session, p: &Value) -> Result<Value> {
    let (clip, ei) = target(s, p, "tracking.detach")?;
    let follower = clip_p(p, "follower").ok_or_else(|| bad("tracking.detach", "need `follower`"))?;
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let link = {
        let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
        let (_, tr) = seq.find_item(clip).ok_or_else(|| bad("tracking.detach", "no such clip"))?;
        links_of(region_effect(tr, ei)?).into_iter().find(|l| l.clip == follower)
    }
    .ok_or_else(|| bad("tracking.detach", "that clip is not attached"))?;
    s.edit("Detach from Tracking", move |pr, _| {
        let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
        if let Some((_, it)) = q.find_item_mut(follower)
            && let Some(m) = it.effect_mut("motion")
        {
            // back to the values it had when it was attached
            for (pid, v) in [("position", ParamValue::Vec2(link.pos)), ("rotation", ParamValue::Float(link.rot)), ("scale", ParamValue::Float(link.scale))] {
                if let Some(p) = m.param_mut(pid) {
                    p.keyframes.clear();
                    p.value = v;
                }
            }
        }
        let (_, tr) = q.find_item_mut(clip).ok_or(filmcraft_edit::EditError::NoItem(clip))?;
        let e = tr.effects.get_mut(ei).ok_or_else(|| bad("tracking.detach", "the effect is gone"))?;
        let rest: Vec<Link> = links_of(e).into_iter().filter(|l| l.clip != follower).collect();
        if let Some(p) = e.param_mut("links") {
            p.value = ParamValue::Text(write_links(&rest));
        }
        Ok(())
    })?;
    Ok(json!({"clip": clip.0, "effect": ei, "follower": follower.0}))
}

/// The clips attached to a Tracking Motion effect (Effect Controls lists them).
pub fn attached(e: &filmcraft_project::EffectInstance) -> Vec<ClipId> {
    links_of(e).into_iter().map(|l| l.clip).collect()
}

/// Called when a mask-tracking job has finished: if its mask is a Tracking Motion region with
/// attached clips, they follow the new track.
pub fn on_track_finished(s: &mut Session, target: MaskSel) {
    let linked = s
        .active_sequence()
        .and_then(|q| q.find_item(target.clip))
        .and_then(|(_, it)| it.effects.get(target.effect))
        .is_some_and(|e| e.effect == EFFECT && !links_of(e).is_empty());
    if !linked {
        return;
    }
    if let Err(e) = update(s, &json!({"clip": target.clip.0, "effect": target.effect})) {
        s.error_toast("tracking.update", e.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(c: Vec2, half: f64, rot_deg: f64) -> Vec<Vec2> {
        let (s, k) = rot_deg.to_radians().sin_cos();
        [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].iter().map(|(x, y)| Vec2::new(c.x + half * (x * k - y * s), c.y + half * (x * s + y * k))).collect()
    }

    #[test]
    fn links_round_trip_and_damaged_entries_are_dropped() {
        let a = Link { clip: ClipId(7), pos: Vec2::new(960.5, 540.0), rot: -12.25, scale: 150.0, at: Tick(254_016_000_000) };
        let b = Link { clip: ClipId(9), pos: Vec2::new(0.0, 1.0), rot: 0.0, scale: 100.0, at: Tick(0) };
        let text = write_links(&[a.clone(), b.clone()]);
        assert_eq!(parse_links(&text), vec![a.clone(), b.clone()]);
        assert_eq!(parse_links(&format!("junk;{};1,2,3;{}", a.encode(), b.encode())), vec![a, b]);
        assert!(parse_links("").is_empty());
        assert!(Link::decode("1,NaN,2,3,4,5").is_none());
    }

    #[test]
    fn relative_pose_recovers_shift_rotation_and_scale() {
        let start = square(Vec2::new(100.0, 100.0), 20.0, 0.0);
        let r = relative(&start, &square(Vec2::new(130.0, 90.0), 20.0, 0.0)).unwrap_or_else(|| panic!("no pose"));
        assert!((r.center.x - 130.0).abs() < 1e-9 && (r.center.y - 90.0).abs() < 1e-9);
        assert!(r.rot_deg.abs() < 1e-9 && (r.scale - 1.0).abs() < 1e-9);
        let r = relative(&start, &square(Vec2::new(100.0, 100.0), 40.0, 30.0)).unwrap_or_else(|| panic!("no pose"));
        assert!((r.rot_deg - 30.0).abs() < 1e-9, "{r:?}");
        assert!((r.scale - 2.0).abs() < 1e-9, "{r:?}");
    }

    #[test]
    fn relative_pose_refuses_mismatched_or_degenerate_shapes() {
        let sq = square(Vec2::new(0.0, 0.0), 10.0, 0.0);
        assert!(relative(&sq, &sq[..3]).is_none());
        assert!(relative(&[Vec2::ZERO], &[Vec2::ZERO]).is_none());
        let dot = [Vec2::new(5.0, 5.0); 4];
        assert!(relative(&dot, &sq).is_none());
        let nan = [Vec2::new(f64::NAN, 0.0), Vec2::ZERO, Vec2::ZERO, Vec2::ZERO];
        assert!(relative(&sq, &nan).is_none());
    }

    #[test]
    fn follower_offset_turns_and_grows_with_the_region() {
        let o = orbit(Vec2::new(100.0, 0.0), 90.0, 2.0);
        assert!(o.x.abs() < 1e-9 && (o.y - 200.0).abs() < 1e-9, "{o:?}");
        let same = orbit(Vec2::new(30.0, -40.0), 0.0, 1.0);
        assert!((same.x - 30.0).abs() < 1e-9 && (same.y + 40.0).abs() < 1e-9);
        assert_eq!(orbit(Vec2::ZERO, 45.0, 3.0), Vec2::ZERO);
    }

    #[test]
    fn centred_box_is_centred() {
        let b = centred_box((1920.0, 1080.0));
        let c = centroid(&vertices(&b)).unwrap_or_default();
        assert!((c.x - 960.0).abs() < 1e-9 && (c.y - 540.0).abs() < 1e-9);
    }
}
