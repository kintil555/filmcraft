# Handoff: planar tracker + depth (2026-11-01)

State of the work on `feat/neural-depth-planar` (on top of `main` at `2a5b065`). **Nothing here was
built, linted or tested** (the maintainer asked for no builds in that session): expect compile or
clippy fixes on the first `cargo xtask ci`.

## Already on `main` (`2a5b065`)
`crates/render/src/track/planar.rs`: coverage-aware model (homography → similarity → shift),
off-canvas coasting (`MAX_COAST_OFF_CANVAS`), parallax `DepthSample`s, `PlanarOptions`
(`depth_strength`), effect param `depth` ("Depth Detection", 0–100 %) on `tracking_motion`, wired in
`engine/src/masks.rs` (`track`).

## On the branch (`224573b`)
- New L2 crate `crates/depth` (`DepthMap`, `Estimator`, feature `depth-anything` = Depth Anything V2
  Small on candle). Registered in root `Cargo.toml`, `xtask/src/main.rs` (layer 2), `docs/architecture.md`.
- `PlanarTracker::advance_with_depth` / `set_depth`, `AnyTracker` twins; `depth_keep` fits one depth
  plane (inverse depth is affine across a plane) and drops off-plane matches. `advance` was split
  into `frame_motion` + `refine` for the cognitive-complexity limit (30).
- `masks.rs`: `depth_estimator()` reads `FILMCRAFT_DEPTH_MODEL` (folder with `dinov2_vits14.safetensors`
  and `depth_anything_v2_vits.safetensors`); failure only adds a status note.

## To do first
1. **Regenerate `Cargo.lock`** (new optional deps: candle-transformers 0.9). Run `cargo metadata`, commit it.
2. `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`,
   `cargo xtask layers`, `cargo xtask wasm` (new crate must build for wasm with default features).
3. Build once with `--features depth-anything` (engine and app) and fix candle API drift. The API was
   read from candle-transformers 0.9.1 source; `VarBuilder::from_buffered_safetensors` is used
   because `unsafe` is forbidden.
4. Check the weights' file names and repos with a real download (Hugging Face was blocked from the
   sandbox): `lmz/candle-dino-v2`, `jeroenvlek/depth-anything-v2-safetensors`. Confirm the output
   length is 518×518 (`estimate` errors otherwise). Only the Small model is Apache-2.0.
5. Check that the Tracking Motion panel shows the new **Depth Detection** slider; the panel in
   `ui-egui/src/panels/effect_controls.rs` is custom and was not looked at.

## Known gaps
- The new tests (off-canvas, sparse coverage, depth plane) have never run; if `leaving_the_canvas_keeps_the_track`
  is flaky, check the thresholds before the logic.
- No model download or pinned SHA-256 (the speech crate has that pattern, `speech::models`).
- Depth runs on every tracked frame (~1 s each on CPU); a stride option would help.
- Rules that bit this work: never-crash (no `unwrap`/indexing outside tests), `too_many_arguments`
  and `cognitive_complexity` are deny, commits end with the attribution lines.
