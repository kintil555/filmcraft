# Plugins

FilmCraft loads **declarative plugins** from `<data dir>/plugins/<id>/plugin.json` at start-up
(Edit > Preferences > Plugins, or the `plugins.reload` command, rescans the folder). A plugin is data, not
native code: `unsafe` and dynamic loading are forbidden (AGENTS.md §0.3), so a plugin cannot crash or hijack
the editor. Every file is untrusted and size-capped; a broken plugin is shown with its error and skipped.

```json
{ "schema": 1, "id": "my-plugin", "name": "My Plugin", "version": "1.0.0",
  "description": "What it does.", "presets": ["presets.json"] }
```

- `id`: 2-48 chars, `a-z 0-9 -`; the folder name must be the id.
- `presets`: a file name or a list of `filmcraft.effect-presets` files (the format Presets > Import reads),
  plain `*.json` names in the plugin folder, 1 MB each. Their presets show up in Effects > Presets (read-only).
- `patchedPresets` (optional): used instead of `presets` in this fork, which has the `inner_highlight` effect built in.
- Unknown keys are ignored, so one `plugin.json` can also carry the ArtCraft launcher's fields.

Commands (also over MCP): `plugins.list`, `plugins.reload`, `plugins.setEnabled {id, enabled}`.
Disabled ids are kept in preferences (`plugins.disabled`).

New effects are added to FilmCraft itself (`crates/project/src/effect.rs` + `crates/render/src/effects.rs`);
`inner_highlight` (Blend Mode, Opacity, Softness, Edge Thickness, Light Direction, Color) is the first one.
