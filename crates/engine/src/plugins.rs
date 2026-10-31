//! Plugin host: declarative plugins loaded from `<data dir>/plugins/<id>/plugin.json`.
//!
//! A plugin is a folder, never code that runs inside FilmCraft (`unsafe` is forbidden, AGENTS.md §0.3).
//! It can ship effect-preset files (`filmcraft.effect-presets`, the same files Presets ▸ Import reads),
//! which appear in Effects ▸ Presets while the plugin is enabled. Every file is untrusted: sizes are
//! capped, names are plain, and a broken plugin is reported (`error`) instead of failing the load.
//!
//! ```json
//! { "schema": 1, "id": "my-plugin", "name": "My Plugin", "version": "1.0.0",
//!   "description": "…", "presets": ["presets.json"] }
//! ```
//! `presets` may be one file name or a list. `patchedPresets` (optional) replaces `presets` in builds
//! that carry the effects it uses (this fork has `inner_highlight` built in).
//!
//! Commands: `plugins.list`, `plugins.reload`, `plugins.setEnabled`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, str_p};
use crate::presets::{EffectPreset, parse_file};
use crate::{Result, Session};

pub const MANIFEST: &str = "plugin.json";
const PLUGIN_DIR: &str = "plugins";
const MAX_MANIFEST_BYTES: u64 = 64 * 1024;
const MAX_PRESET_BYTES: u64 = 1024 * 1024;
const MAX_PLUGINS: usize = 64;
const MAX_FILES: usize = 16;

fn one_or_many<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Vec<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum V {
        One(String),
        Many(Vec<String>),
    }
    Ok(match V::deserialize(d)? {
        V::One(s) => vec![s],
        V::Many(v) => v,
    })
}

/// A plugin's `plugin.json` (unknown keys, such as the launcher's, are ignored).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct PluginManifest {
    pub schema: u32,
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: String,
    #[serde(deserialize_with = "one_or_many")]
    pub presets: Vec<String>,
    pub patched_presets: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Plugin {
    pub manifest: PluginManifest,
    pub dir: PathBuf,
    pub presets: Vec<EffectPreset>,
    /// Why the plugin could not be loaded (it then contributes nothing).
    pub error: Option<String>,
}

impl Plugin {
    /// The manifest id, or the folder name when the manifest could not be read.
    pub fn id(&self) -> String {
        if self.manifest.id.is_empty() { self.dir.file_name().and_then(|n| n.to_str()).unwrap_or("?").to_string() } else { self.manifest.id.clone() }
    }
}

#[derive(Default)]
pub struct PluginHost {
    dir: Option<PathBuf>,
    pub plugins: Vec<Plugin>,
}

pub fn valid_id(id: &str) -> bool {
    let ok_len = (2..=48).contains(&id.len());
    let ok_chars = id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
    ok_len && ok_chars && !id.starts_with('-')
}

fn valid_file(name: &str) -> bool {
    name.len() <= 64
        && name.ends_with(".json")
        && !name.starts_with('.')
        && !name.contains("..")
        && name != MANIFEST
        && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

fn read_limited(path: &Path, max: u64) -> std::result::Result<Vec<u8>, String> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let meta = std::fs::metadata(path).map_err(|e| format!("{name}: {e}"))?;
    if !meta.is_file() {
        return Err(format!("{name} is not a file"));
    }
    if meta.len() > max {
        return Err(format!("{name} is larger than {} KB", max / 1024));
    }
    std::fs::read(path).map_err(|e| format!("{name}: {e}"))
}

fn fill(plugin: &mut Plugin) -> std::result::Result<(), String> {
    let bytes = read_limited(&plugin.dir.join(MANIFEST), MAX_MANIFEST_BYTES)?;
    let m: PluginManifest = serde_json::from_slice(&bytes).map_err(|e| format!("{MANIFEST}: {e}"))?;
    if m.schema != 1 {
        return Err(format!("schema must be 1 (got {})", m.schema));
    }
    if !valid_id(&m.id) {
        return Err("id must be 2-48 characters: lowercase letters, digits and '-'".into());
    }
    if plugin.dir.file_name().and_then(|n| n.to_str()) != Some(m.id.as_str()) {
        return Err(format!("the folder name must match the id `{}`", m.id));
    }
    let files: Vec<String> = match &m.patched_presets {
        Some(p) => vec![p.clone()],
        None => m.presets.clone(),
    };
    if files.len() > MAX_FILES {
        return Err(format!("at most {MAX_FILES} preset files"));
    }
    let mut presets: Vec<EffectPreset> = Vec::new();
    for file in &files {
        if !valid_file(file) {
            return Err(format!("`{file}` is not a plain .json file name inside the plugin folder"));
        }
        let bytes = read_limited(&plugin.dir.join(file), MAX_PRESET_BYTES)?;
        let mut list = parse_file(&bytes).map_err(|e| format!("{file}: {e}"))?;
        presets.append(&mut list);
    }
    for p in &mut presets {
        // Plugin presets are read-only in the library (not saved to the user's file, can't be deleted).
        p.builtin = true;
    }
    plugin.manifest = m;
    plugin.presets = presets;
    Ok(())
}

fn load_one(dir: &Path) -> Plugin {
    let mut plugin = Plugin { manifest: PluginManifest::default(), dir: dir.to_path_buf(), presets: Vec::new(), error: None };
    if let Err(e) = fill(&mut plugin) {
        log::warn!("plugin {}: {e}", dir.display());
        plugin.presets.clear();
        plugin.error = Some(e);
    }
    plugin
}

impl PluginHost {
    pub fn set_dir(&mut self, data_dir: &Path) {
        self.dir = Some(data_dir.join(PLUGIN_DIR));
    }

    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    /// Rescan the plugins folder (created if missing).
    pub fn reload(&mut self) {
        self.plugins.clear();
        let Some(root) = self.dir.clone() else { return };
        let _ = std::fs::create_dir_all(&root);
        let Ok(entries) = std::fs::read_dir(&root) else { return };
        let mut dirs: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir() && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| !n.starts_with('.')))
            .collect();
        dirs.sort();
        dirs.truncate(MAX_PLUGINS);
        self.plugins = dirs.iter().map(|d| load_one(d)).collect();
    }

    /// Presets of the loaded, enabled plugins (first plugin wins a name clash).
    pub fn presets(&self, disabled: &[String]) -> Vec<EffectPreset> {
        let mut out: Vec<EffectPreset> = Vec::new();
        for p in &self.plugins {
            if p.error.is_some() || disabled.iter().any(|d| *d == p.manifest.id) {
                continue;
            }
            for preset in &p.presets {
                if !out.iter().any(|o| o.name == preset.name) {
                    out.push(preset.clone());
                }
            }
        }
        out
    }
}

impl Session {
    /// Rescan plugins and refresh the plugin presets in the library.
    pub fn apply_plugins(&mut self) {
        self.plugins.reload();
        self.presets.plugin = self.plugins.presets(&self.prefs.plugins.disabled);
    }
}

// ---------------------------------------------------------------- commands

type Run = fn(&mut Session, &Value) -> Result<Value>;

fn spec(id: &'static str, label: &'static str, params: &'static str, run: Run, journal: bool) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled: always, run, journal }
}

pub(crate) fn commands() -> Vec<CommandSpec> {
    vec![
        spec("plugins.list", "List Plugins", "{}", list, false),
        spec("plugins.reload", "Reload Plugins", "{}", reload, false),
        spec("plugins.setEnabled", "Enable or Disable a Plugin", r#"{"id":str,"enabled":bool}"#, set_enabled, false),
    ]
}

fn list_json(s: &Session) -> Value {
    let plugins: Vec<Value> = s
        .plugins
        .plugins
        .iter()
        .map(|p| {
            let id = p.id();
            json!({
                "id": id,
                "name": if p.manifest.name.is_empty() { id.clone() } else { p.manifest.name.clone() },
                "version": p.manifest.version,
                "description": p.manifest.description,
                "enabled": !s.prefs.plugins.disabled.contains(&id),
                "presets": p.presets.len(),
                "error": p.error,
            })
        })
        .collect();
    json!({
        "dir": s.plugins.dir().map(|d| d.display().to_string()),
        "developerMode": s.prefs.plugins.developer_mode,
        "plugins": plugins,
    })
}

fn list(s: &mut Session, _: &Value) -> Result<Value> {
    Ok(list_json(s))
}

fn reload(s: &mut Session, _: &Value) -> Result<Value> {
    s.apply_plugins();
    Ok(list_json(s))
}

fn set_enabled(s: &mut Session, p: &Value) -> Result<Value> {
    let id = str_p(p, "id").ok_or_else(|| bad("plugins.setEnabled", "need an `id`"))?.to_string();
    let enabled = p.get("enabled").and_then(Value::as_bool).ok_or_else(|| bad("plugins.setEnabled", "need `enabled` (true or false)"))?;
    if !s.plugins.plugins.iter().any(|pl| pl.id() == id) {
        return Err(bad("plugins.setEnabled", format!("no plugin `{id}` (try plugins.reload)")));
    }
    s.prefs.plugins.disabled.retain(|d| *d != id);
    if !enabled {
        s.prefs.plugins.disabled.push(id);
    }
    s.presets.plugin = s.plugins.presets(&s.prefs.plugins.disabled);
    if let Some(path) = s.prefs_path.clone() {
        if let Err(e) = s.prefs.save(&path) {
            log::warn!("could not save preferences: {e}");
        }
    }
    Ok(list_json(s))
}
