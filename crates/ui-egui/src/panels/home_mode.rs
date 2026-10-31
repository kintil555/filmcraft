//! Home screen (header Home button, and the first screen at launch when Settings ▸ General ▸ At
//! Startup is "Show Home"). Left: New Project, Open Project and the Home entry, with the Auto Save
//! switch at the bottom. Right: the Recent projects table (name, last opened, size, type) with a
//! filter field. Opening or creating a project switches to Edit.
//!
//! Replacing a project with unsaved changes goes through File ▸ Close Project's Save / Don't Save /
//! Cancel prompt first, so Home never discards work.
//!
//! Automation ids: `home.new`, `home.open`, `home.nav`, `home.autosave`, `home.save`, `home.filter`,
//! `home.new.*` (the New Project dialog, see `file_dialogs::show_new_project`), `home.recent.<n>`, `home.recent.<n>.remove`.

use egui::{Align2, Color32, Rect, Sense, Shape, Stroke, pos2, vec2};
use serde_json::json;

use crate::FilmcraftApp;
use crate::icons::{self, Icon};
use crate::state::Mode;
use crate::theme::Tokens;

/// Most recent projects read for the table.
const MAX_ROWS: usize = 100;
const ROW_H: f32 = 52.0;
const SIDEBAR_W: f32 = 240.0;
/// How often (s) the table re-checks the files on disk. Reading metadata for every row on every
/// frame would be wasteful; this keeps Home free while it sits open.
const REFRESH_SECS: f64 = 3.0;

/// What Home knows about one recent project's file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct FileInfo {
    /// `false` only when the file is known to be gone (an unreadable location counts as present).
    exists: bool,
    /// Seconds since it was last modified.
    age_secs: Option<u64>,
    /// File size in bytes.
    size: Option<u64>,
}

#[derive(Clone, Default)]
struct RecentCache {
    at: f64,
    paths: Vec<String>,
    info: Vec<FileInfo>,
}

fn file_info(path: &str) -> FileInfo {
    match std::fs::metadata(path) {
        Ok(m) => FileInfo { exists: true, age_secs: m.modified().ok().and_then(|t| t.elapsed().ok()).map(|d| d.as_secs()), size: Some(m.len()) },
        Err(e) => FileInfo { exists: e.kind() != std::io::ErrorKind::NotFound, age_secs: None, size: None },
    }
}

/// The cached file info for `paths`, refreshed when the list changed or the cache is stale.
fn recent_info(ctx: &egui::Context, paths: &[String]) -> Vec<FileInfo> {
    let id = egui::Id::unique("home-recent-cache");
    let now = ctx.input(|i| i.time);
    let cache: RecentCache = ctx.data(|d| d.get_temp(id)).unwrap_or_default();
    if cache.paths == paths && cache.info.len() == paths.len() && now - cache.at < REFRESH_SECS {
        return cache.info;
    }
    let info: Vec<FileInfo> = paths.iter().map(|p| file_info(p)).collect();
    ctx.data_mut(|d| d.insert_temp(id, RecentCache { at: now, paths: paths.to_vec(), info: info.clone() }));
    info
}

/// "just now", "5 min ago", "3 h ago", "yesterday", "3 months ago", …
fn age_text(secs: u64) -> String {
    let (mins, hours, days) = (secs / 60, secs / 3600, secs / 86_400);
    if mins < 1 {
        tl!("just now").to_string()
    } else if hours < 1 {
        tlf!("{n} min ago", n = mins)
    } else if days < 1 {
        tlf!("{n} h ago", n = hours)
    } else if days == 1 {
        tl!("yesterday").to_string()
    } else if days < 30 {
        tlf!("{n} days ago", n = days)
    } else if days < 365 {
        tlf!("{n} months ago", n = days / 30)
    } else {
        tlf!("{n} years ago", n = days / 365)
    }
}

/// "17 KB", "2.4 MB"; sizes are shown in KB from the first byte, as the file manager does.
fn size_text(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    if bytes >= MB {
        let tenths = bytes.saturating_mul(10) / MB;
        format!("{}.{} MB", tenths / 10, tenths % 10)
    } else {
        format!("{} KB", bytes.div_ceil(KB).max(1))
    }
}

/// A rounded button at `r`; returns whether it was clicked.
fn pill(app: &mut FilmcraftApp, ui: &mut egui::Ui, id: &str, r: Rect, label: &str, primary: bool) -> bool {
    let t = app.tokens;
    let resp = ui.interact(r, egui::Id::unique(("home", id)), Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand);
    app.auto.add(id, r, label);
    let (bg, fg) = if primary {
        (if resp.hovered() { t.accent_hover } else { t.accent }, Color32::WHITE)
    } else {
        (if resp.hovered() { t.hover } else { t.tl_header_bg }, t.text)
    };
    ui.painter().rect_filled(r, r.height() / 2.0, bg);
    let font = if primary { Tokens::semibold(13.0) } else { Tokens::ui(13.0) };
    ui.painter().text(r.center(), Align2::CENTER_CENTER, label, font, fg);
    resp.clicked()
}

/// If the open project has unsaved changes, ask what to do with them (File ▸ Close Project's
/// prompt) and report that the caller must not replace it yet.
fn blocked_by_unsaved(app: &mut FilmcraftApp, ctx: &egui::Context) -> bool {
    if !app.session.is_dirty() {
        return false;
    }
    app.ui.status = tl!("Save or close the current project first.").to_string();
    let _ = crate::menus::invoke(app, ctx, "file.closeProject", json!({}));
    true
}

fn open_path(app: &mut FilmcraftApp, ctx: &egui::Context, path: &str) {
    if blocked_by_unsaved(app, ctx) {
        return;
    }
    if crate::menus::invoke(app, ctx, "file.open", json!({"path": path})).is_ok() {
        app.ui.mode = Mode::Edit;
    }
}

fn set_auto_save(app: &mut FilmcraftApp, on: bool) {
    let mut next = app.session.prefs.clone();
    next.auto_save.enabled = on;
    if let Err(e) = app.session.set_prefs(next) {
        app.ui.status = e.to_string();
    }
}

fn forget_recent(app: &mut FilmcraftApp, path: &str) {
    let mut next = app.session.prefs.clone();
    next.general.recent_projects.retain(|r| r != path);
    if let Err(e) = app.session.set_prefs(next) {
        app.ui.status = e.to_string();
    }
}

pub fn show(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect) {
    let t = app.tokens;
    ui.painter().rect_filled(rect, t.radius, t.panel_bg);
    let side = Rect::from_min_size(rect.min, vec2(SIDEBAR_W.min(rect.width()), rect.height()));
    let main = Rect::from_min_max(pos2((side.max.x + 24.0).min(rect.max.x), rect.min.y), pos2((rect.max.x - 24.0).max(side.max.x), rect.max.y));
    sidebar(app, ui, side);
    ui.painter().line_segment([side.right_top(), side.right_bottom()], Stroke::new(1.0, t.separator));
    main_area(app, ui, main);
    crate::panels::file_dialogs::show_new_project(app, ui.ctx());
}

/// New Project, Open Project, the Home entry, and Auto Save at the bottom.
fn sidebar(app: &mut FilmcraftApp, ui: &mut egui::Ui, side: Rect) {
    let t = app.tokens;
    let ctx = ui.ctx().clone();
    let (x, w) = (side.min.x + 16.0, (side.width() - 32.0).max(0.0));

    let new = Rect::from_min_size(pos2(x, side.min.y + 20.0), vec2(w.min(110.0), 30.0));
    if pill(app, ui, "home.new", new, tl!("New Project"), true) && !blocked_by_unsaved(app, &ctx) {
        app.file_dialogs.home_new = Some(crate::panels::file_dialogs::NewProjectForm::default());
        app.file_dialogs.home_new_focus = true;
    }

    let open = Rect::from_min_size(pos2(x, side.min.y + 58.0), vec2(w, 28.0));
    let resp = ui.interact(open, egui::Id::unique(("home", "home.open")), Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand);
    app.auto.add("home.open", open, tl!("Open Project"));
    if resp.hovered() {
        ui.painter().rect_filled(open, 4.0, t.hover);
    }
    ui.painter().text(open.min + vec2(12.0, 14.0), Align2::LEFT_CENTER, tl!("Open Project"), Tokens::semibold(12.5), t.text);
    if resp.clicked() && !blocked_by_unsaved(app, &ctx) && crate::menus::invoke(app, &ctx, "file.open", json!({})).is_err() {
        app.ui.status = tl!("Could not open the file dialog.").to_string();
    }

    let rule_y = side.min.y + 98.0;
    ui.painter().line_segment([pos2(x, rule_y), pos2(x + w, rule_y)], Stroke::new(1.0, t.separator));

    // The Home entry is always the selected one: it is the only page.
    let home = Rect::from_min_size(pos2(x, rule_y + 12.0), vec2(w, 30.0));
    ui.painter().rect_filled(home, 4.0, t.hover);
    icons::paint(ui.painter(), Rect::from_center_size(pos2(home.min.x + 18.0, home.center().y), vec2(14.0, 14.0)), Icon::Home, t.text);
    ui.painter().text(pos2(home.min.x + 36.0, home.center().y), Align2::LEFT_CENTER, tl!("Home"), Tokens::ui(12.5), t.text);
    app.auto.add("home.nav", home, tl!("Home"));

    // Auto Save (off by default; the worker only copies a changed project, off the UI thread).
    let mut on = app.session.prefs.auto_save.enabled;
    let box_r = Rect::from_min_size(pos2(x, (side.max.y - 44.0).max(home.max.y + 12.0)), vec2(w, 24.0));
    let label = tlf!("Auto Save every {n} min", n = app.session.prefs.auto_save.interval_minutes);
    let resp = ui.put(box_r, egui::Checkbox::new(&mut on, label.as_str()));
    let changed = resp.changed();
    app.auto.add_response("home.autosave", &resp, &label);
    resp.on_hover_text(tl!("Off by default. Saves a copy in the background, only when the project changed."));
    if changed {
        set_auto_save(app, on);
    }
}

/// The unsaved-changes notice and the Recent table (New Project opens its own dialog).
fn main_area(app: &mut FilmcraftApp, ui: &mut egui::Ui, main: Rect) {
    let t = app.tokens;
    let ctx = ui.ctx().clone();
    let mut y = main.min.y + 20.0;

    // Unsaved changes in the open project.
    if app.session.is_dirty() {
        let msg = tlf!("Unsaved changes in “{name}”.", name = app.session.project.name);
        let g = ui.painter().layout_no_wrap(msg, Tokens::ui(13.0), t.text);
        let w = g.size().x;
        ui.painter().galley(pos2(main.min.x, y + 15.0 - g.size().y / 2.0), g, t.text);
        let save = Rect::from_min_size(pos2(main.min.x + w + 16.0, y), vec2(88.0, 30.0));
        if pill(app, ui, "home.save", save, tl!("Save"), false) {
            let _ = crate::menus::invoke(app, &ctx, "file.save", json!({}));
        }
        y += 44.0;
    }

    recent_table(app, ui, main, y + 4.0);
}

/// "Recent": the filter field, the column headers, and one row per project (newest first). A
/// click opens a project; × (on hover) removes it from the list.
fn recent_table(app: &mut FilmcraftApp, ui: &mut egui::Ui, main: Rect, top: f32) {
    let t = app.tokens;
    let ctx = ui.ctx().clone();
    let w = main.width();
    ui.painter().text(pos2(main.min.x, top + 16.0), Align2::LEFT_CENTER, tl!("Recent"), Tokens::semibold(16.0), t.text);

    // Filter field, right-aligned on the title line.
    let field = Rect::from_min_size(pos2((main.max.x - 220.0).max(main.min.x + 120.0), top + 2.0), vec2(220.0_f32.min(w), 28.0));
    ui.painter().text(pos2(field.min.x - 8.0, field.center().y), Align2::RIGHT_CENTER, tl!("Filter"), Tokens::ui(11.5), t.text_dim);
    let mut filter = std::mem::take(&mut app.file_dialogs.home_filter);
    let edit = ui.put(field, egui::TextEdit::singleline(&mut filter).hint_text(tl!("Filter Recent Files")).desired_width(field.width()).char_limit(120));
    app.auto.add_response("home.filter", &edit, tl!("Filter Recent Files"));
    app.file_dialogs.home_filter = filter;

    // Column positions: name | recent | size | type.
    let (c_recent, c_size_right, c_type) = (main.min.x + w * 0.46, main.min.x + w * 0.80, main.min.x + w * 0.84);
    let head_y = top + 56.0;
    let small = Tokens::ui(10.5);
    ui.painter().text(pos2(main.min.x + 12.0, head_y), Align2::LEFT_CENTER, tl!("NAME"), small.clone(), t.text_dim);
    let r = ui.painter().text(pos2(c_recent, head_y), Align2::LEFT_CENTER, tl!("RECENT"), small.clone(), t.text_dim);
    // sort arrow: newest first
    let a = pos2(r.max.x + 10.0, head_y);
    ui.painter().add(Shape::convex_polygon(vec![a + vec2(-3.5, -2.0), a + vec2(3.5, -2.0), a + vec2(0.0, 3.0)], t.text_dim, Stroke::NONE));
    ui.painter().text(pos2(c_size_right, head_y), Align2::RIGHT_CENTER, tl!("SIZE"), small.clone(), t.text_dim);
    ui.painter().text(pos2(c_type, head_y), Align2::LEFT_CENTER, tl!("TYPE"), small, t.text_dim);
    let rule = head_y + 14.0;
    ui.painter().line_segment([pos2(main.min.x, rule), pos2(main.max.x, rule)], Stroke::new(1.0, t.separator));

    let paths: Vec<String> = app.session.prefs.general.recent_projects.iter().take(MAX_ROWS).cloned().collect();
    let area = Rect::from_min_max(pos2(main.min.x, rule + 1.0), pos2(main.max.x, (main.max.y - 8.0).max(rule + 1.0)));
    if paths.is_empty() {
        ui.painter().text(
            pos2(main.min.x + 12.0, area.min.y + 30.0),
            Align2::LEFT_CENTER,
            tl!("No recent projects yet. Create or open one and it will show up here."),
            Tokens::ui(13.0),
            t.text_dim,
        );
        return;
    }
    let info = recent_info(&ctx, &paths);
    let needle = app.file_dialogs.home_filter.trim().to_lowercase();
    let (mut open, mut remove, mut missing) = (None, None, false);
    let mut body = ui.new_child(egui::UiBuilder::new().max_rect(area));
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(&mut body, |ui| {
        ui.spacing_mut().item_spacing = vec2(0.0, 0.0);
        for (i, path) in paths.iter().enumerate() {
            let name = crate::menus::recent_name(path);
            if !needle.is_empty() && !name.to_lowercase().contains(&needle) {
                continue;
            }
            let fi = info.get(i).copied().unwrap_or(FileInfo { exists: true, age_secs: None, size: None });
            let (row, resp) = ui.allocate_exact_size(vec2(area.width(), ROW_H), Sense::click());
            let resp = resp.on_hover_text(path.as_str());
            app.auto.add(&format!("home.recent.{i}"), row, &name);
            if resp.hovered() {
                ui.painter().rect_filled(row, 4.0, t.hover);
            }
            ui.painter().line_segment([row.left_bottom(), row.right_bottom()], Stroke::new(1.0, t.separator));
            let ink = if fi.exists { t.text } else { t.text_faint };
            let cy = row.center().y;
            let p = ui.painter().with_clip_rect(row);
            let name_clip = p.with_clip_rect(Rect::from_min_max(row.min, pos2((c_recent - 12.0).max(row.min.x), row.max.y)));
            name_clip.text(pos2(row.min.x + 12.0, cy), Align2::LEFT_CENTER, &name, Tokens::ui(13.0), ink);
            let when = match (fi.exists, fi.age_secs) {
                (false, _) => tl!("File not found").to_string(),
                (true, Some(s)) => age_text(s),
                (true, None) => String::new(),
            };
            p.text(pos2(c_recent, cy), Align2::LEFT_CENTER, when, Tokens::ui(11.5), t.text_dim);
            if let Some(bytes) = fi.size {
                p.text(pos2(c_size_right, cy), Align2::RIGHT_CENTER, size_text(bytes), Tokens::ui(11.5), t.text_dim);
            }
            p.text(pos2(c_type, cy), Align2::LEFT_CENTER, tl!("FilmCraft Project"), Tokens::ui(12.0), ink);
            let x_btn = Rect::from_center_size(pos2(row.max.x - 14.0, cy), vec2(20.0, 20.0));
            let x_resp = ui.interact(x_btn, egui::Id::unique(("home", "recent-x", i)), Sense::click());
            app.auto.add(&format!("home.recent.{i}.remove"), x_btn, tl!("Remove from list"));
            if resp.hovered() || x_resp.hovered() {
                let stroke = Stroke::new(1.5, if x_resp.hovered() { t.text } else { t.text_dim });
                let (m, s) = (x_btn.center(), 4.0);
                ui.painter().line_segment([m + vec2(-s, -s), m + vec2(s, s)], stroke);
                ui.painter().line_segment([m + vec2(-s, s), m + vec2(s, -s)], stroke);
            }
            if x_resp.clicked() {
                remove = Some(path.clone());
            } else if resp.clicked() {
                if fi.exists {
                    open = Some(path.clone());
                } else {
                    missing = true;
                }
            }
        }
    });
    if missing {
        app.ui.status = tl!("File not found").to_string();
    }
    if let Some(p) = remove {
        forget_recent(app, &p);
    } else if let Some(p) = open {
        open_path(app, &ctx, &p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn age_text_buckets() {
        assert_eq!(age_text(5), "just now");
        assert_eq!(age_text(5 * 60), "5 min ago");
        assert_eq!(age_text(3 * 3600), "3 h ago");
        assert_eq!(age_text(86_400 + 5), "yesterday");
        assert_eq!(age_text(4 * 86_400), "4 days ago");
        assert_eq!(age_text(95 * 86_400), "3 months ago");
        assert_eq!(age_text(800 * 86_400), "2 years ago");
    }

    #[test]
    fn size_text_rounds_up_to_kb_and_switches_to_mb() {
        assert_eq!(size_text(0), "1 KB");
        assert_eq!(size_text(17 * 1024), "17 KB");
        assert_eq!(size_text(17 * 1024 + 1), "18 KB");
        assert_eq!(size_text(2 * 1024 * 1024 + 512 * 1024), "2.5 MB");
        let _ = size_text(u64::MAX); // no overflow
    }

    #[test]
    fn missing_file_is_flagged_and_a_present_one_is_not() {
        assert!(!file_info("/definitely/not/here.fcproj").exists);
        let dir = std::env::temp_dir();
        assert!(file_info(&dir.to_string_lossy()).exists);
    }
}
