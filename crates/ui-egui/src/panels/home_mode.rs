//! Home screen (header Home button, and the first screen at launch when Settings ▸ General ▸ At
//! Startup is "Show Home"): New Project, Open Project…, the recent projects, and the Auto Save
//! switch. Opening or creating a project switches to Edit.
//!
//! Replacing a project with unsaved changes goes through File ▸ Close Project's Save / Don't Save /
//! Cancel prompt first, so Home never discards work.
//!
//! Automation ids: `home.new`, `home.open`, `home.back`, `home.save`, `home.autosave`,
//! `home.new.name`, `home.new.create`, `home.new.cancel`, `home.recent.<n>`, `home.recent.<n>.remove`,
//! `home.recent.clear`.

use egui::{Align2, Color32, Rect, Sense, pos2, vec2};
use serde_json::json;

use crate::FilmcraftApp;
use crate::icons::{self, Icon};
use crate::state::Mode;
use crate::theme::Tokens;

/// Most recent projects listed (fewer when the window is short).
const MAX_ROWS: usize = 12;
const ROW_H: f32 = 34.0;
/// How often (s) the recent list re-checks the files on disk. Reading metadata for a dozen paths
/// every frame would be wasteful; this keeps Home free while it sits open.
const REFRESH_SECS: f64 = 3.0;

/// What Home knows about one recent project's file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct FileInfo {
    /// `false` only when the file is known to be gone (an unreadable location counts as present).
    exists: bool,
    /// Seconds since it was last modified.
    age_secs: Option<u64>,
}

#[derive(Clone, Default)]
struct RecentCache {
    at: f64,
    paths: Vec<String>,
    info: Vec<FileInfo>,
}

fn file_info(path: &str) -> FileInfo {
    match std::fs::metadata(path) {
        Ok(m) => FileInfo { exists: true, age_secs: m.modified().ok().and_then(|t| t.elapsed().ok()).map(|d| d.as_secs()) },
        Err(e) => FileInfo { exists: e.kind() != std::io::ErrorKind::NotFound, age_secs: None },
    }
}

/// The cached file info for `paths`, refreshed when the list changed or the cache is stale.
fn recent_info(ctx: &egui::Context, paths: &[String]) -> Vec<FileInfo> {
    let id = egui::Id::new("home-recent-cache");
    let now = ctx.input(|i| i.time);
    let cache: RecentCache = ctx.data(|d| d.get_temp(id)).unwrap_or_default();
    if cache.paths == paths && cache.info.len() == paths.len() && now - cache.at < REFRESH_SECS {
        return cache.info;
    }
    let info: Vec<FileInfo> = paths.iter().map(|p| file_info(p)).collect();
    ctx.data_mut(|d| d.insert_temp(id, RecentCache { at: now, paths: paths.to_vec(), info: info.clone() }));
    info
}

/// "5 min ago", "3 h ago", …
fn age_text(secs: u64) -> String {
    let (mins, hours, days) = (secs / 60, secs / 3600, secs / 86_400);
    if mins < 1 {
        tl!("Just now").to_string()
    } else if hours < 1 {
        tlf!("{n} min ago", n = mins)
    } else if days < 1 {
        tlf!("{n} h ago", n = hours)
    } else if days < 365 {
        tlf!("{n} d ago", n = days)
    } else {
        tlf!("{n} y ago", n = days / 365)
    }
}

/// A rounded button at `r`; returns whether it was clicked.
fn pill(app: &mut FilmcraftApp, ui: &mut egui::Ui, id: &str, r: Rect, label: &str, primary: bool) -> bool {
    let t = app.tokens;
    let resp = ui.interact(r, egui::Id::new(("home", id)), Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand);
    app.auto.add(id, r, label);
    let (bg, fg) = if primary {
        (if resp.hovered() { t.accent_hover } else { t.accent }, Color32::WHITE)
    } else {
        (if resp.hovered() { t.hover } else { t.tl_header_bg }, t.text)
    };
    ui.painter().rect_filled(r, r.height() / 2.0, bg);
    ui.painter().text(r.center(), Align2::CENTER_CENTER, label, Tokens::ui(13.0), fg);
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

fn forget_recent(app: &mut FilmcraftApp, path: Option<&str>) {
    let mut next = app.session.prefs.clone();
    match path {
        Some(p) => next.general.recent_projects.retain(|r| r != p),
        None => next.general.recent_projects.clear(),
    }
    if let Err(e) = app.session.set_prefs(next) {
        app.ui.status = e.to_string();
    }
}

pub fn show(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect) {
    let t = app.tokens;
    let ctx = ui.ctx().clone();
    ui.painter().rect_filled(rect, t.radius, t.panel_bg);
    let left = rect.min.x + 24.0;
    ui.painter().text(rect.min + vec2(24.0, 30.0), Align2::LEFT_CENTER, tl!("Home"), Tokens::semibold(22.0), t.text);
    ui.painter().text(
        rect.min + vec2(24.0, 58.0),
        Align2::LEFT_CENTER,
        tl!("Start a new project, or pick up where you left off."),
        Tokens::ui(13.0),
        t.text_dim,
    );
    let back = Rect::from_min_size(pos2(rect.max.x - 24.0 - 150.0, rect.min.y + 16.0), vec2(150.0, 30.0));
    if pill(app, ui, "home.back", back, tl!("Back to editing"), false) {
        app.ui.mode = Mode::Edit;
        return;
    }
    let mut y = rect.min.y + 84.0;

    // Unsaved changes in the open project.
    if app.session.is_dirty() {
        let msg = tlf!("Unsaved changes in “{name}”.", name = app.session.project.name);
        let g = ui.painter().layout_no_wrap(msg, Tokens::ui(13.0), t.text);
        let w = g.size().x;
        ui.painter().galley(pos2(left, y + 15.0 - g.size().y / 2.0), g, t.text);
        let save = Rect::from_min_size(pos2(left + w + 16.0, y), vec2(88.0, 30.0));
        if pill(app, ui, "home.save", save, tl!("Save"), false) {
            let _ = crate::menus::invoke(app, &ctx, "file.save", json!({}));
        }
        y += 44.0;
    }

    // New / Open cards.
    let cards = [
        ("home.new", tl!("New Project"), tl!("An empty project with one sequence"), Icon::NewItem),
        ("home.open", tl!("Open Project…"), tl!("Choose a .fcproj file from disk"), Icon::Folder),
    ];
    let (cw, ch) = (260.0, 110.0);
    let mut chosen = None;
    for (i, (id, title, sub, icon)) in cards.iter().enumerate() {
        let r = Rect::from_min_size(pos2(left + i as f32 * (cw + 16.0), y), vec2(cw, ch));
        let resp = ui.interact(r, egui::Id::new(("home", *id)), Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand);
        app.auto.add(id, r, title);
        ui.painter().rect_filled(r, 8.0, if resp.hovered() { t.hover } else { t.tl_header_bg });
        icons::paint(ui.painter(), Rect::from_min_size(r.min + vec2(16.0, 14.0), vec2(40.0, 40.0)), *icon, Color32::from_rgb(140, 150, 255));
        ui.painter().text(r.min + vec2(16.0, 76.0), Align2::LEFT_CENTER, *title, Tokens::semibold(14.0), t.text);
        ui.painter().text(r.min + vec2(16.0, 96.0), Align2::LEFT_CENTER, *sub, Tokens::ui(11.5), t.text_dim);
        if resp.clicked() {
            chosen = Some(*id);
        }
    }
    y += ch + 16.0;
    match chosen {
        Some("home.new") if !blocked_by_unsaved(app, &ctx) => {
            app.file_dialogs.home_new_name = Some("Untitled".into());
            app.file_dialogs.home_new_focus = true;
        }
        Some("home.open") if !blocked_by_unsaved(app, &ctx) => {
            if crate::menus::invoke(app, &ctx, "file.open", json!({})).is_err() {
                app.ui.status = tl!("Could not open the file dialog.").to_string();
            }
        }
        _ => {}
    }

    // Name field for a new project.
    if let Some(mut name) = app.file_dialogs.home_new_name.take() {
        ui.painter().text(pos2(left, y + 15.0), Align2::LEFT_CENTER, tl!("Project name"), Tokens::ui(13.0), t.text_dim);
        let field = Rect::from_min_size(pos2(left + 110.0, y), vec2(260.0, 30.0));
        let edit = ui.put(field, egui::TextEdit::singleline(&mut name).desired_width(field.width()).char_limit(120));
        app.auto.add("home.new.name", field, tl!("Project name"));
        if std::mem::take(&mut app.file_dialogs.home_new_focus) {
            edit.request_focus();
        }
        let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        let create = Rect::from_min_size(pos2(field.max.x + 12.0, y), vec2(88.0, 30.0));
        let cancel = Rect::from_min_size(pos2(create.max.x + 8.0, y), vec2(88.0, 30.0));
        let mut keep = true;
        if pill(app, ui, "home.new.create", create, tl!("Create"), true) || enter {
            let trimmed = name.trim();
            let project = if trimmed.is_empty() { "Untitled" } else { trimmed };
            match crate::menus::invoke(app, &ctx, "file.newProject", json!({"name": project})) {
                Ok(_) => app.ui.mode = Mode::Edit,
                Err(e) => app.ui.status = e,
            }
            keep = false;
        } else if pill(app, ui, "home.new.cancel", cancel, tl!("Cancel"), false) || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            keep = false;
        }
        if keep {
            app.file_dialogs.home_new_name = Some(name);
        }
        y += 46.0;
    }

    // Auto Save (off by default; the worker only copies a changed project, off the UI thread).
    let mut on = app.session.prefs.auto_save.enabled;
    let box_r = Rect::from_min_size(pos2(left, y), vec2(420.0, 24.0));
    let label = tlf!("Auto Save every {n} min", n = app.session.prefs.auto_save.interval_minutes);
    let resp = ui.put(box_r, egui::Checkbox::new(&mut on, label.as_str()));
    app.auto.add_response("home.autosave", &resp, &label);
    if resp.changed() {
        set_auto_save(app, on);
    }
    ui.painter().text(
        pos2(left + 26.0, y + 34.0),
        Align2::LEFT_CENTER,
        tl!("Off by default. Saves a copy in the background, only when the project changed."),
        Tokens::ui(11.5),
        t.text_dim,
    );
    y += 64.0;

    recent(app, ui, rect, left, y);
}

/// The recent projects (newest first): a click opens one; × removes it from the list.
fn recent(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect, left: f32, top: f32) {
    let t = app.tokens;
    let ctx = ui.ctx().clone();
    ui.painter().text(pos2(left, top + 10.0), Align2::LEFT_CENTER, tl!("Recent projects"), Tokens::semibold(15.0), t.text);
    let paths: Vec<String> = app.session.prefs.general.recent_projects.clone();
    if paths.is_empty() {
        ui.painter().text(
            pos2(left, top + 40.0),
            Align2::LEFT_CENTER,
            tl!("No recent projects yet. Create or open one and it will show up here."),
            Tokens::ui(13.0),
            t.text_dim,
        );
        return;
    }
    let width = (rect.max.x - left - 24.0).clamp(0.0, 760.0);
    let clear = Rect::from_min_size(pos2(left + width - 120.0, top), vec2(120.0, 20.0));
    let clear_resp = ui.interact(clear, egui::Id::new(("home", "home.recent.clear")), Sense::click()).on_hover_cursor(egui::CursorIcon::PointingHand);
    app.auto.add("home.recent.clear", clear, tl!("Clear list"));
    ui.painter().text(
        clear.right_center(),
        Align2::RIGHT_CENTER,
        tl!("Clear list"),
        Tokens::ui(12.0),
        if clear_resp.hovered() { t.text } else { t.text_dim },
    );
    if clear_resp.clicked() {
        forget_recent(app, None);
        return;
    }
    let room = ((rect.max.y - top - 40.0) / ROW_H).floor();
    let rows = if room.is_finite() && room >= 1.0 { (room as usize).min(MAX_ROWS) } else { 1 };
    let shown: Vec<String> = paths.into_iter().take(rows).collect();
    let info = recent_info(&ctx, &shown);
    let (mut open, mut remove) = (None, None);
    for (i, path) in shown.iter().enumerate() {
        let fi = info.get(i).copied().unwrap_or(FileInfo { exists: true, age_secs: None });
        let row = Rect::from_min_size(pos2(left - 8.0, top + 32.0 + i as f32 * ROW_H), vec2(width + 8.0, ROW_H));
        let resp = ui.interact(row, egui::Id::new(("home", "recent", i)), Sense::click()).on_hover_text(path.as_str());
        let name = crate::menus::recent_name(path);
        app.auto.add(&format!("home.recent.{i}"), row, &name);
        if resp.hovered() {
            ui.painter().rect_filled(row, 4.0, t.hover);
        }
        let ink = if fi.exists { t.text } else { t.text_faint };
        let p = ui.painter().with_clip_rect(row);
        icons::paint(&p, Rect::from_center_size(pos2(row.min.x + 20.0, row.center().y), vec2(14.0, 14.0)), Icon::Sequence, t.text_dim);
        let name_r = p.text(pos2(row.min.x + 38.0, row.center().y), Align2::LEFT_CENTER, &name, Tokens::ui(13.0), ink);
        let folder_x = name_r.max.x + 12.0;
        let right_text = match (fi.exists, fi.age_secs) {
            (false, _) => tl!("File not found").to_string(),
            (true, Some(s)) => age_text(s),
            (true, None) => String::new(),
        };
        let x_btn = Rect::from_center_size(pos2(row.max.x - 16.0, row.center().y), vec2(20.0, 20.0));
        let date_r = p.text(pos2(x_btn.min.x - 8.0, row.center().y), Align2::RIGHT_CENTER, &right_text, Tokens::ui(11.5), t.text_dim);
        if let Some(folder) = std::path::Path::new(path).parent() {
            let clip = p.with_clip_rect(Rect::from_min_max(pos2(folder_x, row.min.y), pos2((date_r.min.x - 12.0).max(folder_x), row.max.y)));
            clip.text(pos2(folder_x, row.center().y), Align2::LEFT_CENTER, folder.to_string_lossy(), Tokens::ui(11.5), t.text_dim);
        }
        let x_resp = ui.interact(x_btn, egui::Id::new(("home", "recent-x", i)), Sense::click());
        app.auto.add(&format!("home.recent.{i}.remove"), x_btn, tl!("Remove from list"));
        if resp.hovered() || x_resp.hovered() {
            let c = if x_resp.hovered() { t.text } else { t.text_dim };
            let s = 4.0;
            let m = x_btn.center();
            let stroke = egui::Stroke::new(1.5, c);
            ui.painter().line_segment([m + vec2(-s, -s), m + vec2(s, s)], stroke);
            ui.painter().line_segment([m + vec2(-s, s), m + vec2(s, -s)], stroke);
            x_resp.clone().on_hover_text(tl!("Remove from list"));
        }
        if x_resp.clicked() {
            remove = Some(path.clone());
        } else if resp.clicked() {
            if fi.exists {
                open = Some(path.clone());
            } else {
                app.ui.status = tl!("File not found").to_string();
            }
        }
    }
    if let Some(p) = remove {
        forget_recent(app, Some(&p));
    } else if let Some(p) = open {
        open_path(app, &ctx, &p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn age_text_buckets() {
        assert_eq!(age_text(5), "Just now");
        assert_eq!(age_text(5 * 60), "5 min ago");
        assert_eq!(age_text(3 * 3600), "3 h ago");
        assert_eq!(age_text(2 * 86_400), "2 d ago");
        assert_eq!(age_text(800 * 86_400), "2 y ago");
    }

    #[test]
    fn missing_file_is_flagged_and_a_present_one_is_not() {
        assert!(!file_info("/definitely/not/here.fcproj").exists);
        let dir = std::env::temp_dir();
        assert!(file_info(&dir.to_string_lossy()).exists);
    }
}
