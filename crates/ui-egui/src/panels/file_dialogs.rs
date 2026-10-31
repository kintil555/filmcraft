//! The crash-recovery prompt and the Revert confirmation (Settings ▸ Auto Save lives in
//! `panels::settings`).
//!
//! Automation ids: `recovery.item.<n>`, `recovery.recover`, `recovery.discard`, `recovery.later`;
//! `revert.yes`, `revert.no`.

use egui::{Color32, CornerRadius, Frame, Margin, RichText, Stroke, Ui, vec2};
use serde_json::json;

use crate::theme::Tokens;
use crate::{Dialog, FilmcraftApp};

/// Dialog-local state.
#[derive(Default)]
pub struct FileDialogState {
    pub recovery_choice: usize,
    /// Home ▸ New Project: the form being filled in (`None` while the dialog is closed).
    pub home_new: Option<NewProjectForm>,
    /// Focus the New Project name field on its next frame.
    pub home_new_focus: bool,
    /// Home ▸ Recent ▸ "Filter Recent Files".
    pub home_filter: String,
}

/// Aspect-ratio presets of the New Project dialog: label, width part, height part. The last entry
/// (zero parts) is "Custom": the width and height are free.
pub const NEW_RATIOS: [(&str, u32, u32); 7] =
    [("16:9", 16, 9), ("9:16", 9, 16), ("1:1", 1, 1), ("4:3", 4, 3), ("4:5", 4, 5), ("21:9", 21, 9), ("Custom", 0, 0)];
/// Frame-rate presets of the New Project dialog: label and frames per second.
pub const NEW_FPS: [(&str, f64); 8] =
    [("23.976", 23.976), ("24", 24.0), ("25", 25.0), ("29.97", 29.97), ("30", 30.0), ("50", 50.0), ("59.94", 59.94), ("60", 60.0)];
/// Index of "Custom" in [`NEW_RATIOS`].
pub const NEW_RATIO_CUSTOM: usize = 6;
/// Audio sample rates (Hz) of the New Project dialog.
pub const NEW_SAMPLE_RATES: [u32; 3] = [44_100, 48_000, 96_000];
/// Smallest and largest frame side (px) the dialog accepts; the engine validates the rest.
pub const NEW_SIDE_MIN: u32 = 16;
pub const NEW_SIDE_MAX: u32 = 16_384;

/// Home ▸ New Project: the sequence settings chosen before the project is created.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewProjectForm {
    pub name: String,
    /// Index into [`NEW_RATIOS`].
    pub ratio: usize,
    pub width: u32,
    pub height: u32,
    /// Index into [`NEW_FPS`].
    pub fps: usize,
    /// Index into [`NEW_SAMPLE_RATES`].
    pub sample_rate: usize,
    pub video_tracks: u32,
    pub audio_tracks: u32,
}

impl Default for NewProjectForm {
    fn default() -> Self {
        Self { name: "Untitled".into(), ratio: 0, width: 1920, height: 1080, fps: 0, sample_rate: 1, video_tracks: 3, audio_tracks: 3 }
    }
}

impl NewProjectForm {
    /// Pick a ratio preset: the width stays, the height follows (rounded to an even number).
    pub fn set_ratio(&mut self, idx: usize) {
        self.ratio = idx.min(NEW_RATIO_CUSTOM);
        if let Some(&(_, rw, rh)) = NEW_RATIOS.get(self.ratio)
            && rw > 0
            && rh > 0
        {
            let h = u64::from(self.width) * u64::from(rh) / u64::from(rw);
            let h = u32::try_from(h).unwrap_or(NEW_SIDE_MAX);
            self.height = (h / 2 * 2).clamp(NEW_SIDE_MIN, NEW_SIDE_MAX);
        }
    }

    /// The height was typed while a ratio preset is active: the width follows (rounded to an even number).
    pub fn set_height_keep_ratio(&mut self) {
        if let Some(&(_, rw, rh)) = NEW_RATIOS.get(self.ratio)
            && rw > 0
            && rh > 0
        {
            let w = u64::from(self.height) * u64::from(rw) / u64::from(rh);
            let w = u32::try_from(w).unwrap_or(NEW_SIDE_MAX);
            self.width = (w / 2 * 2).clamp(NEW_SIDE_MIN, NEW_SIDE_MAX);
        }
    }

    /// After the width or height was typed: keep a preset only when it still matches exactly.
    pub fn sync_ratio(&mut self) {
        let hit = NEW_RATIOS.iter().position(|&(_, rw, rh)| rw > 0 && u64::from(self.width) * u64::from(rh) == u64::from(self.height) * u64::from(rw));
        self.ratio = hit.unwrap_or(NEW_RATIO_CUSTOM);
    }

    /// The `file.newSequence` parameters for these settings.
    pub fn sequence_params(&self) -> serde_json::Value {
        let fps = NEW_FPS.get(self.fps).map_or(23.976, |f| f.1);
        let sample_rate = NEW_SAMPLE_RATES.get(self.sample_rate).copied().unwrap_or(48_000);
        json!({
            "width": self.width.clamp(NEW_SIDE_MIN, NEW_SIDE_MAX),
            "height": self.height.clamp(NEW_SIDE_MIN, NEW_SIDE_MAX),
            "fps": fps,
            "sampleRate": sample_rate,
            "video": self.video_tracks.min(32),
            "audio": self.audio_tracks.min(32),
        })
    }

    /// The project name to use: trimmed, or "Untitled" when empty.
    pub fn project_name(&self) -> &str {
        let n = self.name.trim();
        if n.is_empty() { "Untitled" } else { n }
    }
}

fn modal_frame(t: &Tokens) -> Frame {
    Frame::new().fill(t.panel_bg).stroke(Stroke::new(1.0, t.separator)).corner_radius(CornerRadius::same(10)).inner_margin(Margin::same(0))
}

fn button(app: &mut FilmcraftApp, ui: &mut Ui, id: &str, label: &str, primary: bool) -> bool {
    let t = app.tokens;
    let text = RichText::new(label).size(13.0).color(if primary { Color32::WHITE } else { t.text });
    let b = egui::Button::new(text)
        .min_size(vec2(88.0, 30.0))
        .corner_radius(CornerRadius::same(15))
        .fill(if primary { t.accent } else { Color32::TRANSPARENT })
        .stroke(if primary { Stroke::NONE } else { Stroke::new(1.5, t.text_faint) });
    let r = ui.add(b).on_hover_cursor(egui::CursorIcon::PointingHand);
    app.auto.add(id, r.rect, label);
    r.clicked()
}

pub fn show_recovery(app: &mut FilmcraftApp, ctx: &egui::Context) -> bool {
    let t = app.tokens;
    let list = app.session.execute("file.recoveryList", json!({})).unwrap_or_default();
    let items = list.as_array().cloned().unwrap_or_default();
    if items.is_empty() {
        return false;
    }
    let mut choice = app.file_dialogs.recovery_choice.min(items.len() - 1);
    let mut open = true;
    egui::Modal::new(egui::Id::unique("recovery-modal")).frame(modal_frame(&t)).show(ctx, |ui| {
        ui.set_width(520.0);
        Frame::new().inner_margin(Margin { left: 28, right: 28, top: 24, bottom: 20 }).show(ui, |ui| {
            ui.label(RichText::new(tl!("Recover Unsaved Changes")).size(18.0).strong().color(t.text));
            ui.add_space(6.0);
            ui.separator();
            ui.add_space(6.0);
            let c = &items[choice];
            let name = c["projectName"].as_str().unwrap_or(tl!("Untitled"));
            let when = c["savedAt"].as_str().unwrap_or("");
            let why = if c["cleanExit"].as_bool() == Some(true) {
                tlf!("FilmCraft was closed while “{name}” had unsaved changes.", name)
            } else {
                tlf!("FilmCraft quit unexpectedly while “{name}” had unsaved changes.", name)
            };
            ui.label(RichText::new(why).size(13.5).color(t.text));
            ui.add_space(4.0);
            ui.label(RichText::new(tlf!("Recover unsaved changes from {when}?", when)).size(13.5).color(t.text));
            if let Some(p) = c["projectPath"].as_str() {
                ui.label(RichText::new(p).size(11.5).color(t.text_dim));
            } else {
                ui.label(RichText::new(tl!("The project had not been saved yet.")).size(11.5).color(t.text_dim));
            }
            if items.len() > 1 {
                ui.add_space(8.0);
                ui.label(RichText::new(tlf!("{n} sessions have unsaved changes:", n = items.len())).size(12.0).color(t.text_dim));
                for (i, it) in items.iter().enumerate() {
                    let label = format!("{} — {}", it["projectName"].as_str().unwrap_or(tl!("Untitled")), it["savedAt"].as_str().unwrap_or(""));
                    let r = ui.radio(choice == i, RichText::new(&label).size(12.5));
                    app.auto.add(&format!("recovery.item.{i}"), r.rect, &label);
                    if r.clicked() {
                        choice = i;
                    }
                }
            }
            ui.add_space(18.0);
            let id = items[choice]["id"].as_str().unwrap_or("").to_string();
            ui.horizontal(|ui| {
                if button(app, ui, "recovery.discard", tl!("Discard"), false) {
                    if let Err(e) = app.session.execute("file.discardRecovery", json!({ "id": id })) {
                        app.ui.status = e.to_string();
                    }
                    choice = 0;
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if button(app, ui, "recovery.recover", tl!("Recover"), true) {
                        match app.session.execute("file.recover", json!({ "id": id })) {
                            Ok(_) => open = false,
                            Err(e) => app.ui.status = e.to_string(),
                        }
                    }
                    if button(app, ui, "recovery.later", tl!("Not Now"), false) {
                        open = false;
                    }
                });
            });
        });
    });
    app.file_dialogs.recovery_choice = choice;
    open && !app.session.recovery_candidates().is_empty()
}

pub fn show_revert(app: &mut FilmcraftApp, ctx: &egui::Context) -> bool {
    let t = app.tokens;
    let file = app.session.path.as_deref().and_then(|p| std::path::Path::new(p).file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut open = true;
    let r = egui::Modal::new(egui::Id::unique("revert-modal")).frame(modal_frame(&t)).show(ctx, |ui| {
        ui.set_width(460.0);
        Frame::new().inner_margin(Margin { left: 28, right: 28, top: 24, bottom: 20 }).show(ui, |ui| {
            ui.label(RichText::new(tl!("Revert")).size(18.0).strong().color(t.text));
            ui.add_space(6.0);
            ui.separator();
            ui.add_space(6.0);
            ui.label(RichText::new(tlf!("Are you sure you want to discard your changes to '{file}'?", file)).size(13.5).color(t.text));
            ui.add_space(18.0);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if button(app, ui, "revert.yes", tl!("Yes"), true) {
                    if let Err(e) = app.session.execute("file.revert", json!({})) {
                        app.ui.status = e.to_string();
                    }
                    open = false;
                }
                if button(app, ui, "revert.no", tl!("No"), false) {
                    open = false;
                }
            });
        });
    });
    // (the modal's own close also fires on Escape: not while a drag is being cancelled, #580)
    open && (!r.should_close() || ctx.input(|i| i.pointer.any_down()))
}

/// A label cell of the New Project form.
fn form_label(ui: &mut Ui, t: &Tokens, text: &str) {
    ui.label(RichText::new(text).size(13.0).color(t.text_dim));
}

/// Home ▸ New Project: name, aspect ratio, frame size, frame rate, audio rate and track counts.
/// Create makes the project with one sequence using these settings and opens it in Edit.
/// Automation ids: `home.new.name`, `home.new.ratio`, `home.new.width`, `home.new.height`,
/// `home.new.fps`, `home.new.samplerate`, `home.new.video`, `home.new.audio`, `home.new.create`,
/// `home.new.cancel`.
pub fn show_new_project(app: &mut FilmcraftApp, ctx: &egui::Context) {
    let Some(mut form) = app.file_dialogs.home_new.take() else { return };
    let t = app.tokens;
    let focus = std::mem::take(&mut app.file_dialogs.home_new_focus);
    let (mut create, mut cancel) = (false, false);
    let r = egui::Modal::new(egui::Id::unique("home-new-project-modal")).frame(modal_frame(&t)).show(ctx, |ui| {
        ui.set_width(440.0);
        Frame::new().inner_margin(Margin { left: 28, right: 28, top: 24, bottom: 20 }).show(ui, |ui| {
            ui.label(RichText::new(tl!("New Project")).size(18.0).strong().color(t.text));
            ui.add_space(6.0);
            ui.separator();
            ui.add_space(8.0);
            egui::Grid::new("home-new-project-grid").num_columns(2).spacing([18.0, 12.0]).show(ui, |ui| {
                form_label(ui, &t, tl!("Name"));
                let edit = ui.add(egui::TextEdit::singleline(&mut form.name).desired_width(260.0).char_limit(120));
                app.auto.add_response("home.new.name", &edit, tl!("Project name"));
                if focus {
                    edit.request_focus();
                }
                if edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    create = true;
                }
                ui.end_row();

                form_label(ui, &t, tl!("Aspect Ratio"));
                let mut ratio = form.ratio;
                let combo = craft_ui::ComboBox::from_id_salt("home-new-ratio")
                    .selected_text(NEW_RATIOS.get(ratio).map_or("16:9", |r| r.0))
                    .width(260.0)
                    .show_ui(ui, |ui| {
                        for (i, &(label, _, _)) in NEW_RATIOS.iter().enumerate() {
                            ui.selectable_value(&mut ratio, i, label);
                        }
                    });
                app.auto.add_response("home.new.ratio", &combo.response, tl!("Aspect Ratio"));
                if ratio != form.ratio {
                    form.set_ratio(ratio);
                }
                ui.end_row();

                form_label(ui, &t, tl!("Frame Size"));
                ui.horizontal(|ui| {
                    let w = ui.add(egui::DragValue::new(&mut form.width).range(NEW_SIDE_MIN..=NEW_SIDE_MAX).speed(2.0).suffix(" px"));
                    app.auto.add_response("home.new.width", &w, tl!("Width"));
                    ui.label(RichText::new("×").color(t.text_dim));
                    let h = ui.add(egui::DragValue::new(&mut form.height).range(NEW_SIDE_MIN..=NEW_SIDE_MAX).speed(2.0).suffix(" px"));
                    app.auto.add_response("home.new.height", &h, tl!("Height"));
                    if w.changed() {
                        // a preset keeps its ratio (the height follows); Custom leaves the height alone
                        let keep = form.ratio;
                        form.set_ratio(keep);
                    } else if h.changed() {
                        form.set_height_keep_ratio();
                    }
                });
                ui.end_row();

                form_label(ui, &t, tl!("Frame Rate"));
                let mut fps = form.fps;
                let combo = craft_ui::ComboBox::from_id_salt("home-new-fps")
                    .selected_text(NEW_FPS.get(fps).map_or("23.976", |f| f.0).to_string() + " fps")
                    .width(260.0)
                    .show_ui(ui, |ui| {
                        for (i, &(label, _)) in NEW_FPS.iter().enumerate() {
                            ui.selectable_value(&mut fps, i, format!("{label} fps"));
                        }
                    });
                app.auto.add_response("home.new.fps", &combo.response, tl!("Frame Rate"));
                form.fps = fps;
                ui.end_row();

                form_label(ui, &t, tl!("Audio Sample Rate"));
                let mut sr = form.sample_rate;
                let combo = craft_ui::ComboBox::from_id_salt("home-new-samplerate")
                    .selected_text(format!("{} Hz", NEW_SAMPLE_RATES.get(sr).copied().unwrap_or(48_000)))
                    .width(260.0)
                    .show_ui(ui, |ui| {
                        for (i, &hz) in NEW_SAMPLE_RATES.iter().enumerate() {
                            ui.selectable_value(&mut sr, i, format!("{hz} Hz"));
                        }
                    });
                app.auto.add_response("home.new.samplerate", &combo.response, tl!("Audio Sample Rate"));
                form.sample_rate = sr;
                ui.end_row();

                form_label(ui, &t, tl!("Tracks"));
                ui.horizontal(|ui| {
                    let v = ui.add(egui::DragValue::new(&mut form.video_tracks).range(1..=32).prefix(format!("{} ", tl!("Video"))));
                    app.auto.add_response("home.new.video", &v, tl!("Video tracks"));
                    ui.add_space(8.0);
                    let a = ui.add(egui::DragValue::new(&mut form.audio_tracks).range(0..=32).prefix(format!("{} ", tl!("Audio"))));
                    app.auto.add_response("home.new.audio", &a, tl!("Audio tracks"));
                });
                ui.end_row();
            });
            ui.add_space(18.0);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if button(app, ui, "home.new.create", tl!("Create"), true) {
                    create = true;
                }
                if button(app, ui, "home.new.cancel", tl!("Cancel"), false) {
                    cancel = true;
                }
            });
        });
    });
    // (the modal's own close also fires on Escape)
    if r.should_close() {
        cancel = true;
    }
    if create {
        let name = form.project_name().to_string();
        match crate::menus::invoke(app, ctx, "file.newProject", json!({"name": name})) {
            Ok(_) => {
                let mut params = form.sequence_params();
                if let Some(o) = params.as_object_mut() {
                    o.insert("name".into(), json!("Sequence 01"));
                }
                if let Err(e) = crate::menus::invoke(app, ctx, "file.newSequence", params) {
                    app.ui.status = e;
                }
                app.ui.mode = crate::state::Mode::Edit;
            }
            Err(e) => {
                app.ui.status = e;
                app.file_dialogs.home_new = Some(form);
            }
        }
    } else if !cancel {
        app.file_dialogs.home_new = Some(form);
    }
}

/// Draw the dialog if it is one of ours; returns Some(still open).
pub fn show(app: &mut FilmcraftApp, ctx: &egui::Context, d: Dialog) -> Option<bool> {
    match d {
        Dialog::Preferences => Some(crate::panels::settings::show(app, ctx)),
        Dialog::Recovery => Some(show_recovery(app, ctx)),
        Dialog::RevertConfirm => Some(show_revert(app, ctx)),
        _ => None,
    }
}
