use super::*;

impl FilmcraftApp {
    fn workspaces_path(&self) -> Option<std::path::PathBuf> {
        self.session.prefs_path.as_ref().and_then(|p| p.parent()).map(|d| d.join(dock::WORKSPACES_FILE))
    }

    /// Replace the saved workspaces and write them to the data directory.
    pub fn set_workspaces(&mut self, w: dock::WorkspacePrefs) -> Result<(), String> {
        self.workspaces = w;
        match self.workspaces_path() {
            Some(p) => self.workspaces.save(&p).map_err(|e| format!("saving workspaces: {e}")),
            None => Ok(()),
        }
    }

    /// Keep layout and timeline settings across launches: a changed docking layout is written to
    /// the current workspace and changed timeline settings to `ui_state.json`, once no pointer
    /// button is held (so a splitter drag saves once, when released).
    pub(super) fn persist_ui_state(&mut self, ctx: &egui::Context) {
        if !self.workspace_restored || ctx.input(|i| i.pointer.any_down()) {
            return;
        }
        match &self.persisted_dock {
            None => self.persisted_dock = Some(self.ui.dock.clone()),
            Some(last) if *last != self.ui.dock => {
                self.persisted_dock = Some(self.ui.dock.clone());
                let name = self.ui.workspace.clone();
                match dock::SavedWorkspace::new(&name, &self.ui.dock) {
                    Ok(entry) => {
                        let mut next = self.workspaces.clone();
                        match next.saved.iter_mut().find(|s| s.name == name) {
                            Some(s) => *s = entry,
                            None => next.saved.push(entry),
                        }
                        if let Err(e) = self.set_workspaces(next) {
                            log::warn!("{e}");
                        }
                    }
                    Err(e) => log::warn!("layout not saved: {e}"),
                }
            }
            _ => {}
        }
        let now = state::TimelinePrefs::of(&self.ui.timeline);
        match self.persisted_timeline {
            None => self.persisted_timeline = Some(now),
            Some(last) if last != now => {
                self.persisted_timeline = Some(now);
                if let Some(d) = self.workspaces_path().and_then(|p| p.parent().map(|d| d.to_path_buf()))
                    && let Err(e) = now.save(&d.join(state::TimelinePrefs::FILE))
                {
                    log::warn!("saving timeline settings: {e}");
                }
            }
            _ => {}
        }
    }

    pub fn set_workspace(&mut self, name: &str) {
        self.ui.workspace = name.to_string();
        self.ui.dock = dock::saved_docking(&self.workspaces, name);
        self.persisted_dock = Some(self.ui.dock.clone());
        self.pending_panel_raise = None;
        self.ui.keys.maximized = None;
        if self.workspaces.current != name {
            let mut next = self.workspaces.clone();
            next.current = name.to_string();
            if let Err(e) = self.set_workspaces(next) {
                self.ui.status = e;
            }
        }
        if name == "Color" {
            self.ui.show_scopes = false;
        }
    }

    pub fn show_panel(&mut self, p: PanelKind) {
        // closed while floating: it floats again where it was (#1985); taken before the Timeline's
        // own restore below docks it
        let open = self.ui.dock.contains(&p);
        let refloat = match self.ui.closed_floating.iter().position(|(k, _)| *k == p) {
            Some(i) if !open => Some(self.ui.closed_floating.remove(i).1),
            _ => None,
        };
        if p == PanelKind::Timeline {
            self.ui.dock.restore_timeline();
        }
        if !self.ui.dock.contains(&p) {
            let near = match p {
                // a side panel beside Properties (or Essential Sound), never over the Program monitor
                PanelKind::TextToSpeech => [PanelKind::Properties, PanelKind::EssentialSound, PanelKind::EssentialGraphics]
                    .into_iter()
                    .find(|k| self.ui.dock.contains(k))
                    .unwrap_or(PanelKind::Program),
                PanelKind::LumetriColor | PanelKind::EssentialGraphics | PanelKind::EssentialSound | PanelKind::Properties => PanelKind::Program,
                PanelKind::Source
                | PanelKind::EffectControls
                | PanelKind::AudioClipMixer
                | PanelKind::Metadata
                | PanelKind::LumetriScopes
                | PanelKind::AudioTrackMixer
                | PanelKind::Text
                | PanelKind::ReferenceMonitor
                | PanelKind::Timecode => PanelKind::Source,
                _ => PanelKind::Project,
            };
            self.ui.dock.open_near(p, near);
        }
        if let Some(rect) = refloat.filter(|r| r.iter().all(|v| v.is_finite())) {
            let rect = [rect[0], rect[1], rect[2].max(160.0), rect[3].max(120.0)];
            if let Err(e) = self.apply_dock(craft_ui::docking::Action::Float { panel: p, rect }) {
                // refused (it is the only docked panel left): it stays docked, and its floating
                // place is kept rather than lost
                log::warn!("could not float {} again: {e}", p.id());
                self.ui.closed_floating.push((p, rect));
            }
        }
        let _ = self.apply_dock(craft_ui::docking::Action::Activate { panel: p });
        self.ui.focused = p;
    }

    /// Reveal in Project: bring the Project panel forward, showing the bin that holds `item` with
    /// the search cleared, so the (already selected) item is on show.
    pub(super) fn reveal_in_project(&mut self, item: filmcraft_project::ItemId) {
        // the bins on the way down to the item (bounded: a project file can nest bins arbitrarily deep)
        fn path(b: &filmcraft_project::Bin, item: filmcraft_project::ItemId, depth: usize, out: &mut Vec<u64>) -> bool {
            if depth > 256 {
                return false;
            }
            for c in &b.children {
                match c {
                    filmcraft_project::BinEntry::Item(i) if *i == item => return true,
                    filmcraft_project::BinEntry::Bin(inner) => {
                        out.push(inner.id.0);
                        if path(inner, item, depth + 1, out) {
                            return true;
                        }
                        out.pop();
                    }
                    _ => {}
                }
            }
            false
        }
        let mut bins = Vec::new();
        let found = path(&self.session.project.root, item, 0, &mut bins);
        // the list shows the whole tree: open the bins on the way. Icons and freeform show one bin
        // at a time: go into the one that holds the item.
        let list = self.session.prefs.project_panel.view.mode == filmcraft_engine::project_panel::ViewMode::List;
        self.ui.project_panel.bin = if list || !found { None } else { bins.last().copied() };
        if list {
            for b in bins {
                if !self.ui.expanded_bins.contains(&b) {
                    self.ui.expanded_bins.push(b);
                }
            }
        }
        self.ui.project_panel.active_tab = None;
        self.ui.project_panel.selected_bin = None;
        self.ui.project_search.clear();
        self.show_panel(PanelKind::Project);
    }
}
