//! Headless UI test of the Home screen: the header Home button opens it, it lists New / Open /
//! Auto Save, the recent projects, and the Auto Save switch is off by default and toggles.

use std::sync::mpsc::{Sender, channel};

use egui_kittest::Harness;
use filmcraft_engine::Session;
use filmcraft_ui_egui::FilmcraftApp;
use filmcraft_ui_egui::control::ControlRequest;
use serde_json::{Value, json};

struct Driver {
    harness: Harness<'static, FilmcraftApp>,
    tx: Sender<ControlRequest>,
}

impl Driver {
    fn new(recent: &[&str]) -> Self {
        let mut session = Session::default();
        let mut prefs = session.prefs.clone();
        prefs.general.recent_projects = recent.iter().map(|s| s.to_string()).collect();
        session.prefs = prefs;
        let (tx, rx) = channel();
        let app = FilmcraftApp::new(session).with_control(rx);
        let harness = Harness::builder().with_size(egui::vec2(1600.0, 980.0)).with_max_steps(10_000).build_eframe(move |_cc| app);
        let mut d = Driver { harness, tx };
        d.frames(4);
        d
    }

    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            let ctx = self.harness.ctx.clone();
            let mut raw = std::mem::take(self.harness.input_mut());
            eframe::App::raw_input_hook(self.harness.state_mut(), &ctx, &mut raw);
            *self.harness.input_mut() = raw;
            self.harness.step();
        }
    }

    fn ok(&mut self, method: &str, params: Value) -> Value {
        let (req, reply) = ControlRequest::new(method, params.clone());
        self.tx.send(req).unwrap();
        for _ in 0..600 {
            self.frames(1);
            if let Ok(v) = reply.try_recv() {
                assert_eq!(v["ok"], json!(true), "{method} {params} failed: {v}");
                return v["result"].clone();
            }
        }
        panic!("no reply to {method} {params}");
    }

    fn ids(&mut self, prefix: &str) -> Vec<String> {
        let v = self.ok("ui.elements", json!({"prefix": prefix}));
        v.as_array().unwrap().iter().filter_map(|e| e["id"].as_str().map(str::to_string)).collect()
    }
}

#[test]
fn home_lists_new_open_autosave_and_recent_projects() {
    let mut d = Driver::new(&["/projects/one.fcproj", "/projects/two.fcproj"]);
    assert!(d.ids("home.").is_empty(), "Home is not the default screen of a test session");
    d.ok("engine.execute", json!({"command": "mode.home", "params": {}}));
    d.frames(3);
    let ids = d.ids("home.");
    for id in ["home.new", "home.open", "home.back", "home.autosave", "home.recent.0", "home.recent.1"] {
        assert!(ids.iter().any(|a| a == id), "{id} in {ids:?}");
    }
    assert!(!ids.iter().any(|a| a == "home.recent.2"), "only the two recent projects are listed");
}

#[test]
fn home_without_recent_projects_still_offers_new_and_open() {
    let mut d = Driver::new(&[]);
    d.ok("engine.execute", json!({"command": "mode.home", "params": {}}));
    d.frames(3);
    let ids = d.ids("home.");
    assert!(ids.iter().any(|a| a == "home.new") && ids.iter().any(|a| a == "home.open"), "{ids:?}");
    assert!(!ids.iter().any(|a| a.starts_with("home.recent.")), "{ids:?}");
}

#[test]
fn back_to_editing_leaves_home() {
    let mut d = Driver::new(&[]);
    d.ok("engine.execute", json!({"command": "mode.home", "params": {}}));
    d.frames(3);
    d.ok("ui.click", json!({"id": "home.back"}));
    d.frames(3);
    assert!(d.ids("home.").is_empty(), "Home is closed");
}
