//! The dialog a plugin asks for, declared rather than drawn.
//!
//! A plugin says *what it needs* — "an integer 1..=64 called Radius" — and the
//! host renders it in whatever toolkit it happens to use. This is how ImageJ's
//! `GenericDialog` works, and the reason is the same one that decides most of
//! this crate: a plugin compiled by someone else's toolchain cannot be handed
//! an `&mut egui::Ui`. Rust has no stable ABI, so passing a live UI handle
//! across a dynamic library boundary is undefined behaviour that usually looks
//! like it works.
//!
//! The trade is real and worth stating plainly: a plugin cannot draw a custom
//! widget, an interactive preview, or a plot. What it gets instead is a dialog
//! that matches the host's theme for free, works identically in the desktop and
//! browser builds, and cannot crash the application.

/// One control in a plugin's dialog.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamDecl {
    /// Stable identifier, used to read the value back. Not shown to the user.
    pub key: String,
    /// The control's label.
    pub label: String,
    /// Optional one-line explanation, shown as a tooltip or beneath the control.
    pub help: Option<String>,
    pub kind: ParamKind,
}

impl ParamDecl {
    pub fn new(key: impl Into<String>, label: impl Into<String>, kind: ParamKind) -> Self {
        ParamDecl {
            key: key.into(),
            label: label.into(),
            help: None,
            kind,
        }
    }

    pub fn help(mut self, text: impl Into<String>) -> Self {
        self.help = Some(text.into());
        self
    }
}

/// What kind of control, and its bounds and default.
///
/// Ranges are inclusive and are a *contract*, not a hint: the host clamps to
/// them before the plugin ever sees a value, so a plugin never has to defend
/// against a radius of -1. That check belongs on the host side because the host
/// is the one that cannot be trusted to be bug-free from the plugin's point of
/// view, and vice versa.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamKind {
    Int {
        default: i64,
        min: i64,
        max: i64,
    },
    Float {
        default: f64,
        min: f64,
        max: f64,
    },
    Bool {
        default: bool,
    },
    /// A closed set. The value read back is the chosen index.
    Choice {
        default: usize,
        options: Vec<String>,
    },
    Text {
        default: String,
    },
    /// A path the user picks with the host's file dialog. `save` selects
    /// between an open and a save dialog.
    Path {
        default: String,
        save: bool,
    },
    /// Not a control: a line of text in the dialog. Lets a plugin explain
    /// itself without the host inventing a documentation channel.
    Label,
}

/// A value the user chose.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Choice(usize),
    Text(String),
    Path(String),
}

/// The filled-in dialog handed to [`crate::Plugin::run`].
///
/// The accessors take a default rather than returning `Option`, so a plugin
/// reading a key it never declared — or one whose type it got wrong — gets a
/// sane value instead of a panic. A plugin panicking is the host's problem to
/// contain, but it should not be this easy to cause.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Params {
    values: Vec<(String, ParamValue)>,
}

impl Params {
    pub fn new() -> Self {
        Params::default()
    }

    pub fn set(&mut self, key: impl Into<String>, value: ParamValue) {
        let key = key.into();
        match self.values.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = value,
            None => self.values.push((key, value)),
        }
    }

    pub fn get(&self, key: &str) -> Option<&ParamValue> {
        self.values.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &ParamValue)> {
        self.values.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn int(&self, key: &str, default: i64) -> i64 {
        match self.get(key) {
            Some(ParamValue::Int(v)) => *v,
            // A Choice is an index; reading it as an int is a reasonable thing
            // to want and cannot be ambiguous.
            Some(ParamValue::Choice(v)) => *v as i64,
            _ => default,
        }
    }

    pub fn float(&self, key: &str, default: f64) -> f64 {
        match self.get(key) {
            Some(ParamValue::Float(v)) => *v,
            Some(ParamValue::Int(v)) => *v as f64,
            _ => default,
        }
    }

    pub fn bool(&self, key: &str, default: bool) -> bool {
        match self.get(key) {
            Some(ParamValue::Bool(v)) => *v,
            _ => default,
        }
    }

    pub fn choice(&self, key: &str, default: usize) -> usize {
        match self.get(key) {
            Some(ParamValue::Choice(v)) => *v,
            Some(ParamValue::Int(v)) if *v >= 0 => *v as usize,
            _ => default,
        }
    }

    pub fn text<'a>(&'a self, key: &str, default: &'a str) -> &'a str {
        match self.get(key) {
            Some(ParamValue::Text(v)) | Some(ParamValue::Path(v)) => v.as_str(),
            _ => default,
        }
    }

    /// The defaults a declaration list asks for — what the host seeds a fresh
    /// dialog with, and what a headless caller (a test, a macro runner) can use
    /// to run a plugin without showing one.
    pub fn defaults(decls: &[ParamDecl]) -> Params {
        let mut p = Params::new();
        for d in decls {
            let v = match &d.kind {
                ParamKind::Int { default, .. } => ParamValue::Int(*default),
                ParamKind::Float { default, .. } => ParamValue::Float(*default),
                ParamKind::Bool { default } => ParamValue::Bool(*default),
                ParamKind::Choice { default, .. } => ParamValue::Choice(*default),
                ParamKind::Text { default } => ParamValue::Text(default.clone()),
                ParamKind::Path { default, .. } => ParamValue::Path(default.clone()),
                ParamKind::Label => continue,
            };
            p.set(d.key.clone(), v);
        }
        p
    }

    /// Force every value into the bounds its declaration states, and drop any
    /// key that was not declared.
    ///
    /// The host calls this between the dialog and [`crate::Plugin::run`], so a
    /// plugin's own `run` never has to re-validate what it already declared.
    /// A value of the wrong variant is replaced by the declared default rather
    /// than coerced, because a `Text` where an `Int` belongs means the two sides
    /// disagree about the dialog, and guessing would hide that.
    pub fn clamp_to(&self, decls: &[ParamDecl]) -> Params {
        let mut out = Params::new();
        for d in decls {
            match (&d.kind, self.get(&d.key)) {
                (ParamKind::Int { default, min, max }, v) => {
                    let raw = match v {
                        Some(ParamValue::Int(i)) => *i,
                        Some(ParamValue::Choice(i)) => *i as i64,
                        _ => *default,
                    };
                    out.set(
                        d.key.clone(),
                        ParamValue::Int(raw.clamp(*min.min(max), *max.max(min))),
                    );
                }
                (ParamKind::Float { default, min, max }, v) => {
                    let raw = match v {
                        Some(ParamValue::Float(f)) => *f,
                        Some(ParamValue::Int(i)) => *i as f64,
                        _ => *default,
                    };
                    // NaN would survive `clamp` as NaN on some paths and panic
                    // on others; a declared default is the only sane answer.
                    let v = if raw.is_nan() {
                        *default
                    } else {
                        raw.clamp(min.min(*max), max.max(*min))
                    };
                    out.set(d.key.clone(), ParamValue::Float(v));
                }
                (ParamKind::Bool { default }, v) => {
                    let b = match v {
                        Some(ParamValue::Bool(b)) => *b,
                        _ => *default,
                    };
                    out.set(d.key.clone(), ParamValue::Bool(b));
                }
                (ParamKind::Choice { default, options }, v) => {
                    let raw = match v {
                        Some(ParamValue::Choice(i)) => *i,
                        Some(ParamValue::Int(i)) if *i >= 0 => *i as usize,
                        _ => *default,
                    };
                    // An empty option list is a plugin bug, but it must not
                    // become a host panic.
                    let idx = if options.is_empty() {
                        0
                    } else {
                        raw.min(options.len() - 1)
                    };
                    out.set(d.key.clone(), ParamValue::Choice(idx));
                }
                (ParamKind::Text { default }, v) => {
                    let t = match v {
                        Some(ParamValue::Text(t)) => t.clone(),
                        _ => default.clone(),
                    };
                    out.set(d.key.clone(), ParamValue::Text(t));
                }
                (ParamKind::Path { default, .. }, v) => {
                    let t = match v {
                        Some(ParamValue::Path(t)) | Some(ParamValue::Text(t)) => t.clone(),
                        _ => default.clone(),
                    };
                    out.set(d.key.clone(), ParamValue::Path(t));
                }
                (ParamKind::Label, _) => {}
            }
        }
        out
    }
}

#[cfg(test)]
#[path = "params_tests.rs"]
mod tests;
