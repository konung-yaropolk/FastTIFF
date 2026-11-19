//! What the frame works out from the document and keeps until the document
//! changes: one way of keeping it, for everything the app keeps that way.
//!
//! A frame is drawn many times a second and the document behind it changes
//! when a file lands, so what the frame shows of it — the window title, the
//! metadata window's OME tree — is worked out once per document rather than
//! once per frame. Each is a [`Cached`] field held where it is used, keyed by
//! what it was worked out from: the document's generation as the window
//! counts them (`ViewerApp::document_generation`), or a finer revision where
//! one exists. The core keeps its own the same way, by revision
//! ([`fast_tiff_viewer::Revision`]).
//!
//! Not egui's `ctx.data`: that is keyed by an [`egui::Id`] rather than by what
//! the value describes, so whatever stands for the document has to be packed
//! into the value and compared by hand — and the only such thing at hand there
//! was where the document happened to live in memory, which a new one can
//! reuse.

/// A value worked out from whatever `K` identifies, kept until it is asked
/// for under another key.
pub(super) struct Cached<K, V> {
    held: Option<(K, V)>,
}

impl<K, V> Default for Cached<K, V> {
    fn default() -> Self {
        Cached { held: None }
    }
}

impl<K: PartialEq, V> Cached<K, V> {
    /// The value for `key`: the one held when it was worked out for `key`,
    /// otherwise `derive`'s, which is then held instead.
    pub(super) fn get(&mut self, key: K, derive: impl FnOnce() -> V) -> &V {
        if !self.is_for(&key) {
            self.held = Some((key, derive()));
        }
        let (_, value) = self.held.as_ref().expect("held just above");
        value
    }

    /// Whether what is held was worked out for `key`.
    fn is_for(&self, key: &K) -> bool {
        self.held.as_ref().is_some_and(|(k, _)| k == key)
    }
}

impl<K: PartialEq, V: PartialEq> Cached<K, V> {
    /// The value for `key` when it is news: worked out afresh for `key` and
    /// different from the one held before — for a value sent somewhere rather
    /// than drawn, which is sent only when it changes. `None` while `key` is
    /// the key it was last worked out for, and when the new value is the old
    /// one again.
    pub(super) fn changed(&mut self, key: K, derive: impl FnOnce() -> V) -> Option<&V> {
        if self.is_for(&key) {
            return None;
        }
        let value = derive();
        let same = self.held.as_ref().is_some_and(|(_, v)| *v == value);
        self.held = Some((key, value));
        let (_, value) = self.held.as_ref().expect("held just above");
        (!same).then_some(value)
    }
}

#[cfg(test)]
#[path = "cached_tests.rs"]
mod tests;
