use super::*;
use std::path::PathBuf;

#[test]
fn file_types_normalise_extensions() {
    // Written with or without a dot, in any case — matched the same way.
    let t = FileType::new("Olympus OIR", &[".OIR", "oir"]);
    assert_eq!(t.extensions, vec!["oir", "oir"]);
    assert!(t.matches(&PathBuf::from("cells.oir")));
    assert!(
        t.matches(&PathBuf::from("CELLS.OIR")),
        "matching is case-insensitive"
    );
    assert!(!t.matches(&PathBuf::from("cells.tif")));
    assert!(
        !t.matches(&PathBuf::from("oir")),
        "an extensionless file is not a match"
    );
}

struct Stub(&'static str, Confidence);

impl Importer for Stub {
    fn info(&self) -> PluginInfo {
        PluginInfo::new(self.0, "Stub")
    }
    fn file_types(&self) -> Vec<FileType> {
        vec![FileType::new("Stub", &["stub"])]
    }
    fn probe(&self, _p: &Path, _h: &[u8]) -> Confidence {
        self.1
    }
    fn import(
        &mut self,
        _r: &ImportRequest,
        _h: &mut dyn ImportHost,
    ) -> Result<ImportResult, PluginError> {
        Err(PluginError::failed("stub"))
    }
}

/// Confidence has to order, because that is how the host breaks a tie between
/// two importers claiming the same extension.
#[test]
fn confidence_orders_from_no_to_certain() {
    assert!(Confidence::Certain > Confidence::Maybe);
    assert!(Confidence::Maybe > Confidence::No);
    let mut v = [Confidence::Maybe, Confidence::Certain, Confidence::No];
    v.sort();
    assert_eq!(v, [Confidence::No, Confidence::Maybe, Confidence::Certain]);
}

/// The default `probe` answers from the extension alone, which is the right
/// behaviour for a format with no magic number.
#[test]
fn the_default_probe_follows_the_extension() {
    struct Plain;
    impl Importer for Plain {
        fn info(&self) -> PluginInfo {
            PluginInfo::new("x", "X")
        }
        fn file_types(&self) -> Vec<FileType> {
            vec![FileType::new("Plain", &["plain"])]
        }
        fn import(
            &mut self,
            _r: &ImportRequest,
            _h: &mut dyn ImportHost,
        ) -> Result<ImportResult, PluginError> {
            Err(PluginError::failed("x"))
        }
    }
    let p = Plain;
    assert_eq!(p.probe(Path::new("a.plain"), &[]), Confidence::Maybe);
    assert_eq!(p.probe(Path::new("a.other"), &[]), Confidence::No);
}

#[test]
fn an_importer_can_override_probe_to_read_a_magic_number() {
    let certain = Stub("a", Confidence::Certain);
    let no = Stub("b", Confidence::No);
    assert_eq!(
        certain.probe(Path::new("f.stub"), b"MAGIC"),
        Confidence::Certain
    );
    assert_eq!(no.probe(Path::new("f.stub"), b"MAGIC"), Confidence::No);
}
