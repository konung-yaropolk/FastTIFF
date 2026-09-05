use super::*;

/// These read process-wide environment variables, so they must not run
/// concurrently with each other. `cargo test` threads them by default.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct EnvGuard(Vec<(&'static str, Option<std::ffi::OsString>)>);

impl EnvGuard {
    fn set(vars: &[(&'static str, Option<&str>)]) -> Self {
        let saved = vars
            .iter()
            .map(|(k, _)| (*k, std::env::var_os(k)))
            .collect();
        for (k, v) in vars {
            match v {
                // SAFETY: the lock above serialises every test that touches
                // the environment, and nothing else in this crate reads it.
                Some(v) => unsafe { std::env::set_var(k, v) },
                None => unsafe { std::env::remove_var(k) },
            }
        }
        EnvGuard(saved)
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (k, v) in &self.0 {
            match v {
                Some(v) => unsafe { std::env::set_var(k, v) },
                None => unsafe { std::env::remove_var(k) },
            }
        }
    }
}

#[test]
fn the_search_path_is_ordered_and_deduplicated() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let joined = std::env::join_paths(["/tmp/one", "/tmp/two", "/tmp/one"])
        .unwrap()
        .into_string()
        .unwrap();
    let _g = EnvGuard::set(&[(PLUGIN_PATH_VAR, Some(&joined))]);

    let paths = search_paths();
    assert!(!paths.is_empty());
    assert_eq!(
        paths[0],
        std::path::PathBuf::from("/tmp/one"),
        "the override must come first, so a developer can shadow everything"
    );
    assert_eq!(paths[1], std::path::PathBuf::from("/tmp/two"));

    let mut seen = paths.clone();
    let before = seen.len();
    seen.sort();
    seen.dedup();
    assert_eq!(seen.len(), before, "a repeated directory must appear once");
}

/// Without the override the path still has entries — a user who has set nothing
/// must still be told where to put a plugin.
#[test]
fn there_is_always_somewhere_to_put_a_plugin() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _g = EnvGuard::set(&[(PLUGIN_PATH_VAR, None)]);
    let paths = search_paths();
    assert!(!paths.is_empty(), "the search path must never be empty");
    assert!(
        user_plugin_dir().is_some_and(|u| paths.contains(&u)),
        "the user's own writable directory must be on the path"
    );
}

/// A directory that does not exist yet is kept, because the UI shows this list
/// to tell the user where a plugin *would* go.
#[test]
fn missing_directories_are_kept_not_filtered() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let ghost = std::env::temp_dir().join("fasttiff-does-not-exist-9f3a");
    let _ = std::fs::remove_dir_all(&ghost);
    let joined = std::env::join_paths([&ghost])
        .unwrap()
        .into_string()
        .unwrap();
    let _g = EnvGuard::set(&[(PLUGIN_PATH_VAR, Some(&joined))]);
    assert!(!ghost.exists());
    assert!(search_paths().contains(&ghost));
}

#[test]
fn the_library_extension_matches_the_platform() {
    let expected = if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    assert_eq!(LIBRARY_EXT, expected);

    assert!(is_library(std::path::Path::new(&format!(
        "a.{LIBRARY_EXT}"
    ))));
    // Case-insensitively, because Windows writes `.DLL` as readily as `.dll`.
    assert!(is_library(std::path::Path::new(&format!(
        "a.{}",
        LIBRARY_EXT.to_uppercase()
    ))));
    assert!(!is_library(std::path::Path::new("notes.txt")));
    assert!(!is_library(std::path::Path::new("no-extension")));
}

/// On Linux the XDG variables decide, and an empty or relative value must be
/// ignored rather than producing a path relative to the working directory.
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
#[test]
fn xdg_data_home_is_honoured_but_only_when_absolute() {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    {
        let _g = EnvGuard::set(&[
            ("XDG_DATA_HOME", Some("/xdg/data")),
            ("HOME", Some("/home/u")),
        ]);
        assert_eq!(
            user_plugin_dir().unwrap(),
            std::path::PathBuf::from("/xdg/data/fasttiff/plugins")
        );
    }
    {
        // Relative: the spec says to ignore it and fall back to $HOME.
        let _g = EnvGuard::set(&[
            ("XDG_DATA_HOME", Some("relative/path")),
            ("HOME", Some("/home/u")),
        ]);
        assert_eq!(
            user_plugin_dir().unwrap(),
            std::path::PathBuf::from("/home/u/.local/share/fasttiff/plugins")
        );
    }
}
