//! Exercise the desktop executable's headless cleanup mode with real process
//! termination. No window system or external sample TIFF is needed.
use fast_tiff_viewer::plugins::ScratchFile;
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

fn supervisor(path: &Path) -> (Child, i32) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_FastTIFF"))
        .arg(ScratchFile::GUARD)
        .arg(path)
        .arg("data.tif")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut status = [0; 4];
    child
        .stdout
        .as_mut()
        .unwrap()
        .read_exact(&mut status)
        .unwrap();
    (child, i32::from_le_bytes(status))
}

fn wait_until(mut ready: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !ready() {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for scratch ownership"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn terminated_writer_releases_only_its_own_tiff() {
    const WRITER: &str = "FASTTIFF_TEST_CRASH_WRITER_DIR";
    let scratch = ".fasttiff-1-2-3-data.tif.temp";
    if let Some(dir) = std::env::var_os(WRITER) {
        let dir = PathBuf::from(dir);
        let (_lease, status) = supervisor(&dir.join(scratch));
        assert_eq!(status, 0);
        fs::write(dir.join(scratch), b"partial pixels").unwrap();
        fs::write(dir.join("ready"), b"").unwrap();
        std::thread::sleep(Duration::from_secs(30));
        return;
    }

    let dir = std::env::temp_dir().join(format!("fasttiff-crash-{}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let original = dir.join("recording.tif");
    fs::write(&original, b"original pixels").unwrap();
    let neighbor = dir.join(".fasttiff-4-5-6-other.tif.temp");
    let (mut live, status) = supervisor(&neighbor);
    assert_eq!(status, 0);
    fs::write(&neighbor, b"other window").unwrap();
    let (mut duplicate, status) = supervisor(&neighbor);
    assert_ne!(status, 0);
    duplicate.wait().unwrap();

    let mut writer = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "terminated_writer_releases_only_its_own_tiff"])
        .env(WRITER, &dir)
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    wait_until(|| dir.join("ready").exists());
    writer.kill().unwrap();
    writer.wait().unwrap();
    wait_until(|| !dir.join(scratch).exists());
    assert_eq!(fs::read(original).unwrap(), b"original pixels");
    assert_eq!(fs::read(&neighbor).unwrap(), b"other window");

    let saved = dir.join("saved.tif");
    fs::rename(neighbor, &saved).unwrap();
    drop(live.stdin.take());
    live.wait().unwrap();
    assert_eq!(fs::read(saved).unwrap(), b"other window");
    fs::remove_dir_all(dir).unwrap();
}
