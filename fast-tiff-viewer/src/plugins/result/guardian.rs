//! A pipe ties a scratch TIFF to its writer. If the OS terminates the writer,
//! the pipe closes even when Rust destructors never run. The supervisor creates
//! the file exclusively before acknowledging it, so a crash during creation is
//! covered too. On handoff the supervisor launches and watches the new viewer.
use super::{remove_scratch, scratch_name, ScratchFile};
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use std::process::{ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

static ENABLED: AtomicBool = AtomicBool::new(false);

impl ScratchFile {
    /// Internal desktop entry points. Embedders can use ordinary drop cleanup.
    pub const GUARD: &str = "--guard-scratch";
    pub const OPEN: &str = "--from-scratch";

    /// Enable supervision once the executable has installed the guard entry point.
    pub fn enable_supervision() {
        ENABLED.store(true, Ordering::Relaxed);
    }

    /// Run before starting the GUI. Only a file created by this invocation
    /// can be removed. Even a correctly named existing file is left alone.
    pub fn supervise(path: &Path, name: &std::ffi::OsStr, source: Option<&Path>) -> io::Result<()> {
        supervise(
            path,
            &mut io::stdin().lock(),
            &mut io::stdout().lock(),
            || {
                Command::new(std::env::current_exe()?)
                    .arg(Self::OPEN)
                    .arg(path)
                    .arg(name)
                    .args(source)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .spawn()
            },
        )
    }
}

fn supervise(
    path: &Path,
    input: &mut impl Read,
    reply: &mut impl Write,
    mut open_viewer: impl FnMut() -> io::Result<std::process::Child>,
) -> io::Result<()> {
    let created = if scratch_name(path).is_some() {
        File::create_new(path)
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid scratch name",
        ))
    };
    let file = match created {
        Ok(file) => file,
        Err(e) => {
            send_status(reply, Some(&e))?;
            return Err(e);
        }
    };
    drop(file);
    let run = (|| {
        send_status(reply, None)?;
        let mut command = [0];
        while input.read(&mut command)? == 1 && command[0] == b'O' {
            match open_viewer() {
                Ok(mut child) => {
                    // A broken reply pipe means the creator exited. The new
                    // window still owns the TIFF until it saves or closes.
                    let _ = send_status(reply, None);
                    while child.try_wait()?.is_none() && path.try_exists().unwrap_or(true) {
                        std::thread::sleep(Duration::from_secs(1));
                    }
                    break;
                }
                Err(e) => {
                    send_status(reply, Some(&e))?;
                    // The creator can still open the result in its own window.
                    // Continue accepting commands until the owner closes its pipe.
                }
            }
        }
        Ok(())
    })();
    for _ in 0..100 {
        if remove_scratch(path) {
            return run;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    std::fs::remove_file(path)?;
    run
}

fn send_status(out: &mut impl Write, error: Option<&io::Error>) -> io::Result<()> {
    let code = error.map_or(0, |e| e.raw_os_error().unwrap_or(-1));
    out.write_all(&code.to_le_bytes())?;
    out.flush()
}

#[derive(Debug)]
pub(super) struct Guardian {
    commands: ChildStdin,
    replies: ChildStdout,
}

impl Guardian {
    fn status(&mut self) -> io::Result<()> {
        let mut bytes = [0; 4];
        self.replies.read_exact(&mut bytes)?;
        match i32::from_le_bytes(bytes) {
            0 => Ok(()),
            -1 => Err(io::Error::other("scratch supervisor rejected the request")),
            code => Err(io::Error::from_raw_os_error(code)),
        }
    }

    pub(super) fn open_window(&mut self) -> io::Result<()> {
        self.commands.write_all(b"O")?;
        self.status()
    }
}

pub(super) fn create(
    path: &Path,
    stem: &str,
    source: Option<&Path>,
) -> io::Result<(File, Option<Guardian>)> {
    if !ENABLED.load(Ordering::Relaxed) {
        return Ok((File::create_new(path)?, None));
    }
    let mut child = Command::new(std::env::current_exe()?)
        .arg(ScratchFile::GUARD)
        .arg(path)
        .arg(format!("{stem}.tif"))
        .args(source)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    let mut guard = Guardian {
        commands: child.stdin.take().expect("piped stdin"),
        replies: child.stdout.take().expect("piped stdout"),
    };
    // Unix retains the exit status until somebody waits for the subprocess.
    #[cfg(unix)]
    std::thread::Builder::new()
        .name("scratch-supervisor".into())
        .spawn(move || child.wait())?;
    guard.status()?;
    let file = File::options().read(true).write(true).open(path)?;
    Ok((file, Some(guard)))
}

#[cfg(test)]
mod tests {
    use super::*;

    // The receiving window stays alive until the test closes its stdin. This
    // exercises the real child-process lifetime without starting a GUI in CI.
    #[test]
    fn receiving_window() {
        if std::env::var_os("FASTTIFF_TEST_RECEIVING_WINDOW").is_some() {
            io::copy(&mut io::stdin().lock(), &mut io::sink()).unwrap();
            std::process::exit(2);
        }
    }

    #[test]
    fn handoff_survives_creator_exit_and_cleans_up_after_receiver_exit() {
        let path = std::env::temp_dir().join(format!(
            ".fasttiff-{}-0-0-handoff.tif.temp",
            std::process::id()
        ));
        std::thread::scope(|scope| {
            let (send, recv) = std::sync::mpsc::channel();
            let watch = scope.spawn(|| {
                let mut reply = Vec::new();
                supervise(&path, &mut b"O".as_slice(), &mut reply, move || {
                    let mut child = Command::new(std::env::current_exe()?)
                        .args([
                            "--exact",
                            "plugins::result::guardian::tests::receiving_window",
                        ])
                        .env("FASTTIFF_TEST_RECEIVING_WINDOW", "1")
                        .stdin(Stdio::piped())
                        .stdout(Stdio::null())
                        .spawn()?;
                    send.send(child.stdin.take().unwrap()).unwrap();
                    Ok(child)
                })
                .unwrap();
                assert_eq!(reply, [0; 8]); // creation and handoff acknowledged
            });
            let lease = recv.recv_timeout(Duration::from_secs(10)).unwrap();
            assert!(path.exists());
            drop(lease);
            watch.join().unwrap();
        });
        assert!(!path.exists());
    }
}
