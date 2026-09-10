//! Tests for the in-memory handover's sizing rule.
//!
//! The handover itself is not tested here: it spawns a second copy of the
//! viewer, which wants a window server and a GPU, and a test that needed those
//! would be testing the machine rather than the rule.

use super::*;

/// Whatever the platform says, it has to be a believable amount of memory.
///
/// This is the guard that matters: a query that silently returned zero would
/// send every result through a file and nobody would notice, because a file
/// still works.
#[test]
fn available_memory_is_plausible_or_absent() {
    match available_memory() {
        Some(bytes) => {
            assert!(
                bytes >= 1 << 20,
                "reported {bytes} bytes available, which is less than a megabyte"
            );
            assert!(
                bytes <= 1 << 50,
                "reported {bytes} bytes available, which is more than a petabyte"
            );
        }
        // Only where nothing is asked. Both platforms this is built for answer.
        None => assert!(
            !cfg!(any(windows, target_os = "linux")),
            "this platform has a memory query and it declined to answer"
        ),
    }
}

/// A small result never goes to disk, whatever the machine.
#[test]
fn a_small_result_stays_in_memory() {
    assert!(fits_in_memory(0));
    assert!(fits_in_memory(4 << 20));
}

/// And an impossible one always does.
#[test]
fn an_enormous_result_goes_to_a_file() {
    assert!(!fits_in_memory(usize::MAX));
    // A petabyte. The multiply by three must not wrap round into a small
    // number and quietly wave it through — which is what `saturating_mul`
    // is there for.
    assert!(!fits_in_memory(1 << 50));
}

/// The rule asks for three copies' worth, not one.
///
/// Pinned because the reason is not visible from the call site: the copy being
/// written, the copy the new window is building, and headroom so that opening
/// it does not take the machine to its limit.
#[test]
fn the_margin_is_threefold() {
    let Some(available) = available_memory() else {
        return; // nothing to compare against on a platform that does not say
    };
    let third = (available / 3) as usize;
    assert!(
        fits_in_memory(third.saturating_sub(1 << 20)),
        "just under a third of {available} bytes should fit"
    );
    assert!(
        !fits_in_memory(third.saturating_add(available as usize / 10)),
        "comfortably over a third of {available} bytes should not"
    );
}

// ------------------------------------------------- the pipe itself

/// The environment variable that turns [`pipe_sink`] from a no-op into the
/// child half of [`every_byte_reaches_the_child`]. Its value is the file the
/// sink writes what it received to.
const SINK: &str = "FASTTIFF_TEST_PIPE_SINK";

/// The child of the pipe test: read standard input to the end and record what
/// arrived.
///
/// This is a test only so that it can be re-executed — running the test binary
/// again is the one portable way to get a process that does exactly this, with
/// no shell and no assumptions about what is installed. Without the variable
/// set it does nothing at all, which is what happens in every ordinary run.
#[test]
fn pipe_sink() {
    let Ok(out) = std::env::var(SINK) else {
        return;
    };
    use std::io::Read;
    let mut got = Vec::new();
    std::io::stdin()
        .lock()
        .read_to_end(&mut got)
        .expect("the sink could not read its standard input");
    // Length and a running sum: enough to catch a dropped chunk, a truncation,
    // or two chunks swapped, without writing the payload back out again.
    let sum = got
        .iter()
        .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
    std::fs::write(out, format!("{} {}", got.len(), sum)).expect("the sink could not report");
}

/// Every byte handed to the pipe arrives at the other end, in order.
///
/// The payload deliberately crosses the 8 MiB chunk boundary several times:
/// the failure this guards against is a stack that is one chunk short, which
/// on the other side is a TIFF that does not parse — and a result quietly lost
/// after the whole run that produced it.
#[test]
fn every_byte_reaches_the_child() {
    let exe = std::env::current_exe().expect("no test binary to re-run");
    let out = std::env::temp_dir().join(format!(
        "fasttiff-pipe-test-{}-{:?}.txt",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_file(&out);

    // Just over three chunks, so the last one is a partial.
    let payload: Vec<u8> = (0..(25 << 20) + 7).map(|i| (i % 251) as u8).collect();
    let expected = payload
        .iter()
        .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));

    let mut child = Command::new(exe)
        .args(["--exact", "process::tests::pipe_sink", "--nocapture"])
        .env(SINK, &out)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("could not start the sink");

    let mut seen: Vec<f32> = Vec::new();
    feed(&mut child, &payload, &mut |f| {
        seen.push(f);
        true
    })
    .expect("the write failed");
    let status = child.wait().expect("the sink did not finish");
    assert!(status.success(), "the sink exited with {status}");

    let recorded = std::fs::read_to_string(&out).expect("the sink recorded nothing");
    let _ = std::fs::remove_file(&out);
    assert_eq!(
        recorded,
        format!("{} {expected}", payload.len()),
        "what arrived is not what was sent"
    );

    // And the bar moved while it did, ending at the end.
    assert!(seen.len() > 3, "progress was reported {} times", seen.len());
    assert!(
        seen.windows(2).all(|w| w[1] >= w[0]),
        "progress went backwards: {seen:?}"
    );
    assert_eq!(seen.last().copied(), Some(1.0));
}

/// Refusing from the progress callback stops the handover and takes the child
/// with it.
///
/// A cancelled run must not leave a window opening onto half a stack, so the
/// half-fed child is killed rather than having its pipe politely closed.
#[test]
fn cancelling_kills_the_half_fed_child() {
    let exe = std::env::current_exe().expect("no test binary to re-run");
    let out = std::env::temp_dir().join(format!(
        "fasttiff-pipe-cancel-{}-{:?}.txt",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_file(&out);

    let payload = vec![7u8; (17 << 20) + 3];
    let mut child = Command::new(exe)
        .args(["--exact", "process::tests::pipe_sink", "--nocapture"])
        .env(SINK, &out)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("could not start the sink");

    let err =
        feed(&mut child, &payload, &mut |_| false).expect_err("the write should have stopped");
    assert!(
        err.to_string().contains("cancelled"),
        "unexpected error: {err}"
    );
    let _ = child.wait();
    assert!(
        !out.exists(),
        "the child reported a stack it was never given all of"
    );
    let _ = std::fs::remove_file(&out);
}
