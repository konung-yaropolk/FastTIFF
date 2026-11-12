use super::*;
use std::error::Error as _;

/// The chain as anyhow's `{:#}` prints it: each link's `{}`, joined by ": ".
/// Written out from `source()` alone, so it checks the links rather than this
/// type's own alternate formatting.
fn chain_text(e: &(dyn std::error::Error + 'static)) -> String {
    let mut parts = vec![e.to_string()];
    let mut next = e.source();
    while let Some(link) = next {
        parts.push(link.to_string());
        next = link.source();
    }
    parts.join(": ")
}

fn not_found() -> io::Error {
    io::Error::new(io::ErrorKind::NotFound, "no such file")
}

#[test]
fn an_error_is_one_pointer_wide_and_free_to_send() {
    fn send_sync_static<T: Send + Sync + 'static>() {}
    send_sync_static::<Error>();
    send_sync_static::<ErrorKind>();
    // As anyhow's is, so a host can hold one across `catch_unwind`.
    fn unwind_safe<T: std::panic::UnwindSafe + std::panic::RefUnwindSafe>() {}
    unwind_safe::<Error>();
    unwind_safe::<Result<()>>();
    assert_eq!(std::mem::size_of::<Error>(), std::mem::size_of::<usize>());
    assert_eq!(
        std::mem::size_of::<Result<()>>(),
        std::mem::size_of::<usize>()
    );
    assert_eq!(
        std::mem::size_of::<Result<usize>>(),
        2 * std::mem::size_of::<usize>()
    );
}

#[test]
fn a_message_with_nothing_to_format_is_not_copied() {
    let e = err!(Malformed, "TIFF has no image directories");
    assert!(matches!(e.0.message, Some(Cow::Borrowed(_))));
    let at = 8;
    let e = err!(Malformed, "IFD offset {at} out of bounds");
    assert_eq!(e.to_string(), "IFD offset 8 out of bounds");
    assert_eq!(e.kind(), ErrorKind::Malformed);
    assert!(e.source().is_none());
    assert!(e.io_error().is_none(), "only an I/O failure has one");
}

#[test]
fn bail_returns_the_error() {
    fn refuse(width: u32) -> Result<()> {
        if width == 0 {
            bail!(
                InvalidInput,
                "frame dimensions must be non-zero (got {width}x1)"
            );
        }
        Ok(())
    }
    let e = refuse(0).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::InvalidInput);
    assert_eq!(e.to_string(), "frame dimensions must be non-zero (got 0x1)");
    assert!(refuse(1).is_ok());
}

/// A bare I/O error reads as it did under anyhow: its own words once, and
/// whatever it wraps after them.
#[test]
fn an_io_error_converts_as_it_stands() {
    let e = Error::from(not_found());
    assert_eq!(e.kind(), ErrorKind::Io);
    assert_eq!(e.to_string(), "no such file");
    assert_eq!(format!("{e:#}"), "no such file");
    assert!(e.source().is_none());
    // Not a link of its own, but still there to ask what went wrong.
    assert_eq!(
        e.io_error().map(io::Error::kind),
        Some(io::ErrorKind::NotFound)
    );

    // An io::Error around another error is transparent the same way, so the
    // words still come once.
    let inner = io::Error::other(Error::new(ErrorKind::Malformed, "bad strip"));
    let e = Error::from(inner);
    assert_eq!(e.kind(), ErrorKind::Io);
    assert_eq!(format!("{e:#}"), "bad strip");
    assert_eq!(chain_text(&e), "bad strip");
}

/// Context on an I/O error puts the operating system's error in the chain as
/// itself, where a caller can still ask it what went wrong.
#[test]
fn context_on_io_keeps_the_io_error_as_the_source() {
    let r: std::result::Result<(), io::Error> = Err(not_found());
    let e = r
        .with_context(|| format!("could not open {}", "a.tif"))
        .unwrap_err();
    assert_eq!(e.kind(), ErrorKind::Io);
    assert_eq!(e.to_string(), "could not open a.tif");
    assert_eq!(format!("{e:#}"), "could not open a.tif: no such file");
    let io = e
        .source()
        .and_then(|s| s.downcast_ref::<io::Error>())
        .expect("the io::Error is the next link");
    assert_eq!(io.kind(), io::ErrorKind::NotFound);
    assert!(
        std::ptr::eq(e.io_error().unwrap(), io),
        "the same io::Error"
    );

    let r: std::result::Result<(), io::Error> = Err(not_found());
    assert_eq!(
        format!("{:#}", r.context("could not write the file").unwrap_err()),
        "could not write the file: no such file"
    );
}

/// Wrapping a converted I/O error does not repeat its words: the chain is the
/// two links anyhow's would be.
#[test]
fn context_on_a_converted_io_error_is_two_links() {
    let r: Result<()> = Err(Error::from(not_found()));
    let e = r.with_context(|| "frame 3").unwrap_err();
    assert_eq!(format!("{e:#}"), "frame 3: no such file");
    assert_eq!(chain_text(&e), "frame 3: no such file");
    assert!(e.source().unwrap().is::<io::Error>());
    assert_eq!(
        e.io_error().map(io::Error::kind),
        Some(io::ErrorKind::NotFound)
    );

    // And through any number of links of the crate's own.
    let r: std::result::Result<(), io::Error> = Err(not_found());
    let e = r
        .context("could not open a.tif")
        .context("loading")
        .with_context(|| "frame 3")
        .unwrap_err();
    assert_eq!(
        format!("{e:#}"),
        "frame 3: loading: could not open a.tif: no such file"
    );
    assert_eq!(
        e.io_error().map(io::Error::kind),
        Some(io::ErrorKind::NotFound)
    );
}

/// Every link of a chain says its own part and keeps the innermost kind, and
/// `{:#}` is exactly the walk anyhow's `{:#}` does.
#[test]
fn context_wraps_outermost_first_and_keeps_the_kind() {
    let r: Result<()> = Err(err!(
        TooLarge,
        "frame geometry overflows address space: 9x9"
    ));
    let e = r
        .context("decoding")
        .with_context(|| format!("frame {}", 4))
        .unwrap_err();
    assert_eq!(e.kind(), ErrorKind::TooLarge);
    assert_eq!(e.to_string(), "frame 4");
    let want = "frame 4: decoding: frame geometry overflows address space: 9x9";
    assert_eq!(format!("{e:#}"), want);
    assert_eq!(chain_text(&e), want);
    let first = e.source().unwrap();
    assert_eq!(first.to_string(), "decoding");
    let inner = first.downcast_ref::<Error>().unwrap();
    assert_eq!(inner.kind(), ErrorKind::TooLarge);
    assert!(e.io_error().is_none());
}

#[test]
fn a_missing_value_takes_the_kind_it_is_given() {
    let none: Option<u32> = None;
    let e = none
        .context(ErrorKind::Malformed, "IFD missing ImageWidth")
        .unwrap_err();
    assert_eq!(
        (e.kind(), e.to_string()),
        (ErrorKind::Malformed, "IFD missing ImageWidth".into())
    );
    let tag = 273;
    let e = none
        .with_context(ErrorKind::Unsupported, || format!("tag {tag} has no data"))
        .unwrap_err();
    assert_eq!(
        (e.kind(), e.to_string()),
        (ErrorKind::Unsupported, "tag 273 has no data".into())
    );
    assert_eq!(Some(5).context(ErrorKind::Malformed, "unused").unwrap(), 5);
}

#[test]
fn debug_names_the_kind_and_the_chain() {
    let r: std::result::Result<(), io::Error> = Err(not_found());
    let e = r.context("could not open a.tif").unwrap_err();
    assert_eq!(format!("{e:?}"), "Io: could not open a.tif: no such file");
    let pretty = format!("{e:#?}");
    assert!(
        pretty.contains("kind: Io") && pretty.contains("NotFound"),
        "{pretty}"
    );
}
