//! Decide where to write before accepting the first plane.
use super::{create_file, file_stem, ImageSpec, ResultDest, ScratchFile};
use std::{
    fs::File,
    io::{self, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

pub(super) fn estimate(spec: &ImageSpec) -> u64 {
    let pixels = (spec.width as u64) * (spec.height as u64);
    // The writer uses one uncompressed strip per plane. Allow a BigTIFF
    // directory per strip, all channel tables, and the carried description.
    let plane = pixels
        .saturating_mul(spec.pixel_type.bytes() as u64)
        .saturating_add(1024);
    let mut bytes = plane
        .saturating_mul(spec.plane_count().unwrap_or(usize::MAX) as u64)
        .saturating_add(4096)
        .saturating_add((spec.channels as u64).saturating_mul(1024));
    if let Some(info) = &spec.metadata {
        for text in info
            .description
            .iter()
            .chain(info.unit.iter())
            .chain(&info.channel_names)
        {
            bytes = bytes.saturating_add((text.len() as u64).saturating_mul(2));
        }
    }
    bytes
}

/// The file handle drops before its owner unlinks the scratch path.
pub(super) struct DiskOutput {
    file: File,
    pub(super) owner: ScratchFile,
    fallback: Option<PathBuf>,
    needed: u64,
}

impl DiskOutput {
    fn create(dir: &Path, stem: &str, needed: u64, fallback: Option<PathBuf>) -> io::Result<Self> {
        let (file, path) = create_file(dir, stem)?;
        let owner = ScratchFile::own(path)?;
        Ok(Self {
            file,
            owner,
            fallback,
            needed,
        })
    }

    /// Copy the bytes already accepted by the OS, then resume at the same
    /// offset. The encoder seeks back to patch the TIFF header at the end.
    fn relocate(&mut self, dir: &Path, pending: usize) -> io::Result<()> {
        let offset = self.file.stream_position()?;
        let length = self.file.metadata()?.len();
        let needed = self
            .needed
            .max(length)
            .max(offset.saturating_add(pending as u64));
        let free = available(dir)?;
        if free < needed {
            return Err(io::Error::new(
                io::ErrorKind::StorageFull,
                format!("{free} bytes available, {needed} needed"),
            ));
        }
        let name = self
            .owner
            .path()
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_prefix('.'))
            .and_then(|n| n.strip_suffix(".tif.temp"))
            .expect("name assigned by create_file");
        let mut next = Self::create(dir, name, needed, None)?;
        let copied = io::copy(&mut File::open(self.owner.path())?, &mut next.file)?;
        if copied != length {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "incomplete copy of temporary TIFF",
            ));
        }
        next.file.seek(SeekFrom::Start(offset))?;
        log::info!(
            "Continuing temporary TIFF from {} at {}",
            self.owner.path().display(),
            next.owner.path().display()
        );
        *self = next;
        Ok(())
    }

    fn try_io<T>(
        &mut self,
        pending: usize,
        mut operation: impl FnMut(&mut File) -> io::Result<T>,
    ) -> io::Result<T> {
        let first = match operation(&mut self.file) {
            Err(e) if capacity_error(&e) => e,
            result => return result,
        };
        let primary = self.owner.path().to_owned();
        let Some(dir) = self.fallback.take() else {
            return Err(io::Error::new(
                first.kind(),
                format!("Writing {}: {first}", primary.display()),
            ));
        };
        self.relocate(&dir, pending)
            .and_then(|()| operation(&mut self.file))
            .map_err(|e| {
                io::Error::new(
                    e.kind(),
                    format!(
                        "Writing {}: {first}. Trying source directory {}: {e}",
                        primary.display(),
                        dir.display()
                    ),
                )
            })
    }
}

fn capacity_error(e: &io::Error) -> bool {
    #[cfg(windows)]
    if matches!(e.raw_os_error(), Some(39 | 112 | 1295 | 223)) {
        // ERROR_HANDLE_DISK_FULL, ERROR_DISK_FULL, ERROR_DISK_QUOTA_EXCEEDED,
        // ERROR_FILE_TOO_LARGE (e.g. a FAT32 volume).
        return true;
    }
    matches!(
        e.kind(),
        io::ErrorKind::StorageFull
            | io::ErrorKind::QuotaExceeded
            | io::ErrorKind::FileTooLarge
            | io::ErrorKind::WriteZero
    )
}

impl Write for DiskOutput {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // A single write either reports how much it accepted or fails before
        // accepting anything. write_all outside this layer handles short writes.
        self.try_io(buf.len(), |file| match file.write(buf)? {
            0 if !buf.is_empty() => Err(io::ErrorKind::WriteZero.into()),
            n => Ok(n),
        })
    }
    fn flush(&mut self) -> io::Result<()> {
        self.try_io(0, |file| file.flush())
    }
}

impl Seek for DiskOutput {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.file.seek(pos)
    }
}

pub(super) fn allocate(dest: &ResultDest, spec: &ImageSpec) -> anyhow::Result<DiskOutput> {
    let (root, fallback) = match dest {
        ResultDest::Dir(dir) => (dir, None),
        ResultDest::WithFallback { primary, fallback } => (primary, Some(fallback.as_path())),
        ResultDest::Memory => unreachable!(),
    };
    let needed = estimate(spec);
    let stem = file_stem(&spec.name);
    choose(
        root,
        fallback,
        needed,
        |dir| {
            if dir == root {
                std::fs::create_dir_all(dir)?;
            }
            available(dir)
        },
        |dir| {
            let alternate = fallback
                .filter(|p| dir == root && *p != dir)
                .map(Path::to_owned);
            DiskOutput::create(dir, &stem, needed, alternate)
        },
    )
}

fn choose<T>(
    primary: &Path,
    fallback: Option<&Path>,
    needed: u64,
    mut free: impl FnMut(&Path) -> io::Result<u64>,
    mut create: impl FnMut(&Path) -> io::Result<T>,
) -> anyhow::Result<T> {
    let mut failures = Vec::new();
    for dir in std::iter::once(primary).chain(fallback.filter(|d| *d != primary)) {
        match free(dir) {
            Ok(bytes) if bytes < needed => {
                failures.push(format!("{} has {bytes} bytes free", dir.display()))
            }
            Ok(_) => match create(dir) {
                Ok(file) => {
                    if !failures.is_empty() {
                        log::info!(
                            "Using {} for the temporary TIFF ({needed} bytes): {}",
                            dir.display(),
                            failures.join("; ")
                        );
                    }
                    return Ok(file);
                }
                Err(e) => failures.push(format!("{}: {e}", dir.display())),
            },
            Err(e) => failures.push(format!("{}: {e}", dir.display())),
        }
    }
    anyhow::bail!(
        "Temporary TIFF needs {needed} bytes. {}",
        failures.join("; ")
    )
}

#[cfg(windows)]
fn available(dir: &Path) -> io::Result<u64> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
    // canonicalize also gives Win32 an extended-length path. UNC directory
    // names passed to GetDiskFreeSpaceExW require the trailing backslash.
    let mut path: Vec<u16> = dir.canonicalize()?.as_os_str().encode_wide().collect();
    if !path.ends_with(&[u16::from(b'\\')]) {
        path.push(u16::from(b'\\'));
    }
    path.push(0);
    let mut bytes = 0;
    // SAFETY: the directory is NUL terminated and the optional outputs are null.
    if unsafe {
        GetDiskFreeSpaceExW(
            path.as_ptr(),
            &mut bytes,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(bytes)
}

#[cfg(unix)]
fn available(dir: &Path) -> io::Result<u64> {
    let s = rustix::fs::statvfs(dir)?;
    Ok(s.f_bavail.saturating_mul(s.f_frsize))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "set FASTTIFF_TEST_SECOND_VOLUME to a writable volume with more free space than the temp volume"]
    fn oversized_stabilization_chooses_the_source_drive_before_writing() {
        use fasttiff_plugin_api::PixelType;
        let source = PathBuf::from(std::env::var_os("FASTTIFF_TEST_SECOND_VOLUME").unwrap())
            .join(format!("ft-capacity-source-{}", std::process::id()));
        std::fs::create_dir(&source).unwrap();
        let dest = ResultDest::for_source(Some(&source.join("recording.tif")));
        let free = available(&std::env::temp_dir()).unwrap();
        let plane = 1024 * 1024 * 2 + 1024;
        let spec = ImageSpec {
            width: 1024,
            height: 1024,
            pixel_type: PixelType::U16,
            channels: 1,
            slices: 1,
            frames: usize::try_from(free / plane + 512).unwrap(),
            channel_colors: vec![],
            metadata: None,
            name: "stabilized".into(),
        };
        assert!(estimate(&spec) > free && estimate(&spec) < available(&source).unwrap());
        let out = allocate(&dest, &spec).unwrap();
        assert!(out.owner.path().starts_with(&source));
        assert_eq!(out.file.metadata().unwrap().len(), 0);
        drop(out);
        std::fs::remove_dir(source).unwrap();
    }
    #[test]
    fn chooses_first_writable_location_with_room() {
        let temp = Path::new("temp");
        let source = Path::new("source");
        for (a, b, expected) in [
            (100, 100, Some(temp)),
            (99, 100, Some(source)),
            (99, 99, None),
        ] {
            let result = choose(
                temp,
                Some(source),
                100,
                |p| Ok(if p == temp { a } else { b }),
                |p| Ok(p.to_owned()),
            );
            assert_eq!(result.ok().as_deref(), expected);
        }
    }
    /// Exercise both TIFF write patterns: appending pixels, and seeking back
    /// to replace the placeholder header after the directory is written.
    #[test]
    fn late_disk_full_resumes_on_source_volume_without_changing_tiff_bytes() {
        use fast_tiff_lib::{SampleType, TiffWriter, WriterOptions};
        use fasttiff_plugin_api::PixelType;
        let spec = ImageSpec {
            width: 512,
            height: 8,
            channels: 1,
            slices: 1,
            frames: 2,
            pixel_type: PixelType::U8,
            channel_colors: vec![],
            metadata: None,
            name: "movie".into(),
        };
        let mut tiff = TiffWriter::new(
            io::Cursor::new(Vec::new()),
            WriterOptions::new(512, 8, SampleType::U8),
        )
        .unwrap();
        for value in [7, 11] {
            tiff.write_frame_u8(&[value; 4096]).unwrap();
        }
        let bytes = tiff.finish().unwrap().into_inner();
        assert!(estimate(&spec) >= bytes.len() as u64);
        let root = std::env::temp_dir().join(format!("ft-late-full-{}", std::process::id()));
        // Set the existing two-volume test variable to exercise a real drive boundary.
        let second = std::env::var_os("FASTTIFF_TEST_SECOND_VOLUME")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join(format!("ft-late-source-{}", std::process::id()));
        std::fs::create_dir(&second).unwrap();
        for patch_header in [false, true] {
            let mut out = allocate(
                &ResultDest::WithFallback {
                    primary: root.clone(),
                    fallback: second.clone(),
                },
                &spec,
            )
            .unwrap();
            let first = out.owner.path().to_owned();
            assert!(first.starts_with(&root));
            let pending = if patch_header {
                out.write_all(&[0; 16]).unwrap();
                out.write_all(&bytes[16..]).unwrap();
                out.rewind().unwrap();
                &bytes[..16]
            } else {
                out.write_all(&bytes[..2048]).unwrap();
                &bytes[2048..]
            };
            let mut fail = true;
            let n = out
                .try_io(pending.len(), |f| {
                    if std::mem::take(&mut fail) {
                        #[cfg(windows)]
                        return Err(io::Error::from_raw_os_error(112));
                        #[cfg(not(windows))]
                        return Err(io::ErrorKind::StorageFull.into());
                    }
                    f.write(pending)
                })
                .unwrap();
            out.write_all(&pending[n..]).unwrap();
            out.flush().unwrap();
            assert!(out.owner.path().starts_with(&second));
            assert!(!first.exists());
            assert_eq!(std::fs::read(out.owner.path()).unwrap(), bytes);
        }
        std::fs::remove_dir(&root).unwrap();
        std::fs::remove_dir(&second).unwrap();
    }
}
