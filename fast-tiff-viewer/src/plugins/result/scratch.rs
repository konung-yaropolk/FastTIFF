//! Decide where to write before accepting the first plane.
use super::{create_file, file_stem, ImageSpec, ResultDest, ScratchFile};
use anyhow::Context;
use std::{
    fs::File,
    io::{self, Seek, SeekFrom, Write},
    path::Path,
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
}

impl Write for DiskOutput {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.file.write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

impl Seek for DiskOutput {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.file.seek(pos)
    }
}

pub(super) fn allocate(dest: &ResultDest, spec: &ImageSpec) -> anyhow::Result<DiskOutput> {
    let source;
    let (dir, original) = match dest {
        ResultDest::Dir(dir) => {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("Creating temporary TIFF directory {}", dir.display()))?;
            (dir.as_path(), None)
        }
        ResultDest::BesideSource(path) => {
            source = std::path::absolute(path)?;
            (
                source.parent().context("Missing source directory")?,
                Some(source.as_path()),
            )
        }
        ResultDest::Memory => unreachable!(),
    };
    let needed = estimate(spec);
    let free = available(dir)
        .with_context(|| format!("Checking space for a temporary TIFF in {}", dir.display()))?;
    anyhow::ensure!(
        free >= needed,
        "Temporary TIFF needs {needed} bytes. {} has {free} bytes free",
        dir.display()
    );
    let (file, owner) = create_file(dir, &file_stem(&spec.name), original)
        .with_context(|| format!("Creating temporary TIFF in {}", dir.display()))?;
    Ok(DiskOutput { file, owner })
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
