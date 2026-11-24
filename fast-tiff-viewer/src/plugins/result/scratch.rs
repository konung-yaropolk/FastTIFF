//! Decide where to write before accepting the first plane.
use super::{create_file, file_stem, ImageSpec, ResultDest, ScratchFile};
use std::{
    fs::File,
    io,
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

pub(super) fn allocate(dest: &ResultDest, spec: &ImageSpec) -> anyhow::Result<(File, ScratchFile)> {
    let (root, fallback) = match dest {
        ResultDest::Dir(dir) => (dir, None),
        ResultDest::WithFallback { primary, fallback } => (primary, Some(fallback)),
        ResultDest::Memory => unreachable!(),
    };
    let needed = estimate(spec);
    let stem = file_stem(&spec.name);
    choose(
        root,
        fallback.map(PathBuf::as_path),
        needed,
        |dir| {
            if dir == root {
                std::fs::create_dir_all(dir)?;
            }
            available(dir)
        },
        |dir| {
            let (file, path) = create_file(dir, &stem)?;
            Ok((file, ScratchFile::own(path)?))
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
                Ok(file) => return Ok(file),
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
    let path: Vec<u16> = dir
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
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
}
