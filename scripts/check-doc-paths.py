#!/usr/bin/env python3
"""Fail on comments and docs that name a file this repository does not have.

Rationale comments here often say why something is the way it is by pointing at
another file ("see `tests/it/plugins.rs`", "the same reason as
`FastTIFF-web/crate`"). Nothing checks those pointers. When the file they name
is moved or deleted, the comment goes on reading as true.

This looks at the comments of source files and manifests, and at the whole of
Markdown files. It collects every token that looks like a path to a file in the
repository, and reports the ones that resolve to nothing. A reference resolves
if it names a tracked file or directory relative to the file it is in, to that
file's crate, or to the repository root. A bare file name (`decode.rs`)
resolves if any tracked file has that name.

Standard library only, so it runs wherever CI has a Python 3.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Extensions of files this repository is made of. A token ending in one of
# these is taken to be a path; anything else (`foo.bar`, `v0.15.0`, `wgpu.rs`
# as a URL host) is not looked at.
EXTS = ('rs', 'toml', 'md', 'yml', 'yaml', 'sh', 'py', 'wgsl', 'h', 'json',
        'html', 'svg', 'ico', 'bat', 'ps1', 'lock', 'txt')

# References that name a file outside this repository on purpose, or one a
# build or a program creates. Each is matched against the reference exactly.
ALLOWED = {
    # Not files at all: the documentation host.
    'docs.rs',
    # Generated: by the Flatpak build, and by the example code that names them.
    'generated-sources.json', 'flatpak-cargo-generator.py', 'recording.txt',
    # C headers: the system's, libtiff's (built from source by the bench), and
    # the NIfTI-1 reference header the format is specified by.
    'math.h', 'stdint.h', 'stdio.h', 'stdlib.h',
    'tif_config.h', 'tiffconf.h', 'tiffvers.h', 'nifti1.h',
    # Upstream sources a port is compared against: Rust's standard library,
    # suite2p's Python, TurboReg's licence.
    'library/std/src/sys/stdio/windows.rs', 'register.py', 'LICENSE.txt',
    # Directories a build or an install creates: cargo's, wasm-pack's, the
    # fuzzer's, the bench's downloaded libtiff, and the install tree's.
    'target', 'target/release', 'pkg', './pkg', 'dist', 'fuzz/corpus', 'vendor',
    'apps', 'mimetypes',
}

TOKEN = re.compile(
    r'(?<![\w./\\-])'                        # not in the middle of a word or path
    r'((?:\.{1,2}/)?(?:[\w-]+/)*[\w-]+(?:\.[\w-]+)*\.(?:' + '|'.join(EXTS) + r'))'
    r'(?![\w/-])'
)
# A directory, written as code: `plugins/builtin/`. Only in backticks, where it
# is unambiguously a path, and only with its trailing slash.
DIRREF = re.compile(r'`((?:\.{1,2}/)?(?:[\w.-]+/)+)`')
URL = re.compile(r'\b[a-z][a-z0-9+.-]*://\S+')


def tracked():
    out = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, capture_output=True, check=True).stdout
    return [p for p in out.decode('utf-8').split('\0') if p]


def comment_text(path, text):
    """The part of a file whose references are prose rather than code."""
    ext = path.rsplit('.', 1)[-1]
    if ext == 'md':
        return text
    lines = []
    if ext in ('rs', 'wgsl', 'h'):
        # `//` comments (doc comments included) and `/* */` blocks. A `//`
        # inside a string literal is rare enough here to accept.
        for m in re.finditer(r'//[^\n]*|/\*.*?\*/', text, flags=re.S):
            lines.append(m.group(0))
    elif ext in ('toml', 'yml', 'yaml', 'sh', 'py'):
        for line in text.split('\n'):
            i = line.find('#')
            if i >= 0:
                lines.append(line[i:])
        if ext == 'py':
            lines += re.findall(r'"""(.*?)"""', text, flags=re.S)
    return '\n'.join(lines)


def crate_root(path, files_set):
    d = os.path.dirname(path)
    while True:
        if (d + '/Cargo.toml' if d else 'Cargo.toml') in files_set:
            return d
        if not d:
            return ''
        d = os.path.dirname(d)


def main():
    files = tracked()
    files_set = set(files)
    dirs = set()
    for f in files:
        d = os.path.dirname(f)
        while d:
            dirs.add(d)
            d = os.path.dirname(d)
    names = {os.path.basename(p) for p in files_set | dirs}
    everything = files_set | dirs

    def exists(rel):
        rel = os.path.normpath(rel).replace('\\', '/')
        return rel in files_set or rel in dirs

    stale = []
    for f in files:
        if f.rsplit('.', 1)[-1] not in ('rs', 'toml', 'md', 'yml', 'yaml', 'sh', 'py', 'wgsl', 'h'):
            continue
        try:
            text = open(os.path.join(ROOT, f), encoding='utf-8').read()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        prose = URL.sub(' ', comment_text(f, text))
        here = os.path.dirname(f)
        crate = crate_root(f, files_set)
        refs = [m.group(1) for m in TOKEN.finditer(prose)]
        refs += [m.group(1).rstrip('/') for m in DIRREF.finditer(prose)]
        for ref in refs:
            if ref in ALLOWED:
                continue
            if any(exists(os.path.join(base, ref)) for base in (here, crate, '')):
                continue
            if '/' not in ref and ref in names:
                continue
            if not ref.startswith('.') and any(p.endswith('/' + ref) for p in everything):
                # Written relative to some crate other than this file's, as a
                # workflow at the root naming `deconvolve/par.rs` does.
                continue
            for n, line in enumerate(text.split('\n'), 1):
                if ref in line:
                    stale.append(f'{f}:{n}: names `{ref}`, which this repository does not have')
    for s in sorted(set(stale)):
        print(s)
    if stale:
        print(f'\n{len(set(stale))} reference(s) to files that do not exist. Point them at the file '
              'that now holds the thing, or drop them; if one names a file outside this '
              'repository on purpose, add it to ALLOWED in scripts/check-doc-paths.py.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
