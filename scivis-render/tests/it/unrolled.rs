//! The per-channel code the shaders unroll by hand, counted against
//! `MAX_CHANNELS`.
//!
//! The channel count reaches both shading languages as a constant (see
//! `scivis-render/src/shaders.rs`), so the arrays, the LUT rows and the
//! composite's texture bindings follow it. What cannot follow it is code
//! written out once per channel: WGSL has no arrays of textures and GLSL 1.40
//! indexes an array of samplers only with a constant, so the composite reads
//! each channel's textures on a line of its own, and the volume shaders read
//! the channels out of the components of two packed textures, one line each.
//! Raising `MAX_CHANNELS` compiles and validates with those lines as they are,
//! and the new channels are simply never drawn. These checks fail instead.
//!
//! They also catch the other way such a line goes wrong, which compiles in
//! both languages and draws: a channel's line that takes one of its values
//! from a neighbour's slot (`ch_params[4]` on channel 5's line, say). Every
//! channel index on a per-channel line has to be that line's own.

use scivis_render::MAX_CHANNELS;
use std::collections::BTreeMap;

fn word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// How often each channel index appears in `src` between `before` and
/// `after`, as in `on(3)` or `ch3_tex`. What is between them must be a
/// number to count: `ch_tex[MAX_CHANNELS]` is not a use of a channel. A
/// `before` that starts with a letter must start a word, so that `on(` is not
/// found in `location(0)`; an empty `after` must end one, so that `m` and
/// digits is the variable `m4` and not the start of `m4x`.
pub fn uses(src: &str, before: &str, after: &str) -> BTreeMap<usize, usize> {
    let wordy = before.bytes().next().is_some_and(word);
    let mut found = BTreeMap::new();
    let mut rest = src;
    while let Some(at) = rest.find(before) {
        let starts_word = !wordy || at == 0 || !word(rest.as_bytes()[at - 1]);
        rest = &rest[at + before.len()..];
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        let ends = if after.is_empty() {
            !rest.as_bytes().get(digits).copied().is_some_and(word)
        } else {
            rest[digits..].starts_with(after)
        };
        if starts_word && digits > 0 && ends {
            *found.entry(rest[..digits].parse().unwrap()).or_default() += 1;
        }
    }
    found
}

/// The arguments of every call of `function` in `src`, each trimmed, split at
/// the commas outside any nested parentheses or brackets. A definition is
/// found too (`vec3 lut_col(float t, int idx)`); its arguments are not
/// numbers, which is how the counts below pass over it.
pub fn calls<'a>(src: &'a str, function: &str) -> Vec<Vec<&'a str>> {
    let open = format!("{function}(");
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(at) = src[from..].find(&open).map(|i| from + i) {
        from = at + open.len();
        if at > 0 && word(src.as_bytes()[at - 1]) {
            continue;
        }
        let (mut depth, mut start, mut args) = (0usize, from, Vec::new());
        for (i, b) in src.bytes().enumerate().skip(from) {
            match b {
                b'(' | b'[' => depth += 1,
                b')' | b']' if depth > 0 => depth -= 1,
                b')' => {
                    args.push(src[start..i].trim());
                    break;
                }
                b',' if depth == 0 => {
                    args.push(src[start..i].trim());
                    start = i + 1;
                }
                _ => {}
            }
        }
        found.push(args);
    }
    found
}

/// Which argument of a call names the channel.
#[derive(Clone, Copy)]
pub enum Arg {
    At(usize),
    Last,
}

/// The channel index each call of `function` in `src` gives as its `arg`,
/// counted: as [`uses`], for an index that is an argument.
pub fn arg_uses(src: &str, function: &str, arg: Arg) -> BTreeMap<usize, usize> {
    let mut found = BTreeMap::new();
    for args in calls(src, function) {
        let given = match arg {
            Arg::At(i) => args.get(i),
            Arg::Last => args.last(),
        };
        if let Some(n) = given.and_then(|a| a.parse::<usize>().ok()) {
            *found.entry(n).or_default() += 1;
        }
    }
    found
}

/// That `found` (`use` in the shader file `what`, by channel) covers every
/// channel there is, and no more: each of `0..MAX_CHANNELS` the same number
/// of times, at least once.
fn assert_covers(what: &str, used: &str, found: &BTreeMap<usize, usize>) {
    let counts: Vec<usize> = found.values().copied().collect();
    let channels: Vec<usize> = found.keys().copied().collect();
    assert!(
        channels == (0..MAX_CHANNELS).collect::<Vec<_>>()
            && counts.windows(2).all(|w| w[0] == w[1]),
        "{what} writes {used} out by hand for channels {found:?} (channel: times), \
         but MAX_CHANNELS is {MAX_CHANNELS}: write it out for each channel 0..{MAX_CHANNELS}, \
         the same number of times"
    );
}

/// That `src` (the shader file `what`) has its per-channel code written out
/// for every channel there is, and for no more: each of `0..MAX_CHANNELS`
/// between `before` and `after` the same number of times, at least once.
pub fn assert_every_channel(what: &str, src: &str, before: &str, after: &str) {
    assert_covers(
        what,
        &format!("`{before}N{after}`"),
        &uses(src, before, after),
    );
}

/// As [`assert_every_channel`], for the channel index that is argument `arg`
/// of each call of `function`.
pub fn assert_every_channel_arg(what: &str, src: &str, function: &str, arg: Arg) {
    let which = match arg {
        Arg::At(i) => format!("argument {i}"),
        Arg::Last => "the last argument".to_string(),
    };
    assert_covers(
        what,
        &format!("`{function}(...)` with the channel as {which}"),
        &arg_uses(src, function, arg),
    );
}

/// Every channel index a line of shader code names, by the ways the shaders
/// name one: `on(N)`, `[N]`, `chN_tex`, `chN_ftex`, the MIP's `mN`, and the
/// channel argument of `norm_ch`, `lut_col`, `albedo_t` and `apply_channel`.
fn channels_named(line: &str) -> BTreeMap<usize, usize> {
    let mut named = BTreeMap::new();
    let mut add = |found: BTreeMap<usize, usize>| {
        for (channel, times) in found {
            *named.entry(channel).or_default() += times;
        }
    };
    add(uses(line, "on(", ")"));
    add(uses(line, "[", "]"));
    add(uses(line, "ch", "_tex"));
    add(uses(line, "ch", "_ftex"));
    add(uses(line, "m", ""));
    add(uses(line, "albedo_t(", ")"));
    add(arg_uses(line, "norm_ch", Arg::Last));
    add(arg_uses(line, "lut_col", Arg::Last));
    add(arg_uses(line, "apply_channel", Arg::At(1)));
    named
}

/// The channel a guard `num_channels > N` (`> Nu` in WGSL) in `line` is for:
/// the line it guards draws channel `N`.
fn guard(line: &str) -> Option<usize> {
    let rest = &line[line.find("num_channels > ")? + "num_channels > ".len()..];
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    rest[..digits].parse().ok()
}

/// That each per-channel statement of `src` (the shader file `what`) is
/// about one channel only, and that there are as many for each channel.
///
/// A per-channel statement is a line that starts `if (on(N))` (the volume
/// shaders) or composites a channel with `apply_channel(…, N, …)` (the
/// composite, with its `num_channels > N` guard, which WGSL puts on the line
/// before). Every channel index it names must be that one: a line that reads
/// a neighbour's window, LUT row, texture or running maximum compiles, and
/// draws one channel with another's settings.
pub fn assert_one_channel_a_statement(what: &str, src: &str) {
    let mut statements = BTreeMap::<usize, usize>::new();
    let mut guarded: Option<&str> = None;
    for line in src.lines() {
        let code = line.split("//").next().unwrap_or_default().trim();
        if code.contains("num_channels > ") && code.ends_with('{') {
            guarded = Some(code);
            continue;
        }
        let statement = match guarded.take() {
            Some(guard_line) => format!("{guard_line} {code}"),
            None => code.to_string(),
        };
        let per_channel = statement.starts_with("if (on(")
            || arg_uses(&statement, "apply_channel", Arg::At(1))
                .keys()
                .next()
                .is_some();
        if !per_channel {
            continue;
        }
        let mut named = channels_named(&statement);
        if let Some(n) = guard(&statement) {
            *named.entry(n).or_default() += 1;
        }
        let channels: Vec<usize> = named.keys().copied().collect();
        let [channel] = channels[..] else {
            panic!(
                "{what}: a per-channel line names channels {channels:?}, not one:\n  {statement}"
            );
        };
        *statements.entry(channel).or_default() += 1;
    }
    assert_covers(what, "its per-channel lines", &statements);
}

/// That each read `{prefix}T.C, N)` in `src` (the volume shader `what`),
/// where `T` is the packed texture (`a` or `b`) and `C` its component, reads
/// channel `N` from where the packer put it, and that every channel is read.
/// The volume shaders take each channel out of a component by hand; one read
/// from the wrong component draws the wrong channel, valid in both languages.
pub fn assert_reads_where_packed(what: &str, src: &str, prefix: &str) {
    use scivis_render::pack::{TexelFormat, VolumeLayout};
    let layout = VolumeLayout::new((1, 1, 1), MAX_CHANNELS, TexelFormat::Unorm8);
    let mut seen = [false; MAX_CHANNELS];
    let mut rest = src;
    while let Some(at) = rest.find(prefix) {
        rest = &rest[at + prefix.len()..];
        let read: String = rest.chars().take_while(|&c| c != ')').collect();
        let (Some((texture, component)), Some(channel)) = (
            read.split_once(',').and_then(|(tc, _)| tc.split_once('.')),
            read.rsplit_once(", ")
                .and_then(|(_, n)| n.parse::<usize>().ok()),
        ) else {
            continue;
        };
        assert!(
            channel < MAX_CHANNELS,
            "{what} reads channel {channel} in `{prefix}{read})`, past MAX_CHANNELS"
        );
        let (t, k) = layout.place(channel);
        let want = format!("{}.{}", ["a", "b"][t], ["x", "y", "z", "w"][k]);
        assert_eq!(
            format!("{texture}.{component}"),
            want,
            "{what} reads channel {channel} from `{texture}.{component}`, but the packer puts it in `{want}`"
        );
        seen[channel] = true;
    }
    let missing: Vec<usize> = (0..MAX_CHANNELS).filter(|&c| !seen[c]).collect();
    assert!(
        missing.is_empty(),
        "{what} never reads channels {missing:?} (`{prefix}`)"
    );
}

/// The checks above, held to what they claim on code made to fail them.
#[test]
fn the_counts_catch_a_line_that_borrows_a_channel() {
    // Channel 1's line reading channel 0's window: the counts are uneven,
    // and the line names two channels.
    let src = "if (on(0)) { t = norm_ch(a.x, 0); m0 = lut_col(t, 0); }\n\
               if (on(1)) { t = norm_ch(a.y, 0); m1 = lut_col(t, 1); }\n";
    let borrowed = std::panic::catch_unwind(|| assert_one_channel_a_statement("test", src));
    let message = *borrowed.unwrap_err().downcast::<String>().unwrap();
    assert!(message.contains("names channels [0, 1]"), "{message}");
    assert_eq!(
        arg_uses(src, "norm_ch", Arg::Last),
        BTreeMap::from([(0, 2)])
    );
    // A definition is a call with no number for an argument.
    assert!(arg_uses("vec3 lut_col(float t, int idx) {", "lut_col", Arg::Last).is_empty());
    // A nested call's commas are its own.
    assert_eq!(
        calls("apply_channel(f(a, b), 3, x[3])", "apply_channel"),
        [["f(a, b)", "3", "x[3]"]]
    );
    // `m4` is a channel's maximum; `max` and `m4x` are not.
    assert_eq!(uses("max(m4, m4x)", "m", ""), BTreeMap::from([(4, 1)]));
}
