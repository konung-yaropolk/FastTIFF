//! Channel selection shared by the two stabilization plugins.
use fasttiff_plugin_api::{HostContext, ParamDecl, ParamKind, Plane, PluginError};
use image_registration_support::ChannelMean;

/// How many channels the measurement selector should offer, and their labels.
///
/// With one channel there is exactly one thing the measurement can be, and
/// "mean of one channel" and "channel 1" are the same arithmetic. So a
/// single-channel stack gets a single option — which the host draws greyed out,
/// because a choice of one is not a choice. Offering "Mean" and "Channel 1"
/// separately there would be two names for one behaviour and an invitation to
/// wonder which is which.
fn channel_options(channels: usize) -> Vec<String> {
    if channels <= 1 {
        return vec!["Mean (only channel)".to_string()];
    }
    let mut v = vec!["Mean of all channels".to_string()];
    v.extend((1..=channels).map(|c| format!("Channel {c}")));
    v
}

pub(crate) fn param(channels: usize) -> ParamDecl {
    ParamDecl::new(
        "ref_channel",
        "Measure on",
        ParamKind::Choice {
            default: 0,
            options: channel_options(channels),
        },
    )
    .help(
        "Which channel the transformation is measured on. It is applied to every channel, \
             because the channels are one specimen photographed at once and must stay registered \
             to each other. The mean is the default: averaging the channels raises what the \
             search is matching.",
    )
}

pub(crate) fn measurement(
    host: &mut dyn HostContext,
    channel: Option<usize>,
    channels: usize,
    t: usize,
    out: &mut Vec<f32>,
    scratch: &mut Vec<f32>,
    mean: &mut ChannelMean,
) -> Result<(), PluginError> {
    if let Some(c) = channel {
        return host.read_plane_f32(Plane::new(c, 0, t), out);
    }
    if channels <= 1 {
        return host.read_plane_f32(Plane::new(0, 0, t), out);
    }
    for c in 0..channels {
        host.read_plane_f32(Plane::new(c, 0, t), scratch)?;
        mean.push(scratch);
    }
    mean.finish_into(out);
    Ok(())
}
