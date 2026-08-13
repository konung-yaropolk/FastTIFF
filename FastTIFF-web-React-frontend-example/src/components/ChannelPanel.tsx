import { useEffect, useState } from "react";
import type { StackInfo } from "../useViewer";
import type { FastTiffViewer } from "../wasm/fasttiff_wasm";

interface Props {
  info: StackInfo;
  viewer: React.MutableRefObject<FastTiffViewer | null>;
  redraw: () => void;
  refresh: () => void;
}

/**
 * Per-channel contrast + the stack-wide display controls.
 *
 * The slider values are held locally so dragging stays at 60 fps — every change
 * goes straight to Rust and asks for a redraw, and the authoritative
 * `StackInfo` is only re-read when something structural changes (a dimension
 * swap, a LUT change).
 */
export function ChannelPanel({ info, viewer, redraw, refresh }: Props) {
  const [chans, setChans] = useState(info.channel_settings);
  useEffect(() => setChans(info.channel_settings), [info]);

  const push = (i: number, min: number, max: number, enabled: boolean) => {
    setChans((cs) => cs.map((c, j) => (j === i ? { ...c, min, max, enabled } : c)));
    viewer.current?.setChannel(i, min, max, enabled);
    redraw();
  };

  return (
    <section className="channels">
      <div className="row wrap">
        {!info.rgb && info.dimension_options.length > 1 && (
          <label className="field">
            Dimension order
            <select
              value={`${info.channels},${info.slices},${info.frames}`}
              onChange={(e) => {
                const [c, z, t] = e.target.value.split(",").map(Number);
                viewer.current?.setDimensionOrder(c, z, t);
                refresh();
              }}
            >
              {info.dimension_options.map(([c, z, t]) => (
                <option key={`${c},${z},${t}`} value={`${c},${z},${t}`}>
                  {info.has_z_axis ? `c: ${c}  z: ${z}  t: ${t}` : `c: ${c}  t: ${t}`}
                </option>
              ))}
            </select>
          </label>
        )}

        {info.pseudocolor_applicable && (
          <label className="check" title="Tint channels ch1 = red, ch2 = green, ch3 = blue, …">
            <input
              type="checkbox"
              onChange={(e) => {
                viewer.current?.setPseudocolor(e.target.checked);
                refresh();
              }}
            />
            Apply pseudocolor
          </label>
        )}

        {info.lut_selector && (
          <label className="field">
            LUT
            <select
              value={info.lut_selector.selected}
              onChange={(e) => {
                viewer.current?.setLut(Number(e.target.value));
                refresh();
              }}
            >
              {info.lut_selector.options.map((name, i) => (
                <option key={name} value={i}>
                  {name}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

      {/* A palette channel's window is a fixed index -> LUT identity, so there
          is nothing to adjust and the desktop hides its slider too. */}
      {!info.palette &&
        chans.map((c) => (
          <div className="chan" key={c.index}>
            <label className="check fixed">
              <input
                type="checkbox"
                checked={c.enabled}
                onChange={(e) => push(c.index, c.min, c.max, e.target.checked)}
              />
              <span style={c.tint ? { color: c.tint } : undefined}>{c.label}</span>
            </label>

            <div className="range">
              <input
                type="range"
                min={c.lo}
                max={c.hi}
                step={(c.hi - c.lo) / 1000 || 1}
                value={c.min}
                style={c.tint ? ({ accentColor: c.tint } as React.CSSProperties) : undefined}
                onChange={(e) => push(c.index, Math.min(Number(e.target.value), c.max), c.max, c.enabled)}
              />
              <input
                type="range"
                min={c.lo}
                max={c.hi}
                step={(c.hi - c.lo) / 1000 || 1}
                value={c.max}
                style={c.tint ? ({ accentColor: c.tint } as React.CSSProperties) : undefined}
                onChange={(e) => push(c.index, c.min, Math.max(Number(e.target.value), c.min), c.enabled)}
              />
            </div>

            <span className="mono small value">
              {fmt(c.min)} – {fmt(c.max)}
            </span>
          </div>
        ))}
    </section>
  );
}

function fmt(v: number) {
  return Math.abs(v) >= 1000 || Number.isInteger(v) ? v.toFixed(0) : v.toFixed(2);
}
