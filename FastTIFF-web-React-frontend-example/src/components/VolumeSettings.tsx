import { useState } from "react";
import type { FastTiffViewer } from "../wasm/fasttiff_wasm";

interface Props {
  viewer: React.MutableRefObject<FastTiffViewer | null>;
  redraw: () => void;
  onClose: () => void;
}

const RENDER = ["MIP", "Alpha (DVR)", "Isosurface"];
const INTERP = ["Nearest", "Linear", "Cubic"];
const NAV = ["CAD", "Blender", "Maya", "Minecraft Spectator"];
const NAV_HELP = [
  "Left-drag: orbit · Middle/right-drag: pan · Scroll: fly",
  "Middle-drag: orbit · Shift+Middle: pan · Scroll: fly",
  "Alt+Left: orbit · Alt+Middle: pan · Scroll: fly",
  "Left-drag: look · Scroll: fly",
];

/** The 3D render-settings pop-up, mirroring the desktop's ⚙ window. */
export function VolumeSettings({ viewer, redraw, onClose }: Props) {
  const [render, setRender] = useState(0);
  const [interp, setInterp] = useState(1);
  const [nav, setNav] = useState(0);
  const [density, setDensity] = useState(100);
  const [iso, setIso] = useState(0.1);
  const [scale, setScale] = useState<[number, number, number]>(() => {
    const s = viewer.current?.voxelScale();
    return s ? [s[0], s[1], s[2]] : [1, 1, 1];
  });

  const apply = (fn: () => void) => {
    fn();
    redraw();
  };

  return (
    <aside className="panel">
      <header>
        3D render settings
        <button className="btn icon" onClick={onClose} title="Close">
          ✕
        </button>
      </header>

      <label className="field">
        Mode
        <select
          value={render}
          onChange={(e) => {
            const m = Number(e.target.value);
            setRender(m);
            apply(() => viewer.current?.setVolumeRender(m));
          }}
        >
          {RENDER.map((n, i) => (
            <option key={n} value={i}>
              {n}
            </option>
          ))}
        </select>
      </label>

      {/* Density only affects alpha DVR; iso only the isosurface — same gating
          as the desktop panel, so a slider that does nothing is never shown. */}
      {render === 1 && (
        <label className="field">
          Density {density.toFixed(0)}
          <input
            type="range"
            min={1}
            max={400}
            value={density}
            onChange={(e) => {
              const d = Number(e.target.value);
              setDensity(d);
              apply(() => viewer.current?.setDensity(d));
            }}
          />
        </label>
      )}
      {render === 2 && (
        <label className="field">
          Threshold {iso.toFixed(3)}
          <input
            type="range"
            min={0}
            max={1}
            step={0.001}
            value={iso}
            onChange={(e) => {
              const t = Number(e.target.value);
              setIso(t);
              apply(() => viewer.current?.setIso(t));
            }}
          />
        </label>
      )}

      <label className="field">
        Interpolation
        <select
          value={interp}
          onChange={(e) => {
            const m = Number(e.target.value);
            setInterp(m);
            apply(() => viewer.current?.setVolumeInterp(m));
          }}
        >
          {INTERP.map((n, i) => (
            <option key={n} value={i}>
              {n}
            </option>
          ))}
        </select>
      </label>

      <label className="field">
        Navigation
        <select
          value={nav}
          onChange={(e) => {
            const m = Number(e.target.value);
            setNav(m);
            apply(() => viewer.current?.setNavMode(m));
          }}
        >
          {NAV.map((n, i) => (
            <option key={n} value={i}>
              {n}
            </option>
          ))}
        </select>
      </label>
      <p className="hint">{NAV_HELP[nav]}</p>

      <div className="field">
        Voxel scale
        <div className="row">
          {(["x", "y", "z"] as const).map((axis, i) => (
            <input
              key={axis}
              type="number"
              step={0.1}
              min={0.01}
              value={scale[i]}
              aria-label={`voxel scale ${axis}`}
              onChange={(e) => {
                const next = [...scale] as [number, number, number];
                next[i] = Number(e.target.value) || 0.01;
                setScale(next);
                apply(() => viewer.current?.setVoxelScale(next[0], next[1], next[2]));
              }}
            />
          ))}
        </div>
      </div>

      <button className="btn" onClick={() => apply(() => viewer.current?.resetCamera())}>
        Reset position
      </button>
    </aside>
  );
}
