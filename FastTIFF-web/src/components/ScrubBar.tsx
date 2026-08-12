import type { StackInfo } from "../useViewer";

interface Props {
  info: StackInfo | null;
  frame: number;
  playing: boolean;
  volume: boolean;
  panelOpen: boolean;
  onTogglePanel: () => void;
  onFrame: (i: number) => void;
  onStep: (d: number) => void;
  onPlay: (p: boolean) => void;
  onFps: (fps: number) => void;
}

export function ScrubBar(p: Props) {
  const { info } = p;
  if (!info) {
    return (
      <footer className="scrub">
        <span className="meta weak">Open a TIFF stack to begin.</span>
      </footer>
    );
  }

  const max = Math.max(0, info.frames - 1);
  // In 3D the frame axis is the volume's depth, so play/scrub only apply when
  // the stack also has a separate time axis.
  const navEnabled = info.frames > 1 && !(p.volume && !info.is_4d);

  return (
    <footer className="scrub">
      <button
        className="btn icon"
        onClick={p.onTogglePanel}
        title="Show/hide channel & contrast settings"
      >
        {p.panelOpen ? "▲" : "▼"}
      </button>
      <button
        className="btn icon"
        disabled={!navEnabled}
        onClick={() => p.onPlay(!p.playing)}
        title="Play/pause looped movie (Space)"
      >
        {p.playing ? "❚❚" : "▶"}
      </button>
      <button className="btn icon" disabled={!navEnabled} onClick={() => p.onFrame(0)} title="First frame">
        |◀
      </button>
      <button className="btn icon" disabled={!navEnabled} onClick={() => p.onStep(-1)} title="Previous frame (←)">
        ◀
      </button>

      <input
        className="slider"
        type="range"
        min={0}
        max={max}
        value={Math.min(p.frame, max)}
        disabled={!navEnabled}
        onChange={(e) => p.onFrame(Number(e.target.value))}
      />

      <button className="btn icon" disabled={!navEnabled} onClick={() => p.onStep(1)} title="Next frame (→)">
        ▶
      </button>
      <button className="btn icon" disabled={!navEnabled} onClick={() => p.onFrame(max)} title="Last frame">
        ▶|
      </button>

      {navEnabled && (
        <label className="fps" title="Playback speed (frames per second)">
          <input
            type="number"
            min={0.1}
            max={1000}
            step={1}
            defaultValue={info.fps}
            onChange={(e) => p.onFps(Number(e.target.value))}
          />
          fps
        </label>
      )}
    </footer>
  );
}
