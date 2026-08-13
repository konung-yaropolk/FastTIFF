import type { StackInfo } from "../useViewer";

interface Props {
  info: StackInfo | null;
  volume: boolean;
  zoom: number;
  frame: number;
  onOpen: (files: FileList | null) => void;
  onToggleVolume: (on: boolean) => void;
  onSettings: () => void;
  onMetadata: () => void;
  onFit: () => void;
}

/** Zoom as the desktop shows it: up to 2 decimals, trailing zeros trimmed. */
function pct(zoom: number) {
  const s = (zoom * 100).toFixed(2);
  return `${s.replace(/\.?0+$/, "")}%`;
}

export function Toolbar(p: Props) {
  const { info } = p;
  // In 3D the frame axis becomes the volume's depth, so the counter and time
  // only mean something when the stack has a separate time axis.
  const showFrames = info && !(p.volume && !info.is_4d);

  return (
    <header className="toolbar">
      <label className="btn primary">
        Open TIFF…
        <input
          type="file"
          accept=".tif,.tiff"
          hidden
          onChange={(e) => {
            p.onOpen(e.target.files);
            e.currentTarget.value = "";
          }}
        />
      </label>

      {info && (
        <>
          <span className="sep" />
          <div className="segmented">
            <button
              className={!p.volume ? "on" : ""}
              onClick={() => p.onToggleVolume(false)}
              title="Movie (2D) view"
            >
              2D
            </button>
            <button
              className={p.volume ? "on" : ""}
              disabled={!info.can_volume}
              onClick={() => p.onToggleVolume(true)}
              title={
                info.can_volume
                  ? "Volume (3D) view — drag to rotate, scroll to fly"
                  : "Needs at least two frames"
              }
            >
              3D
            </button>
          </div>
          <button
            className="btn icon"
            disabled={!info.can_volume}
            onClick={p.onSettings}
            title="3D render settings"
          >
            ⚙
          </button>

          <span className="sep" />
          {!p.volume && (
            <>
              <button className="btn ghost mono" onClick={p.onFit} title="Fit to window">
                {pct(p.zoom)}
              </button>
              <span className="sep" />
            </>
          )}
          <span className="meta">
            {info.width}×{info.height} px, {info.bits}-bit,{" "}
            {info.rgb ? "RGB" : `${info.channels} channel(s)`}
          </span>

          {showFrames && (
            <>
              <span className="sep" />
              <span className="mono">
                Frame {String(p.frame + 1).padStart(String(info.frames).length, " ")} /{" "}
                {info.frames}
              </span>
              {info.frame_interval_s != null && (
                <>
                  <span className="sep" />
                  <span className="mono">
                    t = {(p.frame * info.frame_interval_s).toFixed(3)}s
                  </span>
                </>
              )}
            </>
          )}

          <span className="spacer" />
          <button className="btn ghost" onClick={p.onMetadata} title="See metadata">
            ( i )
          </button>
        </>
      )}

      {!info && (
        <>
          <span className="sep" />
          <span className="meta weak">FastTIFF for the web — WebGPU / WebGL2</span>
        </>
      )}
    </header>
  );
}
