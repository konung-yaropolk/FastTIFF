import { useCallback, useEffect, useRef, useState } from "react";
import { useViewer } from "./useViewer";
import { Toolbar } from "./components/Toolbar";
import { ScrubBar } from "./components/ScrubBar";
import { ChannelPanel } from "./components/ChannelPanel";
import { VolumeSettings } from "./components/VolumeSettings";
import { MetadataPanel } from "./components/MetadataPanel";
import "./App.css";

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const v = useViewer(canvasRef);
  const { viewer, info, redraw, refresh } = v;

  const [volume, setVolume] = useState(false);
  const [panelOpen, setPanelOpen] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [metaOpen, setMetaOpen] = useState(false);
  const [dragging, setDragging] = useState(false);

  // A stack that can't build a volume must not strand the user in 3D — the
  // desktop app drops back to 2D for the same reason.
  useEffect(() => {
    if (volume && info && !info.can_volume) {
      setVolume(false);
      viewer.current?.setViewMode(false);
      redraw();
    }
  }, [volume, info, viewer, redraw]);

  const toggleVolume = useCallback(
    (on: boolean) => {
      setVolume(on);
      viewer.current?.setViewMode(on);
      // Entering 3D stops playback unless the stack has a real time axis.
      if (on && info && !info.is_4d) {
        viewer.current?.setPlaying(false);
        v.setPlaying(false);
      }
      redraw();
    },
    [info, viewer, redraw, v],
  );

  const openFile = useCallback(
    (files: FileList | null) => {
      const f = files?.[0];
      if (f) void v.load(f);
    },
    [v],
  );

  // --- canvas interaction --------------------------------------------------
  const drag = useRef<{ x: number; y: number; button: number } | null>(null);
  const scrollAccum = useRef(0);

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!info) return;
    (e.target as HTMLCanvasElement).setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, button: e.button };
    if (volume && e.button === 0) viewer.current?.beginOrbit();
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = drag.current;
    if (!d || !viewer.current) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const dx = (e.clientX - d.x) * dpr;
    const dy = (e.clientY - d.y) * dpr;
    drag.current = { ...d, x: e.clientX, y: e.clientY };
    if (volume) {
      // Left orbits, middle/right pans — the CAD mapping the desktop defaults to.
      if (d.button === 0) viewer.current.orbitDrag(dx, dy);
      else viewer.current.panDrag(dx, dy, canvasRef.current?.height ?? 1);
      redraw();
    } else {
      v.panBy(dx, dy);
    }
  };

  const endDrag = (e: React.PointerEvent<HTMLCanvasElement>) => {
    drag.current = null;
    (e.target as HTMLCanvasElement).releasePointerCapture?.(e.pointerId);
  };

  // Wheel: zoom in 2D with Ctrl, otherwise scrub frames; fly the camera in 3D.
  // Non-passive so preventDefault actually stops the page scrolling.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (e: WheelEvent) => {
      if (!viewer.current || !info) return;
      e.preventDefault();
      if (volume) {
        viewer.current.wheelFly(-e.deltaY / 100);
        redraw();
        return;
      }
      if (e.ctrlKey || e.metaKey) {
        const rect = canvas.getBoundingClientRect();
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const anchor: [number, number] = [
          (e.clientX - rect.left) * dpr,
          (e.clientY - rect.top) * dpr,
        ];
        v.setZoom(v.zoom * (e.deltaY < 0 ? 1.25 : 0.8), anchor);
        return;
      }
      // Frame scrubbing: accumulate so a touchpad's pixel deltas don't jump.
      const step = e.shiftKey ? Math.max(1, Math.round(info.frames * 0.1)) : 1;
      scrollAccum.current += e.deltaY / 100;
      const whole = Math.trunc(scrollAccum.current);
      if (whole !== 0) {
        scrollAccum.current -= whole;
        v.stepFrame(whole * step);
      }
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [info, volume, v, viewer, redraw]);

  // Arrow keys scrub (2D) or orbit (3D); space toggles playback.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!viewer.current || !info) return;
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (e.code === "Space") {
        e.preventDefault();
        v.setPlaying(!v.playing);
        return;
      }
      const step = e.shiftKey ? Math.max(1, Math.round(info.frames * 0.1)) : 1;
      if (volume) {
        const r = 4;
        if (e.key === "ArrowLeft") viewer.current.orbitDrag(-r, 0);
        else if (e.key === "ArrowRight") viewer.current.orbitDrag(r, 0);
        else if (e.key === "ArrowUp") viewer.current.orbitDrag(0, -r);
        else if (e.key === "ArrowDown") viewer.current.orbitDrag(0, r);
        else return;
        e.preventDefault();
        redraw();
        return;
      }
      if (e.key === "ArrowLeft") v.stepFrame(-step);
      else if (e.key === "ArrowRight") v.stepFrame(step);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [info, volume, v, viewer, redraw]);

  return (
    <div
      className="app"
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        openFile(e.dataTransfer.files);
      }}
    >
      <Toolbar
        info={info}
        volume={volume}
        zoom={v.zoom}
        frame={v.frame}
        onOpen={openFile}
        onToggleVolume={toggleVolume}
        onSettings={() => setSettingsOpen((s) => !s)}
        onMetadata={() => setMetaOpen((s) => !s)}
        onFit={v.fit}
      />

      <div className="stage">
        <canvas
          ref={canvasRef}
          className={volume ? "canvas grab" : "canvas"}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          onContextMenu={(e) => e.preventDefault()}
        />

        {v.status === "loading" && <div className="overlay">Starting the GPU…</div>}
        {v.status === "error" && (
          <div className="overlay error">
            <strong>Could not start the GPU</strong>
            <p>{v.error}</p>
            <p className="hint">
              This viewer needs WebGPU or WebGL2. Try a current Chrome, Edge or Firefox.
            </p>
          </div>
        )}
        {v.status === "ready" && !info && (
          <div className="overlay drop">
            <strong>Drop a TIFF here</strong>
            <p>or use “Open TIFF…” above</p>
            <p className="hint">
              Scroll — frames · Shift+scroll — fast · Ctrl+scroll — zoom · Space — play
            </p>
            <p className="hint">Files are decoded in your browser and never uploaded.</p>
          </div>
        )}
        {volume && info && !v.volumeReady && (
          <div className="overlay">Building the volume…</div>
        )}
        {dragging && <div className="overlay drop active">Release to open</div>}

        {v.error && info && <div className="toast">{v.error}</div>}
        {info?.status && <div className="note">{info.status}</div>}
      </div>

      {settingsOpen && info && (
        <VolumeSettings
          viewer={viewer}
          redraw={redraw}
          onClose={() => setSettingsOpen(false)}
        />
      )}
      {metaOpen && info && (
        <MetadataPanel
          info={info}
          description={viewer.current?.description() ?? null}
          onClose={() => setMetaOpen(false)}
        />
      )}

      <ScrubBar
        info={info}
        frame={v.frame}
        playing={v.playing}
        volume={volume}
        panelOpen={panelOpen}
        onTogglePanel={() => setPanelOpen((p) => !p)}
        onFrame={v.setFrame}
        onStep={v.stepFrame}
        onPlay={v.setPlaying}
        onFps={(fps) => {
          viewer.current?.setFps(fps);
          refresh();
        }}
      />

      {panelOpen && info && (
        <ChannelPanel info={info} viewer={viewer} redraw={redraw} refresh={refresh} />
      )}
    </div>
  );
}
