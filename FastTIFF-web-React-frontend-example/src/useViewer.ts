import { useCallback, useEffect, useRef, useState } from "react";
import init, { FastTiffViewer } from "./wasm/fasttiff_wasm";

/** Mirrors the `StackInfo` the Rust side serializes. */
export interface ChannelInfo {
  index: number;
  label: string;
  min: number;
  max: number;
  lo: number;
  hi: number;
  enabled: boolean;
  tint: string | null;
}

export interface LutSelector {
  selected: number;
  options: string[];
}

export interface StackInfo {
  name: string;
  width: number;
  height: number;
  bits: number;
  frames: number;
  channels: number;
  slices: number;
  rgb: boolean;
  palette: boolean;
  can_volume: boolean;
  is_4d: boolean;
  fps: number;
  status: string | null;
  frame_interval_s: number | null;
  channel_settings: ChannelInfo[];
  pseudocolor_applicable: boolean;
  lut_selector: LutSelector | null;
  dimension_options: [number, number, number][];
  has_z_axis: boolean;
}

type Status = "loading" | "ready" | "error";

/**
 * Owns the wasm viewer bound to `canvasRef`, plus the animation loop.
 *
 * The render loop is deliberately demand-driven rather than a permanent rAF:
 * a static 2D frame needs no redraws at all, so the loop parks itself and any
 * mutation calls `redraw()`. Playback and in-flight volume builds keep it going
 * by returning `true` from the Rust `render()`.
 */
export function useViewer(canvasRef: React.RefObject<HTMLCanvasElement>) {
  const viewer = useRef<FastTiffViewer | null>(null);
  const [status, setStatus] = useState<Status>("loading");
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<StackInfo | null>(null);
  const [frame, setFrameState] = useState(0);
  const [playing, setPlayingState] = useState(false);
  const [zoom, setZoomState] = useState(1);
  // Mirrors the Rust side's "a volume is uploaded" flag. It has to live in
  // React state, not be read inline during render: the volume finishes building
  // inside the animation loop, which React doesn't observe, so an inline read
  // would leave the loading overlay up forever.
  const [volumeReady, setVolumeReady] = useState(false);

  // Loop bookkeeping, in refs so callbacks never go stale.
  const running = useRef(false);
  const pending = useRef(false);
  const pan = useRef<[number, number]>([0, 0]);
  const zoomRef = useRef(1);
  const volumeReadyRef = useRef(false);

  /** Ask for one more frame; the loop keeps itself alive while Rust wants it. */
  const redraw = useCallback(() => {
    if (!viewer.current || pending.current) return;
    pending.current = true;
    requestAnimationFrame(function step() {
      pending.current = false;
      const v = viewer.current;
      if (!v) return;
      const more = v.render();
      // Playback advances in Rust; mirror the frame index into React so the
      // scrubber and counter follow along.
      if (v.isPlaying()) {
        v.tickPlayback(performance.now() / 1000);
        setFrameState(v.frameIndex());
      }
      // Only push a state update when it actually flips — this runs every frame.
      const ready = v.volumeReady();
      if (volumeReadyRef.current !== ready) {
        volumeReadyRef.current = ready;
        setVolumeReady(ready);
      }
      if (more) {
        pending.current = true;
        requestAnimationFrame(step);
      }
    });
  }, []);

  // --- one-time wasm init -------------------------------------------------
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await init();
        const canvas = canvasRef.current;
        if (!canvas || cancelled) return;
        sizeCanvas(canvas);
        const v = await FastTiffViewer.create(canvas);
        if (cancelled) {
          v.free();
          return;
        }
        viewer.current = v;
        v.resize(canvas.width, canvas.height);
        running.current = true;
        setStatus("ready");
        redraw();
      } catch (e) {
        if (!cancelled) {
          setError(describe(e));
          setStatus("error");
        }
      }
    })();
    return () => {
      cancelled = true;
      running.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- keep the drawing buffer matched to the element ---------------------
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      if (!viewer.current) return;
      sizeCanvas(canvas);
      viewer.current.resize(canvas.width, canvas.height);
      redraw();
    });
    ro.observe(canvas);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [redraw]);

  /** Re-read `StackInfo` after anything that can change it. */
  const refresh = useCallback(() => {
    const v = viewer.current;
    if (!v) return;
    setInfo((v.info() as StackInfo | null) ?? null);
    setFrameState(v.frameIndex());
    redraw();
  }, [redraw]);

  const load = useCallback(
    async (file: File) => {
      const v = viewer.current;
      if (!v) return;
      setError(null);
      try {
        const bytes = new Uint8Array(await file.arrayBuffer());
        const next = v.load(bytes, file.name) as StackInfo;
        setInfo(next);
        setFrameState(0);
        setPlayingState(false);
        // Open at fit-to-canvas, like the desktop app's initial fit.
        const fit = Math.min(1, v.fitZoom());
        zoomRef.current = fit;
        pan.current = [0, 0];
        v.setZoomPan(fit, 0, 0);
        setZoomState(fit);
        redraw();
      } catch (e) {
        setError(describe(e));
      }
    },
    [redraw],
  );

  const setFrame = useCallback(
    (i: number) => {
      viewer.current?.setFrame(i);
      setFrameState(viewer.current?.frameIndex() ?? 0);
      redraw();
    },
    [redraw],
  );

  const stepFrame = useCallback(
    (d: number) => {
      viewer.current?.stepFrame(d);
      setFrameState(viewer.current?.frameIndex() ?? 0);
      redraw();
    },
    [redraw],
  );

  const setPlaying = useCallback(
    (p: boolean) => {
      viewer.current?.setPlaying(p);
      setPlayingState(p);
      // Seed the clock so the first tick doesn't jump by however long we sat idle.
      if (p) viewer.current?.tickPlayback(performance.now() / 1000);
      redraw();
    },
    [redraw],
  );

  const setZoom = useCallback(
    (z: number, anchor?: [number, number]) => {
      const v = viewer.current;
      if (!v) return;
      const prev = zoomRef.current;
      const next = Math.min(64, Math.max(0.02, z));
      // Keep the point under the cursor fixed, as the desktop zoom does.
      if (anchor) {
        const [ax, ay] = anchor;
        pan.current = [
          (pan.current[0] + ax) * (next / prev) - ax,
          (pan.current[1] + ay) * (next / prev) - ay,
        ];
      }
      zoomRef.current = next;
      v.setZoomPan(next, pan.current[0], pan.current[1]);
      const [mx, my] = v.panRange();
      pan.current = [clamp(pan.current[0], 0, mx), clamp(pan.current[1], 0, my)];
      v.setZoomPan(next, pan.current[0], pan.current[1]);
      setZoomState(next);
      redraw();
    },
    [redraw],
  );

  const panBy = useCallback(
    (dx: number, dy: number) => {
      const v = viewer.current;
      if (!v) return;
      const [mx, my] = v.panRange();
      pan.current = [clamp(pan.current[0] - dx, 0, mx), clamp(pan.current[1] - dy, 0, my)];
      v.setZoomPan(zoomRef.current, pan.current[0], pan.current[1]);
      redraw();
    },
    [redraw],
  );

  const fit = useCallback(() => {
    const v = viewer.current;
    if (!v) return;
    pan.current = [0, 0];
    setZoom(v.fitZoom());
  }, [setZoom]);

  return {
    viewer,
    status,
    error,
    info,
    frame,
    playing,
    zoom,
    volumeReady,
    redraw,
    refresh,
    load,
    setFrame,
    stepFrame,
    setPlaying,
    setZoom,
    panBy,
    fit,
    setError,
  };
}

/** Match the drawing buffer to the element's CSS size in device pixels. */
function sizeCanvas(canvas: HTMLCanvasElement) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
}

function clamp(v: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, v));
}

function describe(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}
