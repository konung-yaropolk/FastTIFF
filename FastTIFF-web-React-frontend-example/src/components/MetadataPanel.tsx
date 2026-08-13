import type { StackInfo } from "../useViewer";

interface Props {
  info: StackInfo;
  description: string | null;
  onClose: () => void;
}

/** The file-metadata pop-up, mirroring the desktop's ( i ) window. */
export function MetadataPanel({ info, description, onClose }: Props) {
  const rows: [string, string][] = [
    ["File", info.name],
    ["Dimensions", `${info.width} × ${info.height} px`],
    ["Bit depth", `${info.bits}-bit`],
    ["Channels", info.rgb ? "RGB (3 sample planes)" : String(info.channels)],
    ["Z-slices", String(info.slices)],
    ["Frames", String(info.frames)],
    ["Palette", info.palette ? "yes (indexed)" : "no"],
    ["Playback", `${info.fps} fps`],
  ];
  if (info.frame_interval_s != null) {
    rows.push(["Frame interval", `${info.frame_interval_s} s`]);
  }

  return (
    <aside className="panel">
      <header>
        File metadata
        <button className="btn icon" onClick={onClose} title="Close">
          ✕
        </button>
      </header>
      <table className="kv">
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k}>
              <th>{k}</th>
              <td>{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {description && (
        <>
          <h4>ImageDescription</h4>
          <pre className="desc">{description}</pre>
        </>
      )}
    </aside>
  );
}
