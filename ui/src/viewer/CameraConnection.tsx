/**
 * Notes under the camera URL in Machine config → Connections: which app the
 * server found on the applied URL (detected, not chosen — `camera.ts`), and a
 * warning when the camera URL points at the ESP32's address (two devices on
 * one IP: the camera calls would hit the firmware, or the other way round).
 */

import { useViewerStore } from './modelStore';
import { CAMERA_KIND_LABEL, cameraKind, cameraNode, urlHost } from './camera';

export function CameraConnectionNotes({
  cameraUrl,
  espIp,
}: {
  /** The camera URL field as typed (may not be applied yet). */
  cameraUrl: string;
  /** The ESP IP field as typed. */
  espIp: string;
}) {
  const kind = useViewerStore((s) => cameraKind(s.model));
  const node = useViewerStore((s) => cameraNode(s.model));
  const resolvedEsp = useViewerStore((s) =>
    typeof s.model.esp_ip === 'string' ? (s.model.esp_ip as string) : '',
  );

  const camHost = urlHost(cameraUrl);
  const clash = camHost !== '' && [espIp, resolvedEsp].some((ip) => urlHost(ip) === camHost);

  const detail =
    kind === 'capture_node' && node
      ? ` ${node.version ? `v${node.version} ` : ''}· ${node.state === 'ready' ? node.mode : node.state}`
      : '';
  return (
    <>
      <div className="pl-[6.5rem] text-[11px] text-inkmute">
        type:{' '}
        <span className={kind === 'capture_node' || kind === 'ipwebcam' ? 'text-inkdim' : 'text-amber-300/90'}>
          {CAMERA_KIND_LABEL[kind]}
          {detail}
        </span>{' '}
        <span className="text-inkmute">(detected)</span>
      </div>
      {clash && (
        <div className="ml-[6.5rem] rounded-md border border-amber-500/40 bg-amber-500/10 px-2 py-1
                        text-[11px] leading-snug text-amber-200">
          The camera URL points at the ESP32&apos;s address ({camHost}). The phone has
          its own IP — the capture node shows it on its screen.
        </div>
      )}
    </>
  );
}
