/**
 * Capture preview modal — opened by clicking a capture frustum or its photo
 * card in the 3D scene. Shows the ORIGINAL photo plus pose metadata and a
 * download button.
 *
 * Radix Dialogs are plain HTML and must live OUTSIDE the R3F <Canvas>; this is
 * rendered as a sibling of the canvas in `SceneRenderer`. The pattern mirrors
 * `LibraryView`'s `PhotoModal` but is intentionally self-contained so the two
 * stay decoupled.
 */

import { Button } from '../components/ui/button';
import { Dialog, DialogContent, DialogTitle } from '../components/ui/dialog';
import { API_BASE } from './api';
import type { Commands } from './commands';

/** Subset of a `model.captures` entry the modal renders. Mirrors the server
 *  `Capture` shape (pose metadata + image URLs). All optional but `index` so
 *  this tolerates partial records. */
export interface CaptureView {
  capture_id?: string;
  full_url?: string;
  thumb_url?: string;
  index?: number;
  az_deg?: number;
  el_deg?: number;
  timestamp?: string;
  camera_preset?: string;
  stored_width?: number;
  stored_height?: number;
}

/** Server timestamps are UTC ISO-8601 — render uniformly in local time. */
function formatTime(iso: unknown): string {
  if (typeof iso !== 'string' || !iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const p = (n: number) => String(n).padStart(2, '0');
  return (
    `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ` +
    `${p(d.getHours())}:${p(d.getMinutes())}`
  );
}

/** Fetch a file and save it under `filename` via a transient <a download>. */
async function downloadFile(url: string, filename: string): Promise<void> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`download: HTTP ${r.status}`);
  const blob = await r.blob();
  const objUrl = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = objUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(objUrl);
}

/** Full-size photo preview for a scene-picked capture.
 *
 *  When `commands` is supplied, a destructive "Delete photo" button is shown
 *  that deletes this capture from the active scan (model + pool files +
 *  manifest, server-side) and closes the modal. The prop is optional so the
 *  modal still renders read-only previews without a delete affordance. */
export function CaptureModal({
  capture,
  open,
  onClose,
  commands,
}: {
  capture: CaptureView | null;
  open: boolean;
  onClose: () => void;
  commands?: Commands;
}) {
  if (!capture) return null;
  const captureId = capture.capture_id;
  // Delete is only offered when we have BOTH a way to send the command and a
  // capture_id to target (a partial record without an id can't be deleted).
  const canDelete = commands != null && typeof captureId === 'string';

  function onDelete(): void {
    if (!commands || typeof captureId !== 'string') return;
    if (
      !window.confirm(
        `Delete this photo (capture ${captureId})? ` +
          'It is removed from the scan and deleted from disk. This cannot be undone.',
      )
    ) {
      return;
    }
    commands.deleteCapture(captureId);
    // Close immediately — the scene + capture list update when the server
    // broadcasts the shrunk model.captures.
    onClose();
  }

  const fullUrl = API_BASE + (capture.full_url ?? capture.thumb_url ?? '');
  const fileName = `${capture.capture_id ?? `capture_${capture.index ?? 0}`}.jpg`;
  const meta: ReadonlyArray<readonly [string, string]> = [
    ['index', String(capture.index ?? '—')],
    [
      'az / el',
      `${(capture.az_deg ?? 0).toFixed(1)}° / ${(capture.el_deg ?? 0).toFixed(1)}°`,
    ],
    ['timestamp', formatTime(capture.timestamp)],
    ['camera', String(capture.camera_preset ?? '—')],
    [
      'stored size',
      capture.stored_width && capture.stored_height
        ? `${capture.stored_width} × ${capture.stored_height}`
        : '—',
    ],
    ['capture id', String(capture.capture_id ?? '—')],
  ];

  return (
    <Dialog open={open} onOpenChange={(next) => !next && onClose()}>
      <DialogContent className="flex max-h-[88vh] max-w-none p-0">
        <div className="flex items-center justify-center bg-app p-3">
          <img
            src={fullUrl}
            alt={capture.capture_id ?? 'capture'}
            className="max-h-[84vh] max-w-[66vw] object-contain"
          />
        </div>
        <div className="flex w-[300px] shrink-0 flex-col gap-3 overflow-y-auto p-4">
          <DialogTitle>Capture</DialogTitle>
          <div className="flex flex-col gap-1.5">
            {meta.map(([k, v]) => (
              <div key={k} className="flex gap-2 text-[13px]">
                <span className="w-28 shrink-0 text-inkmute">{k}</span>
                <span className="break-all font-mono text-inkdim">{v}</span>
              </div>
            ))}
          </div>
          <div className="mt-auto flex items-center gap-2 pt-2">
            {canDelete && (
              // Destructive — pushed to the left, apart from the safe actions.
              <Button
                variant="destructive"
                size="sm"
                className="mr-auto"
                onClick={onDelete}
                title="Remove this photo from the scan and delete it from disk"
              >
                Delete photo
              </Button>
            )}
            <Button
              onClick={() =>
                downloadFile(fullUrl, fileName).catch((e) =>
                  window.alert(String(e)),
                )
              }
            >
              Download original
            </Button>
            <Button variant="secondary" size="sm" onClick={onClose}>
              Close
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
