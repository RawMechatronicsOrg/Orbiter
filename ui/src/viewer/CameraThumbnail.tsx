/**
 * Live camera monitor — a small, always-on MJPEG thumbnail pinned to the
 * top-right of the 3D viewer for at-a-glance control of what the phone sees.
 * Click it to open a full-size modal of the same stream.
 *
 * The server re-multiplexes the phone's MJPEG at `GET /camera/stream.mjpeg`
 * (a multipart/x-mixed-replace response that is a drop-in for `<img src>` —
 * no JS decoder needed). `GET /camera/stream/status` drives the live dot.
 * Each `<img>` opens its own stream connection, but the server shares ONE
 * upstream phone connection across all viewers, so thumbnail + modal is fine.
 */

import { useEffect, useState } from 'react';
import { useViewerStore } from './modelStore';
import { cameraStreamUrl, getJson } from './api';
import { Dialog, DialogClose, DialogContent, DialogTitle } from '../components/ui/dialog';

const STREAM_URL = cameraStreamUrl();
const STATUS_PATH = '/camera/stream/status';
const STATUS_POLL_MS = 2000;

interface StreamStatus {
  connected: boolean;
  have_frame: boolean;
  seq: number;
  url: string;
}

function StatusDot({ live, label }: { live: boolean; label: string }) {
  return (
    <span
      className="flex items-center gap-1.5 text-[11px] font-semibold uppercase
                 tracking-[0.16em] text-inkmute"
    >
      <span
        className={
          'inline-block h-2 w-2 rounded-full ' +
          (live
            ? 'bg-emerald-400 shadow-[0_0_6px] shadow-emerald-400/70'
            : 'bg-rose-500/70')
        }
      />
      {label}
    </span>
  );
}

export function CameraThumbnail() {
  const cameraUrl = useViewerStore((s) =>
    typeof s.model.camera_url === 'string' ? (s.model.camera_url as string) : '',
  );
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<StreamStatus | null>(null);

  // Poll the upstream connectivity snapshot for the live/offline dot.
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const s = await getJson<StreamStatus>(STATUS_PATH);
        if (alive) setStatus(s);
      } catch {
        if (alive) setStatus(null);
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), STATUS_POLL_MS);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);

  // No camera configured — show a hint instead of a dead stream connection.
  if (!cameraUrl) {
    return (
      <div
        className="absolute right-3 top-3 z-20 w-52 rounded-lg border border-cardline
                   bg-[#131c2e]/90 px-3 py-2 text-[11px] text-inkmute backdrop-blur-sm"
      >
        <StatusDot live={false} label="No camera" />
        <div className="mt-1 leading-snug">
          Set the camera URL in <span className="text-ink">Machine&nbsp;config</span> to
          see the live stream.
        </div>
      </div>
    );
  }

  const live = status?.connected === true && status?.have_frame === true;

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        title="Open full-size camera stream"
        className="group absolute right-3 top-3 z-20 w-52 overflow-hidden rounded-lg
                   border border-cardline bg-[#131c2e]/90 text-left shadow-lg
                   backdrop-blur-sm transition-colors hover:border-sky-400/60
                   focus:outline-none focus:ring-2 focus:ring-sky-400/50"
      >
        <div className="flex items-center justify-between px-2.5 py-1.5">
          <StatusDot live={live} label="Camera" />
          <span
            className="text-[10px] uppercase tracking-wider text-inkmute opacity-0
                       transition-opacity group-hover:opacity-100"
          >
            expand ⤢
          </span>
        </div>
        <div className="relative aspect-video w-full bg-black">
          <img
            src={STREAM_URL}
            alt="camera stream"
            className="h-full w-full object-contain"
            draggable={false}
          />
          {!live && (
            <div className="absolute inset-0 flex items-center justify-center
                            text-[11px] text-inkmute">
              waiting for camera…
            </div>
          )}
        </div>
      </button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent
          aria-describedby={undefined}
          className="w-[92vw] max-w-[1280px] p-0"
        >
          <div className="flex items-center justify-between border-b border-cardline px-4 py-2.5">
            <DialogTitle>Camera stream</DialogTitle>
            <div className="flex items-center gap-4">
              <StatusDot live={live} label={live ? 'live' : 'offline'} />
              <DialogClose
                aria-label="Close"
                className="rounded px-2 py-0.5 text-[15px] leading-none text-inkmute
                           transition-colors hover:text-ink focus:outline-none
                           focus:ring-2 focus:ring-sky-400/50"
              >
                ✕
              </DialogClose>
            </div>
          </div>
          <div className="flex max-h-[80vh] items-center justify-center bg-black">
            {/* Mount the large stream only while open so the second upstream
                connection is opened on demand and closed when the modal does. */}
            {open && (
              <img
                src={STREAM_URL}
                alt="camera stream"
                className="max-h-[80vh] w-auto max-w-full object-contain"
                draggable={false}
              />
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
