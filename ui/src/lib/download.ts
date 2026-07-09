/** Shared file-download helpers (extracted from LibraryView / CaptureModal). */

/** Save a blob to disk under `filename` via a transient <a download>. */
export function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/** Fetch a file and save it (cross-origin friendly — controls the filename). */
export async function downloadFile(url: string, filename: string): Promise<void> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`download: HTTP ${r.status}`);
  triggerDownload(await r.blob(), filename);
}
