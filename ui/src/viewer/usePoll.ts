import { useEffect, useRef } from 'react';

/** Run `fn` now and then `ms` after each run has finished — never two at
 *  once, not even when `ms` changes mid-request, so a slow device does not
 *  pile requests up on the server's single link to the firmware. `fn`
 *  reports its own errors; a throw only ends that round. */
export function usePoll(fn: () => Promise<unknown>, ms: number, enabled = true): void {
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const inFlight = useRef<Promise<unknown> | null>(null);
  useEffect(() => {
    if (!enabled) return undefined;
    let stopped = false;
    let timer: number | undefined;
    const tick = async () => {
      if (inFlight.current) await inFlight.current.catch(() => undefined);
      if (stopped) return;
      const run = fnRef.current().catch(() => undefined);
      inFlight.current = run;
      await run;
      inFlight.current = null;
      if (!stopped) timer = window.setTimeout(tick, ms);
    };
    void tick();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [ms, enabled]);
}
