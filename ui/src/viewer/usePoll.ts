import { useEffect, useRef } from 'react';

/** Run `fn` now and then `ms` after each run has finished — never two at
 *  once, so a slow device does not pile requests up on the server's single
 *  link to the firmware. */
export function usePoll(fn: () => Promise<unknown>, ms: number, enabled = true): void {
  const fnRef = useRef(fn);
  fnRef.current = fn;
  useEffect(() => {
    if (!enabled) return undefined;
    let stopped = false;
    let timer: number | undefined;
    const tick = async () => {
      try {
        await fnRef.current();
      } finally {
        if (!stopped) timer = window.setTimeout(tick, ms);
      }
    };
    void tick();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [ms, enabled]);
}
