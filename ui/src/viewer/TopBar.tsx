/** Top status strip — branding + tab switcher + live connection / pose readout. */

import { useViewerStore } from './modelStore';
import type { ConnState } from './WsClient';
import type { Commands } from './commands';
import { cls, num } from './ui';
import { TabsList, TabsTrigger } from '../components/ui/tabs';
import { Button } from '../components/ui/button';

/** The top-level views of the app. */
export type ViewerTab = 'scaner' | 'library' | 'help';

const TABS: ReadonlyArray<readonly [ViewerTab, string]> = [
  ['scaner', 'Scaner'],
  ['library', 'Library'],
  ['help', 'Help'],
];

function Dot({ online }: { online: boolean }) {
  return (
    <span
      className={
        'w-2 h-2 rounded-full inline-block mr-1.5 ' +
        (online ? 'bg-emerald-400 animate-pulse' : 'bg-red-400')
      }
    />
  );
}

/** Per-state look of the server connection indicator. */
const CONN_UI: Record<
  ConnState,
  { dot: string; pulse: boolean; text: string; label: string }
> = {
  online: { dot: 'bg-emerald-400', pulse: true, text: 'text-inkdim', label: 'server' },
  connecting: {
    dot: 'bg-amber-400',
    pulse: true,
    text: 'text-amber-300',
    label: 'reconnecting…',
  },
  offline: { dot: 'bg-red-400', pulse: false, text: 'text-red-300', label: 'offline' },
};

/** WebSocket connection indicator — colour + label track the reconnect loop. */
function ServerStatus() {
  const connState = useViewerStore((s) => s.connState);
  const ui = CONN_UI[connState];
  return (
    <span className={ui.text}>
      <span
        className={
          'w-2 h-2 rounded-full inline-block mr-1.5 ' +
          ui.dot +
          (ui.pulse ? ' animate-pulse' : '')
        }
      />
      {ui.label}
    </span>
  );
}

export function TopBar({ commands }: { commands: Commands }) {
  const model = useViewerStore((s) => s.model);
  const espOnline = model.esp_online === true;

  const rebootEsp = () => {
    if (
      window.confirm(
        'Reboot ESP firmware now? The device will be offline for a few seconds.',
      )
    ) {
      commands.rebootEsp();
    }
  };

  return (
    <div className={cls.topbar}>
      <span className="font-bold tracking-[0.2em] text-sky-200">ORBITER</span>
      <TabsList>
        {TABS.map(([key, label]) => (
          <TabsTrigger
            key={key}
            value={key}
            className="uppercase tracking-[0.1em]"
          >
            {label}
          </TabsTrigger>
        ))}
      </TabsList>
      <ServerStatus />
      <span>
        <Dot online={espOnline} />
        ESP
      </span>
      <Button
        variant="outline"
        size="xs"
        disabled={!espOnline}
        onClick={rebootEsp}
        title="Restart ESP32 firmware"
        className="hover:border-red-500 hover:text-red-300"
      >
        Reboot FW
      </Button>
      <span className="text-inkmute">{String(model.motion_state ?? '—')}</span>
      {model.scan_running === true && (
        <span className="text-violet-400">
          scan {num(model, 'scan_progress')}/{num(model, 'scan_total')}
        </span>
      )}
    </div>
  );
}
