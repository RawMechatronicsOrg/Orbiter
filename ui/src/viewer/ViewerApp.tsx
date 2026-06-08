/**
 * The generic viewer: a Top / Left / Right bar frame around the
 * domain-agnostic scene renderer, all driven by one WebSocket to the
 * Python server. The top-level Scaner / Library switch is a Radix Tabs
 * root — TopBar renders its TabsList, content goes in TabsContent below.
 */

import { useEffect, useMemo, useState } from 'react';
import { WsClient } from './WsClient';
import { SceneRenderer } from './SceneRenderer';
import { TopBar, type ViewerTab } from './TopBar';
import { LeftBar } from './LeftBar';
import { RightBar } from './RightBar';
import { PositionOverlay } from './PositionOverlay';
import { CameraThumbnail } from './CameraThumbnail';
import { SceneExplorerPanel } from './SceneExplorerPanel';
import { LibraryView } from './LibraryView';
import { HelpView } from './HelpView';
import { LogPanel } from './LogPanel';
import { CaptureProgressModal } from './CaptureProgressModal';
import { makeCommands } from './commands';
import { bindClientToStore, useViewerStore } from './modelStore';
import { Tabs, TabsContent } from '../components/ui/tabs';
import { wsSceneUrl } from './api';

export function ViewerApp() {
  const client = useMemo(() => new WsClient(wsSceneUrl()), []);
  const commands = useMemo(() => makeCommands(client), [client]);
  const [tab, setTab] = useState<ViewerTab>('scaner');

  useEffect(() => {
    bindClientToStore(client.handlers);
    client.connect();
    return () => client.close();
  }, [client]);

  // Warn before leaving the page while the active scan has unsaved changes.
  useEffect(() => {
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      if (useViewerStore.getState().model.scan_dirty === true) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  }, []);

  return (
    <Tabs
      value={tab}
      onValueChange={(v) => setTab(v as ViewerTab)}
      className="fixed inset-0 flex flex-col bg-stage"
    >
      <TopBar commands={commands} />
      <TabsContent
        value="scaner"
        className="m-0 flex min-h-0 min-w-0 flex-1 data-[state=inactive]:hidden"
      >
        <LeftBar commands={commands} />
        <div className="relative min-w-0 flex-1">
          <SceneRenderer client={client} />
          <SceneExplorerPanel />
          <CameraThumbnail />
          <PositionOverlay commands={commands} />
        </div>
        <RightBar commands={commands} />
      </TabsContent>
      <TabsContent
        value="library"
        className="m-0 flex min-h-0 min-w-0 flex-1 data-[state=inactive]:hidden"
      >
        <LibraryView commands={commands} />
      </TabsContent>
      <TabsContent
        value="help"
        className="m-0 flex min-h-0 min-w-0 flex-1 data-[state=inactive]:hidden"
      >
        <HelpView />
      </TabsContent>
      <LogPanel />
      <CaptureProgressModal />
    </Tabs>
  );
}

/** Re-export so consumers can read connection state without importing the store. */
export { useViewerStore };
