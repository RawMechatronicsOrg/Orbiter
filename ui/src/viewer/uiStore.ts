/**
 * Tiny zustand store for top-level UI navigation state. Lifted out of
 * ViewerApp so non-children (e.g. the Library's "Open in 3D" button) can
 * switch tabs without prop drilling.
 */

import { create } from 'zustand';

/** The top-level views of the app. */
export type ViewerTab = 'scaner' | 'library' | 'help';

interface UiState {
  activeTab: ViewerTab;
  setActiveTab: (tab: ViewerTab) => void;
}

export const useUiStore = create<UiState>((set) => ({
  activeTab: 'scaner',
  setActiveTab: (activeTab) => set({ activeTab }),
}));
