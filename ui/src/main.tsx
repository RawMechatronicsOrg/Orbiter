import React from 'react';
import ReactDOM from 'react-dom/client';
import { ViewerApp } from './viewer/ViewerApp';
import './styles.css';

// The generic viewer is the whole app — it renders the server-pushed scene
// graph and dispatches commands over /ws/scene. The legacy scanner/library
// app was removed at the Phase-8 cut-over.
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ViewerApp />
  </React.StrictMode>
);
