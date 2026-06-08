/**
 * Scene Explorer — a collapsible tree of the live scene graph with connecting
 * guide lines. The tree is built from the server's node list (`parent` links,
 * rooted at the WORLD frame); each row toggles that node's visibility (a hidden
 * parent hides its whole subtree).
 */

import { Fragment, useMemo, useState, type ReactNode } from 'react';
import { useViewerStore, type SceneNodeInfo } from './modelStore';
import { cls } from './ui';

const LINE = '#3a4862';

/** Tree connecting lines for one row: pass-through verticals for ancestors
 *  that still have siblings below, plus the elbow into this node. */
function Gutter({ prefix, isLast }: { prefix: boolean[]; isLast: boolean }) {
  return (
    <>
      {prefix.map((continues, i) => (
        <span key={i} className="relative inline-block h-7 w-4 shrink-0">
          {continues && (
            <span
              className="absolute"
              style={{ left: 8, top: 0, bottom: 0, width: 1, background: LINE }}
            />
          )}
        </span>
      ))}
      <span className="relative inline-block h-7 w-4 shrink-0">
        {/* vertical from the parent line down to this row's centre */}
        <span
          className="absolute"
          style={{ left: 8, top: 0, height: 14, width: 1, background: LINE }}
        />
        {/* elbow into the node */}
        <span
          className="absolute"
          style={{ left: 8, top: 14, width: 8, height: 1, background: LINE }}
        />
        {/* continue the vertical for following siblings */}
        {!isLast && (
          <span
            className="absolute"
            style={{ left: 8, top: 14, bottom: 0, width: 1, background: LINE }}
          />
        )}
      </span>
    </>
  );
}

export function SceneExplorer() {
  const sceneNodes = useViewerStore((s) => s.sceneNodes);
  const hidden = useViewerStore((s) => s.hiddenNodes);
  const toggleNode = useViewerStore((s) => s.toggleNode);
  const [collapsed, setCollapsed] = useState<string[]>([]);

  /** parent id (or null for roots) → child nodes. */
  const childrenOf = useMemo(() => {
    const m = new Map<string | null, SceneNodeInfo[]>();
    for (const n of sceneNodes) {
      const arr = m.get(n.parent);
      if (arr) arr.push(n);
      else m.set(n.parent, [n]);
    }
    return m;
  }, [sceneNodes]);

  const toggleCollapse = (id: string) =>
    setCollapsed((c) => (c.includes(id) ? c.filter((x) => x !== id) : [...c, id]));

  function row(
    node: SceneNodeInfo,
    prefix: boolean[],
    isLast: boolean,
    isRoot: boolean,
  ): ReactNode {
    const kids = childrenOf.get(node.id) ?? [];
    const isCollapsed = collapsed.includes(node.id);
    const visible = !hidden.includes(node.id);
    const childPrefix = isRoot ? [] : [...prefix, !isLast];
    return (
      <Fragment key={node.id}>
        <div className="flex h-7 items-center text-[13px]">
          {!isRoot && <Gutter prefix={prefix} isLast={isLast} />}
          {kids.length > 0 ? (
            <button
              className="w-4 shrink-0 text-inkdim hover:text-ink"
              onClick={() => toggleCollapse(node.id)}
            >
              {isCollapsed ? '▸' : '▾'}
            </button>
          ) : (
            <span className="w-4 shrink-0" />
          )}
          <input
            type="checkbox"
            className={cls.check}
            checked={visible}
            onChange={() => toggleNode(node.id)}
          />
          <span
            className={
              'ml-1.5 ' + (visible ? 'text-ink' : 'text-inkmute line-through')
            }
          >
            {node.id}
          </span>
          <span className="ml-auto pl-2 text-[11px] text-inkdim">{node.type}</span>
        </div>
        {!isCollapsed &&
          kids.map((k, i) => row(k, childPrefix, i === kids.length - 1, false))}
      </Fragment>
    );
  }

  const roots = childrenOf.get(null) ?? [];

  return (
    <div className={`${cls.card} flex-1 min-h-0 overflow-y-auto`}>
      <div className="mb-2 flex items-center justify-between text-[12px] font-semibold uppercase tracking-[0.14em] text-ink">
        <span>Scene explorer</span>
        <span className="text-inkdim">{sceneNodes.length} nodes</span>
      </div>
      <div className="flex flex-col">
        {roots.length === 0 ? (
          <span className="text-[12px] text-inkmute">scene empty</span>
        ) : (
          roots.map((r, i) => row(r, [], i === roots.length - 1, true))
        )}
      </div>
    </div>
  );
}
