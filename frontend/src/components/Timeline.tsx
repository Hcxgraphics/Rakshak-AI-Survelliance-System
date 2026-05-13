import type { DetectionResponse } from "../lib/api";

export function Timeline({ items }: { items: DetectionResponse[] }) {
  return (
    <div className="glass rounded-lg p-4">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-300">Timeline</h2>
      <div className="space-y-3">
        {items.length === 0 ? (
          <p className="text-sm text-slate-500">Process media or start the camera to build a timeline.</p>
        ) : (
          items.slice(-7).map((item, index) => (
            <div key={item.request_id} className="flex gap-3">
              <div className="flex flex-col items-center">
                <div className="h-3 w-3 rounded-full bg-cyan-300 shadow-glow" />
                {index < items.slice(-7).length - 1 ? <div className="mt-1 h-full w-px bg-white/10" /> : null}
              </div>
              <div className="pb-2">
                <div className="text-sm font-medium text-slate-100">{item.summary}</div>
                <div className="text-xs text-slate-500">
                  {item.risk_level} · {Math.round(item.fusion_confidence * 100)}%
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
