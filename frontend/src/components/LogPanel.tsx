import type { LogItem } from "../lib/api";

export function LogPanel({ logs }: { logs: LogItem[] }) {
  return (
    <div className="glass rounded-lg p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">Detection Logs</h2>
        <span className="rounded-full bg-white/10 px-2 py-1 text-xs text-slate-400">{logs.length}</span>
      </div>
      <div className="max-h-72 space-y-2 overflow-y-auto pr-1">
        {logs.length === 0 ? (
          <p className="text-sm text-slate-500">No detections yet.</p>
        ) : (
          logs
            .slice()
            .reverse()
            .map((log, index) => (
              <div key={`${log.timestamp}-${index}`} className="rounded-lg border border-white/10 bg-black/20 p-3">
                <div className="flex items-center justify-between gap-3">
                  <span className="text-sm font-medium text-slate-100">{log.summary}</span>
                  <span className="text-xs text-slate-500">{log.timestamp}</span>
                </div>
                <div className="mt-2 flex items-center gap-2 text-xs text-slate-400">
                  <span>{log.source}</span>
                  <span className="h-1 w-1 rounded-full bg-slate-600" />
                  <span>{log.risk_level}</span>
                  <span className="h-1 w-1 rounded-full bg-slate-600" />
                  <span>{Math.round((log.confidence ?? 0) * 100)}%</span>
                </div>
              </div>
            ))
        )}
      </div>
    </div>
  );
}
