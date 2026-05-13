type Props = {
  label: string;
  icon: string;
  active: boolean;
  value?: string;
};

export function DetectionBadge({ label, icon, active, value }: Props) {
  return (
    <div
      className={`rounded-lg border px-3 py-2 transition ${
        active
          ? "border-cyan-300/50 bg-cyan-300/12 text-white shadow-glow"
          : "border-white/10 bg-white/[0.04] text-slate-400"
      }`}
    >
      <div className="flex items-center gap-2">
        <span className="text-lg">{icon}</span>
        <span className="text-sm font-medium">{label}</span>
      </div>
      {value ? <div className="mt-1 text-xs text-slate-300">{value}</div> : null}
    </div>
  );
}
