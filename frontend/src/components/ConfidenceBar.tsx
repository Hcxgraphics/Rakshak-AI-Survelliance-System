type Props = {
  label: string;
  value: number;
  tone?: "green" | "amber" | "red" | "blue";
};

const tones = {
  green: "from-emerald-400 to-lime-300",
  amber: "from-amber-300 to-orange-400",
  red: "from-rose-400 to-red-500",
  blue: "from-sky-300 to-cyan-400",
};

export function ConfidenceBar({ label, value, tone = "blue" }: Props) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm text-slate-300">
        <span>{label}</span>
        <span className="font-medium text-slate-100">{pct.toFixed(0)}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-white/10">
        <div className={`h-full rounded-full bg-gradient-to-r ${tones[tone]}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
