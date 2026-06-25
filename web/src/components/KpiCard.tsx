"use client";

interface KpiCardProps {
  title: string;
  value: string;
  subtitle?: string;
  trend?: string;
  trendUp?: boolean;
}

export function KpiCard({ title, value, subtitle, trend, trendUp }: KpiCardProps) {
  return (
    <div className="bg-slate-800/50 rounded-lg border border-slate-700/50 p-5">
      <p className="text-sm text-slate-400 font-medium">{title}</p>
      <p className="text-2xl font-bold text-white mt-1">{value}</p>
      {subtitle && <p className="text-xs text-slate-500 mt-1">{subtitle}</p>}
      {trend && (
        <p className={`text-sm mt-2 font-medium ${trendUp ? "text-emerald-400" : "text-red-400"}`}>
          {trendUp ? "+" : ""}{trend}
        </p>
      )}
    </div>
  );
}
