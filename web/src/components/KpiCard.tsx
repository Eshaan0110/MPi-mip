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
    <div className="bg-white border border-gray-200 dark:bg-slate-800/50 rounded-lg dark:border-slate-700/50 p-5">
      <p className="text-sm text-gray-500 dark:text-slate-400 font-medium">{title}</p>
      <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">{value}</p>
      {subtitle && <p className="text-xs text-gray-400 dark:text-slate-500 mt-1">{subtitle}</p>}
      {trend && (
        <p className={`text-sm mt-2 font-medium ${trendUp ? "text-emerald-500 dark:text-emerald-400" : "text-red-500 dark:text-red-400"}`}>
          {trendUp ? "+" : ""}{trend}
        </p>
      )}
    </div>
  );
}
