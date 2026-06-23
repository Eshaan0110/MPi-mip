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
    <div className="bg-white rounded-lg border border-gray-200 p-5 shadow-sm">
      <p className="text-sm text-gray-500 font-medium">{title}</p>
      <p className="text-2xl font-bold text-brand-700 mt-1">{value}</p>
      {subtitle && <p className="text-xs text-gray-400 mt-1">{subtitle}</p>}
      {trend && (
        <p className={`text-sm mt-2 font-medium ${trendUp ? "text-green-600" : "text-red-500"}`}>
          {trendUp ? "+" : ""}{trend}
        </p>
      )}
    </div>
  );
}
