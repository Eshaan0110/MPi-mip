"use client";

import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  ComposedChart,
  Legend,
} from "recharts";

interface DataPoint {
  month: string;
  actual?: number;
  forecast?: number;
  lower?: number;
  upper?: number;
}

interface MultiLinePoint {
  month: string;
  [key: string]: number | string | undefined;
}

interface ForecastChartProps {
  data: DataPoint[];
  title: string;
  unit?: string;
  multiLines?: { key: string; label: string; color: string }[];
  multiData?: MultiLinePoint[];
}

const COLORS = [
  "#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed",
  "#0891b2", "#be185d", "#65a30d", "#ea580c", "#6366f1",
];

function toMillions(v: number): number {
  return v / 10;
}

function formatM(v: number, decimals = 1): string {
  return v.toFixed(decimals) + " M";
}

function formatAxisTick(v: number): string {
  return formatM(v, 0);
}

function formatMonth(m: string): string {
  const d = new Date(m);
  if (isNaN(d.getTime())) return m;
  return d.toLocaleDateString("en-IN", { month: "short", year: "2-digit" });
}

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-lg px-4 py-3 text-sm">
      <p className="font-semibold text-gray-700 mb-1.5">{formatMonth(label)}</p>
      {payload.map((p: any, i: number) => {
        if (p.dataKey === "ciRange") {
          const [low, high] = p.value || [];
          return (
            <p key={i} className="text-gray-500 text-xs">
              90% CI: {formatM(low)} – {formatM(high)}
            </p>
          );
        }
        return (
          <p key={i} style={{ color: p.color || p.stroke }} className="flex justify-between gap-4">
            <span>{p.name}:</span>
            <span className="font-medium">{formatM(p.value)}</span>
          </p>
        );
      })}
    </div>
  );
}

export function ForecastChart({ data, title, unit, multiLines, multiData }: ForecastChartProps) {
  if (multiLines && multiData) {
    const chartData = multiData.map((d) => {
      const out: any = { month: d.month };
      for (const line of multiLines) {
        const raw = d[line.key];
        out[line.key] = typeof raw === "number" ? toMillions(raw) : raw;
      }
      return out;
    });

    return (
      <div className="bg-white rounded-xl border border-gray-100 p-6 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-1">{title}</h3>
        <p className="text-xs text-gray-400 mb-4">Values in Millions</p>
        <ResponsiveContainer width="100%" height={360}>
          <ComposedChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" vertical={false} />
            <XAxis dataKey="month" tick={{ fontSize: 11, fill: "#9ca3af" }} tickFormatter={formatMonth} axisLine={{ stroke: "#e5e7eb" }} />
            <YAxis tick={{ fontSize: 11, fill: "#9ca3af" }} tickFormatter={formatAxisTick} axisLine={false} tickLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
            {multiLines.map((line) => (
              <Line
                key={line.key}
                type="monotone"
                dataKey={line.key}
                stroke={line.color}
                strokeWidth={2}
                dot={{ r: 2, strokeWidth: 0, fill: line.color }}
                name={line.label}
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    );
  }

  const hasCi = data.some((d) => d.lower !== undefined && d.upper !== undefined);
  const chartData = data.map((d) => ({
    month: d.month,
    actual: d.actual !== undefined ? toMillions(d.actual) : undefined,
    forecast: d.forecast !== undefined ? toMillions(d.forecast) : undefined,
    ciRange:
      d.lower !== undefined && d.upper !== undefined
        ? [toMillions(d.lower), toMillions(d.upper)]
        : undefined,
  }));

  return (
    <div className="bg-white rounded-xl border border-gray-100 p-6 shadow-sm">
      <h3 className="text-sm font-semibold text-gray-700 mb-1">{title}</h3>
      <p className="text-xs text-gray-400 mb-4">Values in Millions{unit ? ` (${unit})` : ""}</p>
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" vertical={false} />
          <XAxis dataKey="month" tick={{ fontSize: 11, fill: "#9ca3af" }} tickFormatter={formatMonth} axisLine={{ stroke: "#e5e7eb" }} />
          <YAxis tick={{ fontSize: 11, fill: "#9ca3af" }} tickFormatter={formatAxisTick} axisLine={false} tickLine={false} />
          <Tooltip content={<CustomTooltip />} />
          {hasCi && (
            <Area
              dataKey="ciRange"
              stroke="none"
              fill="#2563eb"
              fillOpacity={0.1}
              name="90% CI"
              type="monotone"
            />
          )}
          {chartData.some((d) => d.actual !== undefined) && (
            <Line type="monotone" dataKey="actual" stroke="#111827" strokeWidth={2} dot={{ r: 2.5, strokeWidth: 0, fill: "#111827" }} name="Actual" />
          )}
          {chartData.some((d) => d.forecast !== undefined) && (
            <Line type="monotone" dataKey="forecast" stroke="#2563eb" strokeWidth={2.5} dot={{ r: 3, strokeWidth: 0, fill: "#2563eb" }} name="Forecast" />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export { COLORS };
