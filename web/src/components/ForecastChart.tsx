"use client";

import { useState, useCallback, useRef } from "react";
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
  Brush,
  ReferenceArea,
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
  "#60a5fa", "#f87171", "#34d399", "#fbbf24", "#a78bfa",
  "#22d3ee", "#f472b6", "#a3e635", "#fb923c", "#818cf8",
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

function formatMonthLong(m: string): string {
  const d = new Date(m.length === 7 ? m + "-01" : m);
  if (isNaN(d.getTime())) return m;
  return d.toLocaleDateString("en-IN", { month: "long", year: "numeric" });
}

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-slate-800 border border-slate-600 rounded-lg shadow-xl px-4 py-3 text-sm min-w-[180px]">
      <p className="font-semibold text-slate-200 mb-2 text-[13px]">{formatMonthLong(label)}</p>
      {payload.map((p: any, i: number) => {
        if (p.dataKey === "ciRange") {
          const [low, high] = p.value || [];
          return (
            <p key={i} className="text-slate-400 text-xs mt-1 pt-1 border-t border-slate-700">
              90% CI: {formatM(low)} – {formatM(high)}
            </p>
          );
        }
        return (
          <p key={i} style={{ color: p.color || p.stroke }} className="flex justify-between gap-4 py-0.5">
            <span>{p.name}:</span>
            <span className="font-semibold">{formatM(p.value)}</span>
          </p>
        );
      })}
    </div>
  );
}

function ZoomControls({ isZoomed, onReset }: { isZoomed: boolean; onReset: () => void }) {
  return (
    <div className="flex items-center gap-3 text-xs text-slate-500">
      <span>Drag on chart to zoom</span>
      {isZoomed && (
        <button
          onClick={onReset}
          className="px-2.5 py-1 rounded-md bg-slate-700 text-slate-300 hover:bg-slate-600 transition-colors font-medium"
        >
          Reset zoom
        </button>
      )}
    </div>
  );
}

export function ForecastChart({ data, title, unit, multiLines, multiData }: ForecastChartProps) {
  const [refAreaLeft, setRefAreaLeft] = useState<string>("");
  const [refAreaRight, setRefAreaRight] = useState<string>("");
  const [zoomLeft, setZoomLeft] = useState<string | null>(null);
  const [zoomRight, setZoomRight] = useState<string | null>(null);
  const isDragging = useRef(false);

  const resetZoom = useCallback(() => {
    setZoomLeft(null);
    setZoomRight(null);
    setRefAreaLeft("");
    setRefAreaRight("");
  }, []);

  const handleMouseDown = useCallback((e: any) => {
    if (e?.activeLabel) {
      setRefAreaLeft(e.activeLabel);
      isDragging.current = true;
    }
  }, []);

  const handleMouseMove = useCallback((e: any) => {
    if (isDragging.current && e?.activeLabel) {
      setRefAreaRight(e.activeLabel);
    }
  }, []);

  const handleMouseUp = useCallback((allMonths: string[]) => {
    if (!refAreaLeft || !refAreaRight || refAreaLeft === refAreaRight) {
      setRefAreaLeft("");
      setRefAreaRight("");
      isDragging.current = false;
      return;
    }
    const idxL = allMonths.indexOf(refAreaLeft);
    const idxR = allMonths.indexOf(refAreaRight);
    const [left, right] = idxL <= idxR ? [refAreaLeft, refAreaRight] : [refAreaRight, refAreaLeft];
    setZoomLeft(left);
    setZoomRight(right);
    setRefAreaLeft("");
    setRefAreaRight("");
    isDragging.current = false;
  }, [refAreaLeft, refAreaRight]);

  if (multiLines && multiData) {
    const chartData = multiData.map((d) => {
      const out: any = { month: d.month };
      for (const line of multiLines) {
        const raw = d[line.key];
        out[line.key] = typeof raw === "number" ? toMillions(raw) : raw;
      }
      return out;
    });

    const allMonths = chartData.map((d: any) => d.month);
    const filteredData = zoomLeft && zoomRight
      ? chartData.filter((d: any) => d.month >= zoomLeft && d.month <= zoomRight)
      : chartData;

    return (
      <div className="bg-slate-800/50 rounded-xl border border-slate-700/50 p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
            <p className="text-xs text-slate-500 mt-0.5">Values in Millions</p>
          </div>
          <ZoomControls isZoomed={!!(zoomLeft && zoomRight)} onReset={resetZoom} />
        </div>
        <ResponsiveContainer width="100%" height={400}>
          <ComposedChart
            data={filteredData}
            margin={{ top: 5, right: 20, bottom: 5, left: 10 }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={() => handleMouseUp(allMonths)}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            <XAxis dataKey="month" tick={{ fontSize: 12, fill: "#94a3b8" }} tickFormatter={formatMonth} axisLine={{ stroke: "#475569" }} minTickGap={30} />
            <YAxis tick={{ fontSize: 12, fill: "#94a3b8" }} tickFormatter={formatAxisTick} axisLine={false} tickLine={false} width={65} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12, paddingTop: 12, color: "#94a3b8" }} />
            {multiLines.map((line) => (
              <Line
                key={line.key}
                type="monotone"
                dataKey={line.key}
                stroke={line.color}
                strokeWidth={2.5}
                dot={{ r: 3, strokeWidth: 0, fill: line.color }}
                activeDot={{ r: 5, strokeWidth: 2, stroke: "#1e293b" }}
                name={line.label}
              />
            ))}
            {refAreaLeft && refAreaRight && (
              <ReferenceArea x1={refAreaLeft} x2={refAreaRight} strokeOpacity={0.3} fill="#60a5fa" fillOpacity={0.15} />
            )}
            {!zoomLeft && (
              <Brush dataKey="month" height={28} stroke="#475569" fill="#1e293b" tickFormatter={formatMonth} travellerWidth={10} />
            )}
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

  const allMonths = chartData.map((d) => d.month);
  const filteredData = zoomLeft && zoomRight
    ? chartData.filter((d) => d.month >= zoomLeft && d.month <= zoomRight)
    : chartData;

  return (
    <div className="bg-slate-800/50 rounded-xl border border-slate-700/50 p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
          <p className="text-xs text-slate-500 mt-0.5">Values in Millions{unit ? ` (${unit})` : ""}</p>
        </div>
        <ZoomControls isZoomed={!!(zoomLeft && zoomRight)} onReset={resetZoom} />
      </div>
      <ResponsiveContainer width="100%" height={340}>
        <ComposedChart
          data={filteredData}
          margin={{ top: 5, right: 20, bottom: 5, left: 10 }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={() => handleMouseUp(allMonths)}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
          <XAxis dataKey="month" tick={{ fontSize: 12, fill: "#94a3b8" }} tickFormatter={formatMonth} axisLine={{ stroke: "#475569" }} minTickGap={30} />
          <YAxis tick={{ fontSize: 12, fill: "#94a3b8" }} tickFormatter={formatAxisTick} axisLine={false} tickLine={false} width={65} />
          <Tooltip content={<CustomTooltip />} />
          {hasCi && (
            <Area dataKey="ciRange" stroke="none" fill="#60a5fa" fillOpacity={0.12} name="90% CI" type="monotone" />
          )}
          {chartData.some((d) => d.actual !== undefined) && (
            <Line type="monotone" dataKey="actual" stroke="#e2e8f0" strokeWidth={2} dot={{ r: 3, strokeWidth: 0, fill: "#e2e8f0" }} activeDot={{ r: 5, strokeWidth: 2, stroke: "#1e293b" }} name="Actual" />
          )}
          {chartData.some((d) => d.forecast !== undefined) && (
            <Line type="monotone" dataKey="forecast" stroke="#60a5fa" strokeWidth={2.5} dot={{ r: 3, strokeWidth: 0, fill: "#60a5fa" }} activeDot={{ r: 6, strokeWidth: 2, stroke: "#1e293b" }} name="Forecast" />
          )}
          {refAreaLeft && refAreaRight && (
            <ReferenceArea x1={refAreaLeft} x2={refAreaRight} strokeOpacity={0.3} fill="#60a5fa" fillOpacity={0.15} />
          )}
          {!zoomLeft && (
            <Brush dataKey="month" height={28} stroke="#475569" fill="#1e293b" tickFormatter={formatMonth} travellerWidth={10} />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export { COLORS };
