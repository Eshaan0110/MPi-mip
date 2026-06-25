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

function formatMonthLong(m: string): string {
  const d = new Date(m.length === 7 ? m + "-01" : m);
  if (isNaN(d.getTime())) return m;
  return d.toLocaleDateString("en-IN", { month: "long", year: "numeric" });
}

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-lg px-4 py-3 text-sm min-w-[180px]">
      <p className="font-semibold text-gray-700 mb-2 text-[13px]">{formatMonthLong(label)}</p>
      {payload.map((p: any, i: number) => {
        if (p.dataKey === "ciRange") {
          const [low, high] = p.value || [];
          return (
            <p key={i} className="text-gray-500 text-xs mt-1 pt-1 border-t border-gray-100">
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
    <div className="flex items-center gap-3 text-xs text-gray-400">
      <span>Drag on chart to zoom</span>
      {isZoomed && (
        <button
          onClick={onReset}
          className="px-2.5 py-1 rounded-md bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors font-medium"
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
      <div className="bg-white rounded-xl border border-gray-100 p-6 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-semibold text-gray-700">{title}</h3>
            <p className="text-xs text-gray-400 mt-0.5">Values in Millions</p>
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
            <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" vertical={false} />
            <XAxis
              dataKey="month"
              tick={{ fontSize: 12, fill: "#6b7280" }}
              tickFormatter={formatMonth}
              axisLine={{ stroke: "#e5e7eb" }}
              minTickGap={30}
            />
            <YAxis
              tick={{ fontSize: 12, fill: "#6b7280" }}
              tickFormatter={formatAxisTick}
              axisLine={false}
              tickLine={false}
              width={65}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12, paddingTop: 12 }} />
            {multiLines.map((line) => (
              <Line
                key={line.key}
                type="monotone"
                dataKey={line.key}
                stroke={line.color}
                strokeWidth={2.5}
                dot={{ r: 3, strokeWidth: 0, fill: line.color }}
                activeDot={{ r: 5, strokeWidth: 2, stroke: "#fff" }}
                name={line.label}
              />
            ))}
            {refAreaLeft && refAreaRight && (
              <ReferenceArea x1={refAreaLeft} x2={refAreaRight} strokeOpacity={0.3} fill="#2563eb" fillOpacity={0.1} />
            )}
            {!zoomLeft && (
              <Brush
                dataKey="month"
                height={28}
                stroke="#d1d5db"
                fill="#f9fafb"
                tickFormatter={formatMonth}
                travellerWidth={10}
              />
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
    <div className="bg-white rounded-xl border border-gray-100 p-6 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-gray-700">{title}</h3>
          <p className="text-xs text-gray-400 mt-0.5">Values in Millions{unit ? ` (${unit})` : ""}</p>
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
          <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" vertical={false} />
          <XAxis
            dataKey="month"
            tick={{ fontSize: 12, fill: "#6b7280" }}
            tickFormatter={formatMonth}
            axisLine={{ stroke: "#e5e7eb" }}
            minTickGap={30}
          />
          <YAxis
            tick={{ fontSize: 12, fill: "#6b7280" }}
            tickFormatter={formatAxisTick}
            axisLine={false}
            tickLine={false}
            width={65}
          />
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
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#111827"
              strokeWidth={2}
              dot={{ r: 3, strokeWidth: 0, fill: "#111827" }}
              activeDot={{ r: 5, strokeWidth: 2, stroke: "#fff" }}
              name="Actual"
            />
          )}
          {chartData.some((d) => d.forecast !== undefined) && (
            <Line
              type="monotone"
              dataKey="forecast"
              stroke="#2563eb"
              strokeWidth={2.5}
              dot={{ r: 3, strokeWidth: 0, fill: "#2563eb" }}
              activeDot={{ r: 6, strokeWidth: 2, stroke: "#fff" }}
              name="Forecast"
            />
          )}
          {refAreaLeft && refAreaRight && (
            <ReferenceArea x1={refAreaLeft} x2={refAreaRight} strokeOpacity={0.3} fill="#2563eb" fillOpacity={0.1} />
          )}
          {!zoomLeft && (
            <Brush
              dataKey="month"
              height={28}
              stroke="#d1d5db"
              fill="#f9fafb"
              tickFormatter={formatMonth}
              travellerWidth={10}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export { COLORS };
