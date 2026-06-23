"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  ComposedChart,
} from "recharts";

interface DataPoint {
  month: string;
  actual?: number;
  forecast?: number;
  lower?: number;
  upper?: number;
}

interface ForecastChartProps {
  data: DataPoint[];
  title: string;
  yLabel?: string;
}

export function ForecastChart({ data, title, yLabel }: ForecastChartProps) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-5 shadow-sm">
      <h3 className="text-sm font-semibold text-gray-700 mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="month" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", style: { fontSize: 11 } } : undefined} />
          <Tooltip />
          {data.some((d) => d.lower !== undefined) && (
            <Area
              dataKey="upper"
              stroke="none"
              fill="#2e75b6"
              fillOpacity={0.1}
              name="Upper CI"
            />
          )}
          {data.some((d) => d.lower !== undefined) && (
            <Area
              dataKey="lower"
              stroke="none"
              fill="#ffffff"
              fillOpacity={1}
              name="Lower CI"
            />
          )}
          {data.some((d) => d.actual !== undefined) && (
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#1b3a5c"
              strokeWidth={2}
              dot={{ r: 2 }}
              name="Actual"
            />
          )}
          {data.some((d) => d.forecast !== undefined) && (
            <Line
              type="monotone"
              dataKey="forecast"
              stroke="#2e75b6"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={{ r: 2 }}
              name="Forecast"
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
