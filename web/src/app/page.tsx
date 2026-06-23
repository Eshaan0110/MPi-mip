"use client";

import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";
import { KpiCard } from "@/components/KpiCard";
import { ForecastChart } from "@/components/ForecastChart";
import { MonthSelector } from "@/components/MonthSelector";
import type { AggregateForecast } from "@/lib/types";

function formatNumber(n: number, decimals = 1): string {
  if (n >= 1e7) return (n / 1e7).toFixed(decimals) + " Cr";
  if (n >= 1e5) return (n / 1e5).toFixed(decimals) + " L";
  if (n >= 1e3) return (n / 1e3).toFixed(decimals) + "K";
  return n.toFixed(decimals);
}

export default function DashboardPage() {
  const [forecasts, setForecasts] = useState<AggregateForecast[]>([]);
  const [months, setMonths] = useState<string[]>([]);
  const [selectedMonth, setSelectedMonth] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      const { data, error: err } = await supabase
        .from("forecasts_aggregate")
        .select("*")
        .order("forecast_month", { ascending: true });

      if (err) {
        setError(err.message);
        setLoading(false);
        return;
      }
      if (data) {
        setForecasts(data);
        const uniqueMonths = [...new Set(data.map((d) => d.forecast_month))].sort();
        setMonths(uniqueMonths);
        if (uniqueMonths.length > 0) setSelectedMonth(uniqueMonths[0]);
      }
      setLoading(false);
    }
    load();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading forecasts...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center my-8">
        <p className="text-red-700 font-medium">Failed to load forecast data</p>
        <p className="text-red-600 text-sm mt-1">{error}</p>
      </div>
    );
  }

  if (forecasts.length === 0) {
    return (
      <div className="text-center py-16">
        <h1 className="text-2xl font-bold text-brand-700 mb-4">
          MIP Dashboard
        </h1>
        <p className="text-gray-500 mb-2">
          No forecast data in database yet.
        </p>
        <p className="text-sm text-gray-400">
          Run the data migration script to populate Supabase, or wait for the
          monthly pipeline to execute.
        </p>
      </div>
    );
  }

  const monthData = forecasts.filter((f) => f.forecast_month === selectedMonth);
  const ccOutstanding = monthData.find((d) => d.metric === "cc_outstanding");
  const dcOutstanding = monthData.find((d) => d.metric === "dc_outstanding");
  const upiVol = monthData.find((d) => d.metric === "upi_vol");
  const ccTxn = monthData.find((d) => d.metric === "cc_txn_vol");
  const dcTxn = monthData.find((d) => d.metric === "dc_txn_vol");

  const ccChartData = forecasts
    .filter((f) => f.metric === "cc_outstanding")
    .map((f) => ({
      month: f.forecast_month,
      forecast: f.yhat,
      lower: f.yhat_lower ?? undefined,
      upper: f.yhat_upper ?? undefined,
    }));

  const dcChartData = forecasts
    .filter((f) => f.metric === "dc_outstanding")
    .map((f) => ({
      month: f.forecast_month,
      forecast: f.yhat,
      lower: f.yhat_lower ?? undefined,
      upper: f.yhat_upper ?? undefined,
    }));

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-brand-700">Dashboard</h1>
        <MonthSelector
          months={months}
          selected={selectedMonth}
          onChange={setSelectedMonth}
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-8">
        <KpiCard
          title="CC Outstanding"
          value={ccOutstanding ? formatNumber(ccOutstanding.yhat) : "—"}
          subtitle="Credit cards in market"
        />
        <KpiCard
          title="DC Outstanding"
          value={dcOutstanding ? formatNumber(dcOutstanding.yhat) : "—"}
          subtitle="Debit cards in market"
        />
        <KpiCard
          title="UPI Volume"
          value={upiVol ? formatNumber(upiVol.yhat) + " Mn" : "—"}
          subtitle="Monthly transactions"
        />
        <KpiCard
          title="CC Txn Volume"
          value={ccTxn ? formatNumber(ccTxn.yhat) : "—"}
          subtitle="CC transactions"
        />
        <KpiCard
          title="DC Txn Volume"
          value={dcTxn ? formatNumber(dcTxn.yhat) : "—"}
          subtitle="DC transactions"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <ForecastChart data={ccChartData} title="Credit Cards Outstanding — Forecast" />
        <ForecastChart data={dcChartData} title="Debit Cards Outstanding — Forecast" />
      </div>

      <div className="bg-white rounded-lg border border-gray-200 p-5 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-4">
          All Metrics — {new Date(selectedMonth + "-01").toLocaleDateString("en-IN", { month: "long", year: "numeric" })}
        </h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-gray-500">
              <th className="pb-2">Metric</th>
              <th className="pb-2 text-right">Forecast</th>
              <th className="pb-2 text-right">Lower 90%</th>
              <th className="pb-2 text-right">Upper 90%</th>
            </tr>
          </thead>
          <tbody>
            {monthData.map((d) => (
              <tr key={d.metric} className="border-b last:border-0">
                <td className="py-2 font-medium">{d.metric.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}</td>
                <td className="py-2 text-right">{formatNumber(d.yhat)}</td>
                <td className="py-2 text-right text-gray-400">{d.yhat_lower ? formatNumber(d.yhat_lower) : "—"}</td>
                <td className="py-2 text-right text-gray-400">{d.yhat_upper ? formatNumber(d.yhat_upper) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
