"use client";

import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";
import { KpiCard } from "@/components/KpiCard";
import { ForecastChart } from "@/components/ForecastChart";
import { MonthSelector } from "@/components/MonthSelector";
import type { AggregateForecast } from "@/lib/types";

function toM(lakh: number): number {
  return lakh / 10;
}

function fmtM(n: number, decimals = 1): string {
  return n.toFixed(decimals) + " M";
}

function formatDate(m: string): string {
  const d = new Date(m.length === 7 ? m + "-01" : m);
  if (isNaN(d.getTime())) return m;
  return d.toLocaleDateString("en-IN", { month: "long", year: "numeric" });
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

      if (err) { setError(err.message); setLoading(false); return; }
      if (data) {
        setForecasts(data);
        const uniqueMonths = [...new Set(data.map((d) => d.forecast_month))].sort();
        setMonths(uniqueMonths);
        if (uniqueMonths.length > 1) setSelectedMonth(uniqueMonths[1]);
        else if (uniqueMonths.length > 0) setSelectedMonth(uniqueMonths[0]);
      }
      setLoading(false);
    }
    load();
  }, []);

  if (loading) return <div className="flex items-center justify-center h-64"><div className="text-gray-400">Loading forecasts...</div></div>;
  if (error) return <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center my-8"><p className="text-red-700 font-medium">Failed to load forecast data</p><p className="text-red-600 text-sm mt-1">{error}</p></div>;
  if (forecasts.length === 0) return <div className="text-center py-16"><h1 className="text-2xl font-bold text-gray-800 mb-4">MIP Dashboard</h1><p className="text-gray-500">No forecast data in database yet.</p></div>;

  const monthIdx = months.indexOf(selectedMonth);
  const prevMonth = monthIdx > 0 ? months[monthIdx - 1] : null;

  const monthData = forecasts.filter((f) => f.forecast_month === selectedMonth);
  const prevMonthData = prevMonth ? forecasts.filter((f) => f.forecast_month === prevMonth) : [];

  const get = (metric: string) => monthData.find((d) => d.metric === metric);
  const getPrev = (metric: string) => prevMonthData.find((d) => d.metric === metric);

  const ccOutstanding = get("cc_outstanding");
  const dcOutstanding = get("dc_outstanding");
  const upiVol = get("upi_vol");
  const ccTxn = get("cc_txn_vol");
  const dcTxn = get("dc_txn_vol");

  const ccPrev = getPrev("cc_outstanding");
  const dcPrev = getPrev("dc_outstanding");

  const ccManufacture = ccOutstanding && ccPrev ? ccOutstanding.yhat - ccPrev.yhat : null;
  const dcManufacture = dcOutstanding && dcPrev ? dcOutstanding.yhat - dcPrev.yhat : null;

  const ccChartData = forecasts.filter((f) => f.metric === "cc_outstanding").map((f) => ({
    month: f.forecast_month, forecast: f.yhat, lower: f.yhat_lower ?? undefined, upper: f.yhat_upper ?? undefined,
  }));
  const dcChartData = forecasts.filter((f) => f.metric === "dc_outstanding").map((f) => ({
    month: f.forecast_month, forecast: f.yhat, lower: f.yhat_lower ?? undefined, upper: f.yhat_upper ?? undefined,
  }));

  const METRIC_LABELS: Record<string, string> = {
    cc_outstanding: "Credit Card Outstanding",
    dc_outstanding: "Debit Card Outstanding",
    cc_txn_vol: "Credit Card Transactions",
    dc_txn_vol: "Debit Card Transactions",
    upi_vol: "UPI Volume",
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">Dashboard</h1>
          <p className="text-sm text-gray-400 mt-0.5">All values in Millions</p>
        </div>
        <MonthSelector months={months} selected={selectedMonth} onChange={setSelectedMonth} />
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-8">
        <KpiCard
          title="Credit Cards"
          value={ccOutstanding ? fmtM(toM(ccOutstanding.yhat)) : "—"}
          subtitle="Outstanding"
        />
        <KpiCard
          title="Debit Cards"
          value={dcOutstanding ? fmtM(toM(dcOutstanding.yhat)) : "—"}
          subtitle="Outstanding"
        />
        <KpiCard
          title="UPI Transactions"
          value={upiVol ? fmtM(toM(upiVol.yhat), 0) : "—"}
          subtitle="Monthly volume"
        />
        <KpiCard
          title="CC New Cards"
          value={ccManufacture !== null ? (ccManufacture >= 0 ? "+" : "") + fmtM(toM(ccManufacture), 2) : "—"}
          subtitle="To manufacture (MoM)"
          trend={ccManufacture !== null ? (ccManufacture >= 0 ? "Growth" : "Decline") : undefined}
          trendUp={ccManufacture !== null ? ccManufacture >= 0 : undefined}
        />
        <KpiCard
          title="DC New Cards"
          value={dcManufacture !== null ? (dcManufacture >= 0 ? "+" : "") + fmtM(toM(dcManufacture), 2) : "—"}
          subtitle="To manufacture (MoM)"
          trend={dcManufacture !== null ? (dcManufacture >= 0 ? "Growth" : "Decline") : undefined}
          trendUp={dcManufacture !== null ? dcManufacture >= 0 : undefined}
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <ForecastChart data={ccChartData} title="Credit Cards Outstanding" />
        <ForecastChart data={dcChartData} title="Debit Cards Outstanding" />
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl border border-gray-100 p-6 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-1">
          All Metrics — {formatDate(selectedMonth)}
        </h3>
        <p className="text-xs text-gray-400 mb-4">Values in Millions</p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-gray-500">
                <th className="pb-3 font-medium">Metric</th>
                <th className="pb-3 text-right font-medium">Forecast</th>
                <th className="pb-3 text-right font-medium">Lower 90%</th>
                <th className="pb-3 text-right font-medium">Upper 90%</th>
              </tr>
            </thead>
            <tbody>
              {monthData.map((d) => (
                <tr key={d.metric} className="border-b last:border-0 hover:bg-gray-50">
                  <td className="py-3 font-medium">{METRIC_LABELS[d.metric] || d.metric}</td>
                  <td className="py-3 text-right font-medium">{fmtM(toM(d.yhat))}</td>
                  <td className="py-3 text-right text-gray-400">{d.yhat_lower ? fmtM(toM(d.yhat_lower)) : "—"}</td>
                  <td className="py-3 text-right text-gray-400">{d.yhat_upper ? fmtM(toM(d.yhat_upper)) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
