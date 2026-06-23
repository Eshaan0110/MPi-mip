"use client";

import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";
import { KpiCard } from "@/components/KpiCard";
import { ForecastChart } from "@/components/ForecastChart";
import { MonthSelector } from "@/components/MonthSelector";
import type { BankForecast } from "@/lib/types";

function formatNumber(n: number): string {
  if (n >= 1e7) return (n / 1e7).toFixed(2) + " Cr";
  if (n >= 1e5) return (n / 1e5).toFixed(2) + " L";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return n.toFixed(0);
}

export default function BankExplorerPage() {
  const [forecasts, setForecasts] = useState<BankForecast[]>([]);
  const [banks, setBanks] = useState<string[]>([]);
  const [months, setMonths] = useState<string[]>([]);
  const [selectedBank, setSelectedBank] = useState("");
  const [selectedMonth, setSelectedMonth] = useState("");
  const [cardType, setCardType] = useState<"CC" | "DC">("CC");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      const { data, error: err } = await supabase
        .from("forecasts_bank")
        .select("*")
        .order("forecast_month", { ascending: true });

      if (err) {
        setError(err.message);
        setLoading(false);
        return;
      }
      if (data) {
        setForecasts(data);
        const uniqueBanks = [...new Set(data.map((d) => d.bank_name))].sort();
        const uniqueMonths = [...new Set(data.map((d) => d.forecast_month))].sort();
        setBanks(uniqueBanks);
        setMonths(uniqueMonths);
        if (uniqueBanks.length > 0) setSelectedBank(uniqueBanks[0]);
        if (uniqueMonths.length > 0) setSelectedMonth(uniqueMonths[0]);
      }
      setLoading(false);
    }
    load();
  }, []);

  if (loading) {
    return <div className="flex items-center justify-center h-64"><div className="text-gray-400">Loading...</div></div>;
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center my-8">
        <p className="text-red-700 font-medium">Failed to load bank data</p>
        <p className="text-red-600 text-sm mt-1">{error}</p>
      </div>
    );
  }

  if (forecasts.length === 0) {
    return (
      <div className="text-center py-16">
        <h1 className="text-2xl font-bold text-brand-700 mb-4">Bank Explorer</h1>
        <p className="text-gray-500">No bank forecast data available yet.</p>
      </div>
    );
  }

  const bankMonthData = forecasts.filter(
    (f) => f.forecast_month === selectedMonth && f.card_type === cardType
  );
  const selectedBankData = bankMonthData.find((f) => f.bank_name === selectedBank);

  const bankChartData = forecasts
    .filter((f) => f.bank_name === selectedBank && f.card_type === cardType)
    .map((f) => ({
      month: f.forecast_month,
      forecast: f.yhat,
      lower: f.yhat_lower ?? undefined,
      upper: f.yhat_upper ?? undefined,
    }));

  const rankedBanks = [...bankMonthData].sort((a, b) => b.yhat - a.yhat);

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-brand-700">Bank Explorer</h1>
        <div className="flex items-center gap-3">
          <div className="flex rounded-md border border-gray-300 overflow-hidden">
            {(["CC", "DC"] as const).map((ct) => (
              <button
                key={ct}
                onClick={() => setCardType(ct)}
                className={`px-3 py-1.5 text-sm font-medium ${
                  cardType === ct
                    ? "bg-brand-500 text-white"
                    : "bg-white text-gray-600 hover:bg-gray-50"
                }`}
              >
                {ct === "CC" ? "Credit" : "Debit"}
              </button>
            ))}
          </div>
          <select
            value={selectedBank}
            onChange={(e) => setSelectedBank(e.target.value)}
            className="border border-gray-300 rounded-md px-3 py-2 text-sm bg-white"
          >
            {banks
              .filter((b) => forecasts.some((f) => f.bank_name === b && f.card_type === cardType))
              .map((b) => (
                <option key={b} value={b}>{b}</option>
              ))}
          </select>
          <MonthSelector months={months} selected={selectedMonth} onChange={setSelectedMonth} />
        </div>
      </div>

      {selectedBankData && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <KpiCard title="Forecast" value={formatNumber(selectedBankData.yhat)} subtitle={`${selectedBank} — ${cardType}`} />
          <KpiCard title="Lower 90% CI" value={selectedBankData.yhat_lower ? formatNumber(selectedBankData.yhat_lower) : "—"} />
          <KpiCard title="Upper 90% CI" value={selectedBankData.yhat_upper ? formatNumber(selectedBankData.yhat_upper) : "—"} />
          <KpiCard title="Model" value={selectedBankData.model_type || "Prophet"} />
        </div>
      )}

      <div className="mb-6">
        <ForecastChart data={bankChartData} title={`${selectedBank} — ${cardType} Forecast`} />
      </div>

      <div className="bg-white rounded-lg border border-gray-200 p-5 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-4">
          All Banks — {cardType} —{" "}
          {new Date(selectedMonth + "-01").toLocaleDateString("en-IN", { month: "long", year: "numeric" })}
        </h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-gray-500">
              <th className="pb-2">#</th>
              <th className="pb-2">Bank</th>
              <th className="pb-2 text-right">Forecast</th>
              <th className="pb-2 text-right">Lower</th>
              <th className="pb-2 text-right">Upper</th>
              <th className="pb-2 text-right">Model</th>
            </tr>
          </thead>
          <tbody>
            {rankedBanks.map((d, i) => (
              <tr
                key={d.bank_name}
                className={`border-b last:border-0 cursor-pointer hover:bg-brand-50 ${
                  d.bank_name === selectedBank ? "bg-brand-50 font-medium" : ""
                }`}
                onClick={() => setSelectedBank(d.bank_name)}
              >
                <td className="py-2 text-gray-400">{i + 1}</td>
                <td className="py-2">{d.bank_name}</td>
                <td className="py-2 text-right">{formatNumber(d.yhat)}</td>
                <td className="py-2 text-right text-gray-400">{d.yhat_lower ? formatNumber(d.yhat_lower) : "—"}</td>
                <td className="py-2 text-right text-gray-400">{d.yhat_upper ? formatNumber(d.yhat_upper) : "—"}</td>
                <td className="py-2 text-right text-gray-400">{d.model_type || "Prophet"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
