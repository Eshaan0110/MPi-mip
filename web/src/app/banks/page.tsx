"use client";

import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";
import { KpiCard } from "@/components/KpiCard";
import { ForecastChart, COLORS } from "@/components/ForecastChart";
import { MonthSelector } from "@/components/MonthSelector";
import type { BankForecast } from "@/lib/types";

function toM(v: number): number { return v / 1_000_000; }
function fmtM(n: number, decimals = 1): string { return n.toFixed(decimals) + " M"; }

const ALLOWED_CC_BANKS = new Set([
  "HDFC Bank", "State Bank of India", "ICICI Bank", "Axis Bank",
  "Kotak Mahindra Bank", "IndusInd Bank", "Bank of Baroda",
  "Yes Bank", "Canara Bank", "HSBC",
]);

const ALLOWED_DC_BANKS = new Set([
  "State Bank of India", "Bank of Baroda", "Canara Bank", "HDFC Bank",
  "Union Bank of India", "Punjab National Bank", "Axis Bank",
  "Bank of India", "Kotak Mahindra Bank", "Indian Bank",
  "Central Bank of India", "UCO Bank", "ICICI Bank",
  "Indian Overseas Bank", "Paytm Payments Bank",
]);

function formatDate(m: string): string {
  const d = new Date(m.length === 7 ? m + "-01" : m);
  if (isNaN(d.getTime())) return m;
  return d.toLocaleDateString("en-IN", { month: "long", year: "numeric" });
}

export default function BankExplorerPage() {
  const [forecasts, setForecasts] = useState<BankForecast[]>([]);
  const [months, setMonths] = useState<string[]>([]);
  const [selectedMonth, setSelectedMonth] = useState("");
  const [cardType, setCardType] = useState<"CC" | "DC">("CC");
  const [selectedBanks, setSelectedBanks] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"single" | "compare">("single");

  useEffect(() => {
    async function load() {
      const { data, error: err } = await supabase
        .from("forecasts_bank")
        .select("*")
        .order("forecast_month", { ascending: true });

      if (err) { setError(err.message); setLoading(false); return; }
      if (data) {
        setForecasts(data);
        const uniqueMonths = [...new Set(data.map((d) => d.forecast_month))].sort();
        setMonths(uniqueMonths);
        if (uniqueMonths.length > 0) setSelectedMonth(uniqueMonths[0]);
        const ccBanks = [...new Set(data.filter((d) => d.card_type === "CC").map((d) => d.bank_name))].filter((b) => ALLOWED_CC_BANKS.has(b)).sort();
        if (ccBanks.length > 0) setSelectedBanks([ccBanks[0]]);
      }
      setLoading(false);
    }
    load();
  }, []);

  if (loading) return <div className="flex items-center justify-center h-64"><div className="text-gray-400">Loading...</div></div>;
  if (error) return <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center my-8"><p className="text-red-700 font-medium">Failed to load bank data</p><p className="text-red-600 text-sm mt-1">{error}</p></div>;
  if (forecasts.length === 0) return <div className="text-center py-16"><h1 className="text-2xl font-bold text-gray-800 mb-4">Bank Explorer</h1><p className="text-gray-500">No bank forecast data available yet.</p></div>;

  const allowedSet = cardType === "CC" ? ALLOWED_CC_BANKS : ALLOWED_DC_BANKS;
  const banksForType = [...new Set(forecasts.filter((f) => f.card_type === cardType).map((f) => f.bank_name))]
    .filter((b) => allowedSet.has(b))
    .sort();

  const handleCardTypeChange = (ct: "CC" | "DC") => {
    setCardType(ct);
    const allowed = ct === "CC" ? ALLOWED_CC_BANKS : ALLOWED_DC_BANKS;
    const newBanks = [...new Set(forecasts.filter((f) => f.card_type === ct).map((f) => f.bank_name))].filter((b) => allowed.has(b)).sort();
    const kept = selectedBanks.filter((b) => newBanks.includes(b));
    if (kept.length === 0 && newBanks.length > 0) setSelectedBanks([newBanks[0]]);
    else setSelectedBanks(kept);
  };

  const toggleBank = (bank: string) => {
    if (tab === "single") {
      setSelectedBanks([bank]);
    } else {
      setSelectedBanks((prev) =>
        prev.includes(bank) ? prev.filter((b) => b !== bank) : [...prev, bank]
      );
    }
  };

  const primaryBank = selectedBanks[0] || "";

  // Data for single bank view — pre-convert to lakhs so ForecastChart's /10 gives Millions
  const RAW_TO_LAKH = 1 / 100_000;
  const bankChartData = forecasts
    .filter((f) => f.bank_name === primaryBank && f.card_type === cardType)
    .map((f) => ({
      month: f.forecast_month,
      forecast: f.yhat * RAW_TO_LAKH,
      lower: f.yhat_lower != null ? f.yhat_lower * RAW_TO_LAKH : undefined,
      upper: f.yhat_upper != null ? f.yhat_upper * RAW_TO_LAKH : undefined,
    }));

  // Data for multi-bank comparison
  const allMonthsForType = [...new Set(
    forecasts.filter((f) => f.card_type === cardType && selectedBanks.includes(f.bank_name)).map((f) => f.forecast_month)
  )].sort();

  const multiData = allMonthsForType.map((m) => {
    const row: any = { month: m };
    for (const bank of selectedBanks) {
      const rec = forecasts.find((f) => f.bank_name === bank && f.card_type === cardType && f.forecast_month === m);
      row[bank] = rec ? rec.yhat * RAW_TO_LAKH : undefined;
    }
    return row;
  });

  const multiLines = selectedBanks.map((bank, i) => ({
    key: bank,
    label: bank,
    color: COLORS[i % COLORS.length],
  }));

  // Ranked table data
  const monthIdx = months.indexOf(selectedMonth);
  const prevMonth = monthIdx > 0 ? months[monthIdx - 1] : null;

  const bankMonthData = forecasts.filter((f) => f.forecast_month === selectedMonth && f.card_type === cardType && allowedSet.has(f.bank_name));
  const prevBankData = prevMonth ? forecasts.filter((f) => f.forecast_month === prevMonth && f.card_type === cardType) : [];

  const rankedBanks = [...bankMonthData].sort((a, b) => b.yhat - a.yhat).map((d) => {
    const prev = prevBankData.find((p) => p.bank_name === d.bank_name);
    const manufacture = prev ? d.yhat - prev.yhat : null;
    return { ...d, manufacture };
  });

  // KPIs for primary bank
  const primaryData = bankMonthData.find((f) => f.bank_name === primaryBank);
  const primaryPrev = prevBankData.find((f) => f.bank_name === primaryBank);
  const primaryManufacture = primaryData && primaryPrev ? primaryData.yhat - primaryPrev.yhat : null;

  return (
    <div>
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">Bank Explorer</h1>
          <p className="text-sm text-gray-400 mt-0.5">All values in Millions</p>
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          {/* Card type toggle */}
          <div className="flex rounded-lg border border-gray-200 overflow-hidden">
            {(["CC", "DC"] as const).map((ct) => (
              <button
                key={ct}
                onClick={() => handleCardTypeChange(ct)}
                className={`px-4 py-2 text-sm font-medium transition-colors ${
                  cardType === ct ? "bg-gray-800 text-white" : "bg-white text-gray-600 hover:bg-gray-50"
                }`}
              >
                {ct === "CC" ? "Credit Card" : "Debit Card"}
              </button>
            ))}
          </div>
          {/* View toggle */}
          <div className="flex rounded-lg border border-gray-200 overflow-hidden">
            <button
              onClick={() => { setTab("single"); if (selectedBanks.length > 1) setSelectedBanks([selectedBanks[0]]); }}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                tab === "single" ? "bg-gray-800 text-white" : "bg-white text-gray-600 hover:bg-gray-50"
              }`}
            >
              Single Bank
            </button>
            <button
              onClick={() => setTab("compare")}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                tab === "compare" ? "bg-gray-800 text-white" : "bg-white text-gray-600 hover:bg-gray-50"
              }`}
            >
              Compare Banks
            </button>
          </div>
          <MonthSelector months={months} selected={selectedMonth} onChange={setSelectedMonth} />
        </div>
      </div>

      {/* Bank selector */}
      {tab === "single" ? (
        <div className="mb-6">
          <select
            value={primaryBank}
            onChange={(e) => setSelectedBanks([e.target.value])}
            className="border border-gray-200 rounded-lg px-4 py-2.5 text-sm bg-white w-full max-w-xs"
          >
            {banksForType.map((b) => (
              <option key={b} value={b}>{b}</option>
            ))}
          </select>
        </div>
      ) : (
        <div className="bg-white rounded-xl border border-gray-100 p-5 shadow-sm mb-6">
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm font-medium text-gray-700">Select banks to compare (click to toggle):</p>
            <div className="flex gap-2">
              <button
                onClick={() => setSelectedBanks([])}
                className="text-xs px-3 py-1 rounded-md border border-gray-200 text-gray-500 hover:bg-gray-50"
              >
                Clear all
              </button>
              <button
                onClick={() => setSelectedBanks(banksForType.slice(0, 5))}
                className="text-xs px-3 py-1 rounded-md border border-gray-200 text-gray-500 hover:bg-gray-50"
              >
                Top 5
              </button>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {banksForType.map((bank) => {
              const isSelected = selectedBanks.includes(bank);
              const colorIdx = isSelected ? selectedBanks.indexOf(bank) : -1;
              return (
                <button
                  key={bank}
                  onClick={() => toggleBank(bank)}
                  className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all border ${
                    isSelected
                      ? "text-white border-transparent"
                      : "bg-white text-gray-600 border-gray-200 hover:border-gray-400"
                  }`}
                  style={isSelected ? { backgroundColor: COLORS[colorIdx % COLORS.length] } : {}}
                >
                  {bank}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* KPI Cards — single view */}
      {tab === "single" && primaryData && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <KpiCard
            title="Forecast"
            value={fmtM(toM(primaryData.yhat))}
            subtitle={`${primaryBank}`}
          />
          <KpiCard
            title="90% CI Range"
            value={primaryData.yhat_lower && primaryData.yhat_upper
              ? `${fmtM(toM(primaryData.yhat_lower))} – ${fmtM(toM(primaryData.yhat_upper))}`
              : "—"}
          />
          <KpiCard
            title="Cards to Manufacture"
            value={primaryManufacture !== null ? (primaryManufacture >= 0 ? "+" : "") + fmtM(toM(primaryManufacture), 2) : "—"}
            subtitle="Net new cards (MoM)"
            trend={primaryManufacture !== null ? (primaryManufacture >= 0 ? "Growth" : "Decline") : undefined}
            trendUp={primaryManufacture !== null ? primaryManufacture >= 0 : undefined}
          />
          <KpiCard title="Model" value={primaryData.model_type || "Prophet"} subtitle="Forecast method" />
        </div>
      )}

      {/* Chart */}
      <div className="mb-6">
        {tab === "single" ? (
          <ForecastChart data={bankChartData} title={`${primaryBank} — ${cardType === "CC" ? "Credit Card" : "Debit Card"} Forecast`} />
        ) : selectedBanks.length > 0 ? (
          <ForecastChart
            data={[]}
            title={`Bank Comparison — ${cardType === "CC" ? "Credit Card" : "Debit Card"} Outstanding`}
            multiLines={multiLines}
            multiData={multiData}
          />
        ) : (
          <div className="bg-white rounded-xl border border-gray-100 p-12 shadow-sm text-center">
            <p className="text-gray-400">Select at least one bank to see the chart</p>
          </div>
        )}
      </div>

      {/* Ranked Table */}
      <div className="bg-white rounded-xl border border-gray-100 p-6 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-1">
          All Banks — {cardType === "CC" ? "Credit Card" : "Debit Card"} — {formatDate(selectedMonth)}
        </h3>
        <p className="text-xs text-gray-400 mb-4">Click a row to select the bank. Values in Millions.</p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-gray-500">
                <th className="pb-3 font-medium w-8">#</th>
                <th className="pb-3 font-medium">Bank</th>
                <th className="pb-3 text-right font-medium">Forecast</th>
                <th className="pb-3 text-right font-medium">90% CI</th>
                <th className="pb-3 text-right font-medium">New Cards (MoM)</th>
                <th className="pb-3 text-right font-medium">Model</th>
              </tr>
            </thead>
            <tbody>
              {rankedBanks.map((d, i) => {
                const isSelected = selectedBanks.includes(d.bank_name);
                return (
                  <tr
                    key={d.bank_name}
                    className={`border-b last:border-0 cursor-pointer transition-colors ${
                      isSelected ? "bg-blue-50 font-medium" : "hover:bg-gray-50"
                    }`}
                    onClick={() => toggleBank(d.bank_name)}
                  >
                    <td className="py-3 text-gray-400">{i + 1}</td>
                    <td className="py-3">
                      {isSelected && tab === "compare" && (
                        <span
                          className="inline-block w-2.5 h-2.5 rounded-full mr-2"
                          style={{ backgroundColor: COLORS[selectedBanks.indexOf(d.bank_name) % COLORS.length] }}
                        />
                      )}
                      {d.bank_name}
                    </td>
                    <td className="py-3 text-right font-medium">{fmtM(toM(d.yhat))}</td>
                    <td className="py-3 text-right text-gray-400">
                      {d.yhat_lower && d.yhat_upper
                        ? `${fmtM(toM(d.yhat_lower))} – ${fmtM(toM(d.yhat_upper))}`
                        : "—"}
                    </td>
                    <td className={`py-3 text-right font-medium ${
                      d.manufacture !== null ? (d.manufacture >= 0 ? "text-green-600" : "text-red-500") : "text-gray-400"
                    }`}>
                      {d.manufacture !== null
                        ? (d.manufacture >= 0 ? "+" : "") + fmtM(toM(d.manufacture), 2)
                        : "—"}
                    </td>
                    <td className="py-3 text-right text-gray-400">{d.model_type || "Prophet"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
