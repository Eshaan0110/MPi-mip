"use client";

import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";
import { KpiCard } from "@/components/KpiCard";
import { ForecastChart, COLORS, type MultiLinePoint } from "@/components/ForecastChart";
import { MonthSelector } from "@/components/MonthSelector";
import type { BankForecast, ProcessedBankSeries } from "@/lib/types";
import { ALLOWED_CC_BANKS, ALLOWED_DC_BANKS, displayBank } from "@/lib/constants";

function toM(v: number): number { return v / 1_000_000; }
function fmtM(n: number, decimals = 1): string { return n.toFixed(decimals) + " M"; }

function formatDate(m: string): string {
  const d = new Date(m.length === 7 ? m + "-01" : m);
  if (isNaN(d.getTime())) return m;
  return d.toLocaleDateString("en-IN", { month: "long", year: "numeric" });
}

export default function BankExplorerPage() {
  const [forecasts, setForecasts] = useState<BankForecast[]>([]);
  const [historicals, setHistoricals] = useState<ProcessedBankSeries[]>([]);
  const [months, setMonths] = useState<string[]>([]);
  const [selectedMonth, setSelectedMonth] = useState("");
  const [cardType, setCardType] = useState<"CC" | "DC">("CC");
  const [selectedBanks, setSelectedBanks] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"single" | "compare">("single");
  const [fromMonth, setFromMonth] = useState("");
  const [toMonth, setToMonth] = useState("");

  useEffect(() => {
    async function load() {
      const [fcRes, histRes] = await Promise.all([
        supabase.from("forecasts_bank").select("*").order("forecast_month", { ascending: true }),
        supabase.from("processed_bank_series").select("*").order("month", { ascending: true }),
      ]);

      if (fcRes.error) { setError(fcRes.error.message); setLoading(false); return; }
      if (fcRes.data) {
        setForecasts(fcRes.data);
        const uniqueMonths = [...new Set(fcRes.data.map((d) => d.forecast_month))].sort();
        setMonths(uniqueMonths);
        if (uniqueMonths.length > 0) {
          setSelectedMonth(uniqueMonths[uniqueMonths.length - 1]);
          setFromMonth(uniqueMonths[0]);
          setToMonth(uniqueMonths[uniqueMonths.length - 1]);
        }
        const ccBanks = [...new Set(fcRes.data.filter((d) => d.card_type === "CC").map((d) => d.bank_name))].filter((b) => ALLOWED_CC_BANKS.has(b)).sort();
        if (ccBanks.length > 0) setSelectedBanks([ccBanks[0]]);
      }
      if (histRes.data) setHistoricals(histRes.data);
      setLoading(false);
    }
    load();
  }, []);

  if (loading) return <div className="flex items-center justify-center h-64"><div className="text-gray-400 dark:text-slate-500">Loading...</div></div>;
  if (error) return <div className="bg-red-100 border border-red-300 dark:bg-red-900/30 dark:border-red-700/50 rounded-lg p-6 text-center my-8"><p className="text-red-700 dark:text-red-400 font-medium">Failed to load bank data</p><p className="text-red-500 text-sm mt-1">{error}</p></div>;
  if (forecasts.length === 0) return <div className="text-center py-16"><h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">Bank Explorer</h1><p className="text-gray-400 dark:text-slate-500">No bank forecast data available yet.</p></div>;

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
    if (tab === "single") { setSelectedBanks([bank]); }
    else { setSelectedBanks((prev) => prev.includes(bank) ? prev.filter((b) => b !== bank) : [...prev, bank]); }
  };

  const primaryBank = selectedBanks[0] || "";
  const RAW_TO_LAKH = 1 / 100_000;

  const bankChartData = (() => {
    const fcByMonth = new Map(
      forecasts
        .filter((f) => f.bank_name === primaryBank && f.card_type === cardType)
        .map((f) => [f.forecast_month, f])
    );
    const actByMonth = new Map(
      historicals
        .filter((h) => h.bank_name === primaryBank && h.card_type === cardType)
        .map((h) => [h.month, h])
    );
    const allM = [...new Set([...fcByMonth.keys(), ...actByMonth.keys()])].sort()
      .filter((m) => (!fromMonth || m >= fromMonth) && (!toMonth || m <= toMonth));
    return allM.map((m) => {
      const fc = fcByMonth.get(m);
      const act = actByMonth.get(m);
      return {
        month: m,
        actual: act ? act.y * RAW_TO_LAKH : undefined,
        forecast: fc ? fc.yhat * RAW_TO_LAKH : undefined,
        lower: fc?.yhat_lower != null ? fc.yhat_lower * RAW_TO_LAKH : undefined,
        upper: fc?.yhat_upper != null ? fc.yhat_upper * RAW_TO_LAKH : undefined,
      };
    });
  })();

  const allMonthsForType = [...new Set(
    forecasts.filter((f) => f.card_type === cardType && selectedBanks.includes(f.bank_name) && (!fromMonth || f.forecast_month >= fromMonth) && (!toMonth || f.forecast_month <= toMonth)).map((f) => f.forecast_month)
  )].sort();

  const multiData = allMonthsForType.map((m) => {
    const row: MultiLinePoint = { month: m };
    for (const bank of selectedBanks) {
      const rec = forecasts.find((f) => f.bank_name === bank && f.card_type === cardType && f.forecast_month === m);
      row[bank] = rec ? rec.yhat * RAW_TO_LAKH : undefined;
    }
    return row;
  });

  const multiLines = selectedBanks.map((bank, i) => ({
    key: bank, label: displayBank(bank), color: COLORS[i % COLORS.length],
  }));

  const monthIdx = months.indexOf(selectedMonth);
  const prevMonth = monthIdx > 0 ? months[monthIdx - 1] : null;
  const bankMonthData = forecasts.filter((f) => f.forecast_month === selectedMonth && f.card_type === cardType && allowedSet.has(f.bank_name));
  const prevBankData = prevMonth ? forecasts.filter((f) => f.forecast_month === prevMonth && f.card_type === cardType) : [];

  const rankedBanks = [...bankMonthData].sort((a, b) => b.yhat - a.yhat).map((d) => {
    const prev = prevBankData.find((p) => p.bank_name === d.bank_name);
    return { ...d, manufacture: prev ? d.yhat - prev.yhat : null };
  });

  const primaryData = bankMonthData.find((f) => f.bank_name === primaryBank);
  const primaryPrev = prevBankData.find((f) => f.bank_name === primaryBank);
  const primaryManufacture = primaryData && primaryPrev ? primaryData.yhat - primaryPrev.yhat : null;

  return (
    <div>
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Bank Explorer</h1>
          <p className="text-sm text-gray-400 dark:text-slate-500 mt-0.5">All values in Millions</p>
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex rounded-lg border border-gray-300 dark:border-slate-600 overflow-hidden">
            {(["CC", "DC"] as const).map((ct) => (
              <button key={ct} onClick={() => handleCardTypeChange(ct)}
                className={`px-4 py-2 text-sm font-medium transition-colors ${cardType === ct ? "bg-blue-600 text-white" : "bg-white dark:bg-slate-800 text-gray-500 dark:text-slate-400 hover:bg-gray-200 dark:hover:bg-slate-700"}`}>
                {ct === "CC" ? "Credit Card" : "Debit Card"}
              </button>
            ))}
          </div>
          <div className="flex rounded-lg border border-gray-300 dark:border-slate-600 overflow-hidden">
            <button onClick={() => { setTab("single"); if (selectedBanks.length > 1) setSelectedBanks([selectedBanks[0]]); }}
              className={`px-4 py-2 text-sm font-medium transition-colors ${tab === "single" ? "bg-blue-600 text-white" : "bg-white dark:bg-slate-800 text-gray-500 dark:text-slate-400 hover:bg-gray-200 dark:hover:bg-slate-700"}`}>Single Bank</button>
            <button onClick={() => setTab("compare")}
              className={`px-4 py-2 text-sm font-medium transition-colors ${tab === "compare" ? "bg-blue-600 text-white" : "bg-white dark:bg-slate-800 text-gray-500 dark:text-slate-400 hover:bg-gray-200 dark:hover:bg-slate-700"}`}>Compare Banks</button>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-500 dark:text-slate-400">From</label>
            <input type="month" value={fromMonth.substring(0, 7)} onChange={(e) => setFromMonth(e.target.value)}
              max={toMonth.substring(0, 7)}
              className="border border-gray-300 dark:border-slate-600 rounded-md px-2 py-1.5 text-sm bg-white dark:bg-slate-800 text-gray-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500" />
            <label className="text-xs text-gray-500 dark:text-slate-400">To</label>
            <input type="month" value={toMonth.substring(0, 7)} onChange={(e) => setToMonth(e.target.value)}
              min={fromMonth.substring(0, 7)}
              className="border border-gray-300 dark:border-slate-600 rounded-md px-2 py-1.5 text-sm bg-white dark:bg-slate-800 text-gray-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500" />
          </div>
          <MonthSelector months={months} selected={selectedMonth} onChange={setSelectedMonth} />
        </div>
      </div>

      {tab === "single" ? (
        <div className="mb-6">
          <select value={primaryBank} onChange={(e) => setSelectedBanks([e.target.value])}
            className="border border-gray-300 dark:border-slate-600 rounded-lg px-4 py-2.5 text-sm bg-white dark:bg-slate-800 text-gray-800 dark:text-slate-200 w-full max-w-xs focus:outline-none focus:ring-2 focus:ring-blue-500">
            {banksForType.map((b) => (<option key={b} value={b}>{displayBank(b)}</option>))}
          </select>
        </div>
      ) : (
        <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-5 mb-6">
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm font-medium text-gray-700 dark:text-slate-300">Select banks to compare:</p>
            <div className="flex gap-2">
              <button onClick={() => setSelectedBanks([])} className="text-xs px-3 py-1 rounded-md bg-gray-200 dark:bg-slate-700 text-gray-500 dark:text-slate-400 hover:bg-gray-300 dark:hover:bg-slate-600">Clear all</button>
              <button onClick={() => setSelectedBanks(banksForType.slice(0, 5))} className="text-xs px-3 py-1 rounded-md bg-gray-200 dark:bg-slate-700 text-gray-500 dark:text-slate-400 hover:bg-gray-300 dark:hover:bg-slate-600">Top 5</button>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {banksForType.map((bank) => {
              const isSelected = selectedBanks.includes(bank);
              const colorIdx = isSelected ? selectedBanks.indexOf(bank) : -1;
              return (
                <button key={bank} onClick={() => toggleBank(bank)}
                  className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all border ${isSelected ? "text-white border-transparent" : "bg-white dark:bg-slate-800 text-gray-500 dark:text-slate-400 border-gray-300 dark:border-slate-600 hover:border-gray-400 dark:hover:border-slate-400"}`}
                  style={isSelected ? { backgroundColor: COLORS[colorIdx % COLORS.length] } : {}}>
                  {displayBank(bank)}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {tab === "single" && primaryData && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <KpiCard title="Forecast" value={fmtM(toM(primaryData.yhat))} subtitle={displayBank(primaryBank)} />
          <KpiCard title="90% CI Range" value={primaryData.yhat_lower && primaryData.yhat_upper ? `${fmtM(toM(primaryData.yhat_lower))} – ${fmtM(toM(primaryData.yhat_upper))}` : "—"} />
          <KpiCard title="Net New Cards (Est.)" value={primaryManufacture !== null ? (primaryManufacture >= 0 ? "+" : "") + fmtM(toM(primaryManufacture), 2) : "—"} subtitle="MoM change in outstanding" trend={primaryManufacture !== null ? (primaryManufacture >= 0 ? "Growth" : "Decline") : undefined} trendUp={primaryManufacture !== null ? primaryManufacture >= 0 : undefined} />
          <KpiCard title="Model" value={primaryData.model_type || "Prophet"} subtitle="Forecast method" />
        </div>
      )}

      <div className="mb-6">
        {tab === "single" ? (
          <ForecastChart data={bankChartData} title={`${displayBank(primaryBank)} — ${cardType === "CC" ? "Credit Card" : "Debit Card"} Forecast`} highlightMonth={selectedMonth} />
        ) : selectedBanks.length > 0 ? (
          <ForecastChart data={[]} title={`Bank Comparison — ${cardType === "CC" ? "Credit Card" : "Debit Card"} Outstanding`} multiLines={multiLines} multiData={multiData} highlightMonth={selectedMonth} />
        ) : (
          <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-12 text-center"><p className="text-gray-400 dark:text-slate-500">Select at least one bank to see the chart</p></div>
        )}
      </div>

      <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-6">
        <h3 className="text-sm font-semibold text-gray-800 dark:text-slate-200 mb-1">All Banks — {cardType === "CC" ? "Credit Card" : "Debit Card"} — {formatDate(selectedMonth)}</h3>
        <p className="text-xs text-gray-400 dark:text-slate-500 mb-4">Click a row to select the bank. Values in Millions.</p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-slate-700 text-left text-gray-500 dark:text-slate-400">
                <th className="pb-3 font-medium w-8">#</th>
                <th className="pb-3 font-medium">Bank</th>
                <th className="pb-3 text-right font-medium">Forecast</th>
                <th className="pb-3 text-right font-medium">90% CI</th>
                <th className="pb-3 text-right font-medium">Net New (Est. MoM)</th>
                <th className="pb-3 text-right font-medium">Model</th>
              </tr>
            </thead>
            <tbody>
              {rankedBanks.map((d, i) => {
                const isSelected = selectedBanks.includes(d.bank_name);
                return (
                  <tr key={d.bank_name} className={`border-b border-gray-200 dark:border-slate-700/50 last:border-0 cursor-pointer transition-colors ${isSelected ? "bg-blue-50 dark:bg-blue-900/20" : "hover:bg-gray-50 dark:hover:bg-slate-700/30"}`} onClick={() => toggleBank(d.bank_name)}>
                    <td className="py-3 text-gray-400 dark:text-slate-500">{i + 1}</td>
                    <td className="py-3 text-gray-800 dark:text-slate-200">
                      {isSelected && tab === "compare" && <span className="inline-block w-2.5 h-2.5 rounded-full mr-2" style={{ backgroundColor: COLORS[selectedBanks.indexOf(d.bank_name) % COLORS.length] }} />}
                      {displayBank(d.bank_name)}
                    </td>
                    <td className="py-3 text-right font-medium text-gray-900 dark:text-white">{fmtM(toM(d.yhat))}</td>
                    <td className="py-3 text-right text-gray-500 dark:text-slate-400">{d.yhat_lower && d.yhat_upper ? `${fmtM(toM(d.yhat_lower))} – ${fmtM(toM(d.yhat_upper))}` : "—"}</td>
                    <td className={`py-3 text-right font-medium ${d.manufacture !== null ? (d.manufacture >= 0 ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400") : "text-gray-400 dark:text-slate-500"}`}>
                      {d.manufacture !== null ? (d.manufacture >= 0 ? "+" : "") + fmtM(toM(d.manufacture), 2) : "—"}
                    </td>
                    <td className="py-3 text-right text-gray-400 dark:text-slate-500">{d.model_type || "Prophet"}</td>
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
