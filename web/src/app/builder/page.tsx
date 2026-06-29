"use client";

import { useEffect, useState, useMemo } from "react";
import { supabase } from "@/lib/supabase";
import { ForecastChart, COLORS } from "@/components/ForecastChart";
import type { AggregateForecast, BankForecast } from "@/lib/types";

function toM(v: number): number { return v / 10; }
function toMBank(v: number): number { return v / 1_000_000; }
function fmtM(n: number, decimals = 1): string { return n.toFixed(decimals) + " M"; }

function fmtMonth(m: string): string {
  const d = new Date(m.length === 7 ? m + "-01" : m);
  if (isNaN(d.getTime())) return m;
  return d.toLocaleDateString("en-IN", { month: "short", year: "numeric" });
}

function fmtMonthInput(m: string): string {
  return m.length >= 7 ? m.substring(0, 7) : m;
}

const ALLOWED_CC_BANKS = new Set([
  "HDFC Bank", "State Bank of India", "ICICI Bank", "Axis Bank",
  "Kotak Mahindra Bank", "IndusInd Bank", "Bank of Baroda",
  "Yes Bank", "Canara Bank", "HSBC", "_RESIDUAL",
]);
const ALLOWED_DC_BANKS = new Set([
  "State Bank of India", "Bank of Baroda", "Canara Bank", "HDFC Bank",
  "Union Bank of India", "Punjab National Bank", "Axis Bank",
  "Bank of India", "Kotak Mahindra Bank", "Indian Bank",
  "Central Bank of India", "UCO Bank", "ICICI Bank",
  "Indian Overseas Bank", "Paytm Payments Bank", "_RESIDUAL",
]);

function displayBank(name: string): string {
  return name === "_RESIDUAL" ? "All Other Banks (Residual)" : name;
}

type ViewLevel = "aggregate" | "bank";

export default function BuilderPage() {
  const [aggData, setAggData] = useState<AggregateForecast[]>([]);
  const [bankData, setBankData] = useState<BankForecast[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [cardType, setCardType] = useState<"CC" | "DC">("CC");
  const [viewLevel, setViewLevel] = useState<ViewLevel>("aggregate");
  const [fromMonth, setFromMonth] = useState("");
  const [toMonth, setToMonth] = useState("");
  const [selectedBanks, setSelectedBanks] = useState<string[]>([]);

  useEffect(() => {
    async function load() {
      const [aggRes, bankRes] = await Promise.all([
        supabase.from("forecasts_aggregate").select("*").order("forecast_month", { ascending: true }),
        supabase.from("forecasts_bank").select("*").order("forecast_month", { ascending: true }),
      ]);
      if (aggRes.error) { setError(aggRes.error.message); setLoading(false); return; }
      if (bankRes.error) { setError(bankRes.error.message); setLoading(false); return; }
      setAggData(aggRes.data || []);
      setBankData(bankRes.data || []);

      const aggMonths = [...new Set((aggRes.data || []).map((d) => d.forecast_month))].sort();
      if (aggMonths.length > 0) {
        setFromMonth(aggMonths[0]);
        setToMonth(aggMonths[aggMonths.length - 1]);
      }
      setLoading(false);
    }
    load();
  }, []);

  const aggMonths = useMemo(() => [...new Set(aggData.map((d) => d.forecast_month))].sort(), [aggData]);
  const bankMonths = useMemo(() => [...new Set(bankData.map((d) => d.forecast_month))].sort(), [bankData]);
  const allMonths = viewLevel === "aggregate" ? aggMonths : bankMonths;

  const allowedBanks = cardType === "CC" ? ALLOWED_CC_BANKS : ALLOWED_DC_BANKS;
  const banksForType = useMemo(() =>
    [...new Set(bankData.filter((f) => f.card_type === cardType).map((f) => f.bank_name))]
      .filter((b) => allowedBanks.has(b))
      .sort(),
    [bankData, cardType, allowedBanks]
  );

  const metric = cardType === "CC" ? "cc_outstanding" : "dc_outstanding";
  const RAW_TO_LAKH = 1 / 100_000;

  const filteredAggData = useMemo(() =>
    aggData
      .filter((f) => f.metric === metric && f.forecast_month >= fromMonth && f.forecast_month <= toMonth)
      .map((f) => ({
        month: f.forecast_month,
        forecast: f.yhat,
        lower: f.yhat_lower ?? undefined,
        upper: f.yhat_upper ?? undefined,
      })),
    [aggData, metric, fromMonth, toMonth]
  );

  const filteredBankChartData = useMemo(() => {
    if (selectedBanks.length === 0) return { multiData: [], multiLines: [] };
    const months = [...new Set(
      bankData
        .filter((f) => f.card_type === cardType && selectedBanks.includes(f.bank_name) && f.forecast_month >= fromMonth && f.forecast_month <= toMonth)
        .map((f) => f.forecast_month)
    )].sort();

    const multiData = months.map((m) => {
      const row: any = { month: m };
      for (const bank of selectedBanks) {
        const rec = bankData.find((f) => f.bank_name === bank && f.card_type === cardType && f.forecast_month === m);
        row[bank] = rec ? rec.yhat * RAW_TO_LAKH : undefined;
      }
      return row;
    });
    const multiLines = selectedBanks.map((bank, i) => ({
      key: bank,
      label: displayBank(bank),
      color: COLORS[i % COLORS.length],
    }));
    return { multiData, multiLines };
  }, [bankData, cardType, selectedBanks, fromMonth, toMonth]);

  const tableData = useMemo(() => {
    if (viewLevel === "aggregate") {
      return aggData
        .filter((f) => f.metric === metric && f.forecast_month >= fromMonth && f.forecast_month <= toMonth)
        .map((f) => ({
          month: f.forecast_month,
          value: toM(f.yhat),
          lower: f.yhat_lower ? toM(f.yhat_lower) : null,
          upper: f.yhat_upper ? toM(f.yhat_upper) : null,
          label: cardType === "CC" ? "CC Outstanding" : "DC Outstanding",
        }));
    }
    if (selectedBanks.length === 0) return [];
    return bankData
      .filter((f) => f.card_type === cardType && selectedBanks.includes(f.bank_name) && f.forecast_month >= fromMonth && f.forecast_month <= toMonth)
      .sort((a, b) => a.forecast_month.localeCompare(b.forecast_month) || a.bank_name.localeCompare(b.bank_name))
      .map((f) => ({
        month: f.forecast_month,
        value: toMBank(f.yhat),
        lower: f.yhat_lower ? toMBank(f.yhat_lower) : null,
        upper: f.yhat_upper ? toMBank(f.yhat_upper) : null,
        label: displayBank(f.bank_name),
      }));
  }, [aggData, bankData, viewLevel, metric, cardType, selectedBanks, fromMonth, toMonth]);

  const toggleBank = (bank: string) => {
    setSelectedBanks((prev) =>
      prev.includes(bank) ? prev.filter((b) => b !== bank) : [...prev, bank]
    );
  };

  if (loading) return <div className="flex items-center justify-center h-64"><div className="text-gray-400 dark:text-slate-500">Loading data...</div></div>;
  if (error) return <div className="bg-red-100 border border-red-300 dark:bg-red-900/30 dark:border-red-700/50 rounded-lg p-6 text-center my-8"><p className="text-red-700 dark:text-red-400 font-medium">Failed to load data</p><p className="text-red-500 text-sm mt-1">{error}</p></div>;

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Forecast Builder</h1>
        <p className="text-sm text-gray-400 dark:text-slate-500 mt-0.5">Select card type, date range, and level to build a custom forecast view</p>
      </div>

      {/* Controls */}
      <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-5 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Card Type */}
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-slate-400 mb-1.5">Card type</label>
            <div className="flex rounded-lg border border-gray-300 dark:border-slate-600 overflow-hidden">
              {(["CC", "DC"] as const).map((ct) => (
                <button
                  key={ct}
                  onClick={() => { setCardType(ct); setSelectedBanks([]); }}
                  className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
                    cardType === ct ? "bg-blue-600 text-white" : "bg-white dark:bg-slate-800 text-gray-500 dark:text-slate-400 hover:bg-gray-200 dark:hover:bg-slate-700"
                  }`}
                >
                  {ct === "CC" ? "Credit Card" : "Debit Card"}
                </button>
              ))}
            </div>
          </div>

          {/* View Level */}
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-slate-400 mb-1.5">Level</label>
            <div className="flex rounded-lg border border-gray-300 dark:border-slate-600 overflow-hidden">
              {(["aggregate", "bank"] as const).map((lv) => (
                <button
                  key={lv}
                  onClick={() => { setViewLevel(lv); if (lv === "aggregate") setSelectedBanks([]); }}
                  className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
                    viewLevel === lv ? "bg-blue-600 text-white" : "bg-white dark:bg-slate-800 text-gray-500 dark:text-slate-400 hover:bg-gray-200 dark:hover:bg-slate-700"
                  }`}
                >
                  {lv === "aggregate" ? "Aggregate" : "Bank-level"}
                </button>
              ))}
            </div>
          </div>

          {/* From Date */}
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-slate-400 mb-1.5">From</label>
            <input
              type="month"
              value={fmtMonthInput(fromMonth)}
              onChange={(e) => setFromMonth(e.target.value)}
              min={allMonths.length > 0 ? fmtMonthInput(allMonths[0]) : undefined}
              max={fmtMonthInput(toMonth)}
              className="w-full border border-gray-300 dark:border-slate-600 rounded-lg px-3 py-2 text-sm bg-white dark:bg-slate-800 text-gray-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* To Date */}
          <div>
            <label className="block text-xs font-medium text-gray-500 dark:text-slate-400 mb-1.5">To</label>
            <input
              type="month"
              value={fmtMonthInput(toMonth)}
              onChange={(e) => setToMonth(e.target.value)}
              min={fmtMonthInput(fromMonth)}
              max={allMonths.length > 0 ? fmtMonthInput(allMonths[allMonths.length - 1]) : undefined}
              className="w-full border border-gray-300 dark:border-slate-600 rounded-lg px-3 py-2 text-sm bg-white dark:bg-slate-800 text-gray-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        {/* Bank Selector */}
        {viewLevel === "bank" && (
          <div className="mt-4 pt-4 border-t border-gray-200 dark:border-slate-700/50">
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-medium text-gray-500 dark:text-slate-400">Select banks to compare</label>
              <div className="flex gap-2">
                <button onClick={() => setSelectedBanks([])} className="text-xs px-2.5 py-1 rounded-md bg-gray-200 dark:bg-slate-700 text-gray-500 dark:text-slate-400 hover:bg-gray-300 dark:hover:bg-slate-600">Clear</button>
                <button onClick={() => setSelectedBanks(banksForType.slice(0, 5))} className="text-xs px-2.5 py-1 rounded-md bg-gray-200 dark:bg-slate-700 text-gray-500 dark:text-slate-400 hover:bg-gray-300 dark:hover:bg-slate-600">Top 5</button>
                <button onClick={() => setSelectedBanks([...banksForType])} className="text-xs px-2.5 py-1 rounded-md bg-gray-200 dark:bg-slate-700 text-gray-500 dark:text-slate-400 hover:bg-gray-300 dark:hover:bg-slate-600">All</button>
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
                        : "bg-white dark:bg-slate-800 text-gray-500 dark:text-slate-400 border-gray-300 dark:border-slate-600 hover:border-gray-400 dark:hover:border-slate-400"
                    }`}
                    style={isSelected ? { backgroundColor: COLORS[colorIdx % COLORS.length] } : {}}
                  >
                    {displayBank(bank)}
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white dark:bg-slate-800/50 rounded-lg border border-gray-200 dark:border-slate-700/50 p-4">
          <p className="text-xs text-gray-400 dark:text-slate-500">Card type</p>
          <p className="text-lg font-bold text-gray-900 dark:text-white mt-0.5">{cardType === "CC" ? "Credit Card" : "Debit Card"}</p>
        </div>
        <div className="bg-white dark:bg-slate-800/50 rounded-lg border border-gray-200 dark:border-slate-700/50 p-4">
          <p className="text-xs text-gray-400 dark:text-slate-500">Level</p>
          <p className="text-lg font-bold text-gray-900 dark:text-white mt-0.5">{viewLevel === "aggregate" ? "India Aggregate" : `${selectedBanks.length} Bank${selectedBanks.length !== 1 ? "s" : ""}`}</p>
        </div>
        <div className="bg-white dark:bg-slate-800/50 rounded-lg border border-gray-200 dark:border-slate-700/50 p-4">
          <p className="text-xs text-gray-400 dark:text-slate-500">From</p>
          <p className="text-lg font-bold text-gray-900 dark:text-white mt-0.5">{fmtMonth(fromMonth)}</p>
        </div>
        <div className="bg-white dark:bg-slate-800/50 rounded-lg border border-gray-200 dark:border-slate-700/50 p-4">
          <p className="text-xs text-gray-400 dark:text-slate-500">To</p>
          <p className="text-lg font-bold text-gray-900 dark:text-white mt-0.5">{fmtMonth(toMonth)}</p>
        </div>
      </div>

      {/* Chart */}
      <div className="mb-6">
        {viewLevel === "aggregate" ? (
          <ForecastChart
            data={filteredAggData}
            title={`${cardType === "CC" ? "Credit Card" : "Debit Card"} Outstanding — ${fmtMonth(fromMonth)} to ${fmtMonth(toMonth)}`}
          />
        ) : selectedBanks.length > 0 ? (
          <ForecastChart
            data={[]}
            title={`Bank Comparison — ${cardType === "CC" ? "Credit Card" : "Debit Card"} — ${fmtMonth(fromMonth)} to ${fmtMonth(toMonth)}`}
            multiLines={filteredBankChartData.multiLines}
            multiData={filteredBankChartData.multiData}
          />
        ) : (
          <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-12 text-center">
            <p className="text-gray-400 dark:text-slate-500">Select at least one bank above to see the chart</p>
          </div>
        )}
      </div>

      {/* Data Table */}
      {tableData.length > 0 && (
        <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-semibold text-gray-800 dark:text-slate-200">Forecast Data</h3>
              <p className="text-xs text-gray-400 dark:text-slate-500 mt-0.5">{tableData.length} rows — Values in Millions</p>
            </div>
          </div>
          <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-white dark:bg-slate-800">
                <tr className="border-b border-gray-200 dark:border-slate-700 text-left text-gray-500 dark:text-slate-400">
                  <th className="pb-3 pr-4 font-medium">Month</th>
                  {viewLevel === "bank" && <th className="pb-3 pr-4 font-medium">Bank</th>}
                  <th className="pb-3 text-right font-medium">Forecast</th>
                  <th className="pb-3 text-right font-medium">Lower 90%</th>
                  <th className="pb-3 text-right font-medium">Upper 90%</th>
                </tr>
              </thead>
              <tbody>
                {tableData.map((d, i) => (
                  <tr key={i} className="border-b border-gray-200 dark:border-slate-700/50 last:border-0 hover:bg-gray-50 dark:hover:bg-slate-700/30">
                    <td className="py-2.5 pr-4 text-gray-700 dark:text-slate-300">{fmtMonth(d.month)}</td>
                    {viewLevel === "bank" && <td className="py-2.5 pr-4 text-gray-700 dark:text-slate-300 font-medium">{d.label}</td>}
                    <td className="py-2.5 text-right font-medium text-gray-900 dark:text-white">{fmtM(d.value)}</td>
                    <td className="py-2.5 text-right text-gray-500 dark:text-slate-400">{d.lower ? fmtM(d.lower) : "—"}</td>
                    <td className="py-2.5 text-right text-gray-500 dark:text-slate-400">{d.upper ? fmtM(d.upper) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
