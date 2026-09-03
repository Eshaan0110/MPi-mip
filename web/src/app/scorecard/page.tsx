"use client";

import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";
import type { ScorecardScore } from "@/lib/types";

interface ModelSummary {
  model: string;
  label: string;
  type: "aggregate" | "bank";
  cardType: "CC" | "DC" | null;
  rollingMape: number;
  rollingAccuracy: number;
  nScored: number;
  months: ScorecardScore[];
}

function parseModelName(name: string): { label: string; type: "aggregate" | "bank"; cardType: "CC" | "DC" | null } {
  if (name === "cc_outstanding") return { label: "CC Outstanding (Aggregate)", type: "aggregate", cardType: "CC" };
  if (name === "dc_outstanding") return { label: "DC Outstanding (Aggregate)", type: "aggregate", cardType: "DC" };
  if (name.startsWith("cc_")) return { label: name.replace("cc_", "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()) + " (CC)", type: "bank", cardType: "CC" };
  if (name.startsWith("dc_")) return { label: name.replace("dc_", "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()) + " (DC)", type: "bank", cardType: "DC" };
  return { label: name.replace(/_/g, " "), type: "aggregate", cardType: null };
}

function accuracyColor(acc: number): string {
  if (acc >= 95) return "text-emerald-700 bg-emerald-100 dark:text-emerald-400 dark:bg-emerald-900/30";
  if (acc >= 90) return "text-emerald-600 bg-emerald-50 dark:text-emerald-400 dark:bg-emerald-900/20";
  if (acc >= 85) return "text-amber-700 bg-amber-100 dark:text-amber-400 dark:bg-amber-900/30";
  return "text-red-700 bg-red-100 dark:text-red-400 dark:bg-red-900/30";
}

function apeColor(ape: number): string {
  if (ape <= 3) return "text-emerald-600 dark:text-emerald-400";
  if (ape <= 7) return "text-amber-600 dark:text-amber-400";
  return "text-red-600 dark:text-red-400";
}

export default function ScorecardPage() {
  const [scores, setScores] = useState<ScorecardScore[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<"all" | "aggregate" | "bank">("all");
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      const { data, error: err } = await supabase
        .from("scorecard_scores")
        .select("*")
        .order("forecast_month", { ascending: false });
      if (err) { setError(err.message); setLoading(false); return; }
      if (data) setScores(data);
      setLoading(false);
    }
    load();
  }, []);

  if (loading) return <div className="flex items-center justify-center h-64"><div className="text-gray-400 dark:text-slate-500">Loading scorecard...</div></div>;
  if (error) return (
    <div className="bg-red-100 border border-red-300 dark:bg-red-900/30 dark:border-red-700/50 rounded-lg p-6 text-center my-8">
      <p className="text-red-700 dark:text-red-400 font-medium">Failed to load scorecard</p>
      <p className="text-red-500 text-sm mt-1">{error}</p>
    </div>
  );

  if (scores.length === 0) return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">Live Scorecard</h1>
      <p className="text-sm text-gray-400 dark:text-slate-500 mb-8">Audited accuracy: forecasts vs first-release actuals</p>
      <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-12 text-center">
        <p className="text-gray-400 dark:text-slate-500">No scorecard data yet. Scores appear after the pipeline runs and actuals land.</p>
      </div>
    </div>
  );

  const byModel = new Map<string, ScorecardScore[]>();
  for (const s of scores) {
    const arr = byModel.get(s.model_name) || [];
    arr.push(s);
    byModel.set(s.model_name, arr);
  }

  const summaries: ModelSummary[] = Array.from(byModel.entries()).map(([model, months]) => {
    const sorted = [...months].sort((a, b) => b.forecast_month.localeCompare(a.forecast_month));
    const rolling = sorted.slice(0, 12);
    const mape = rolling.reduce((sum, m) => sum + m.ape, 0) / rolling.length;
    const { label, type, cardType } = parseModelName(model);
    return {
      model,
      label,
      type,
      cardType,
      rollingMape: Math.round(mape * 100) / 100,
      rollingAccuracy: Math.round((100 - mape) * 100) / 100,
      nScored: months.length,
      months: sorted,
    };
  }).sort((a, b) => a.rollingMape - b.rollingMape);

  const filtered = filter === "all" ? summaries : summaries.filter(s => s.type === filter);

  const aggSummaries = summaries.filter(s => s.type === "aggregate");
  const bankSummaries = summaries.filter(s => s.type === "bank");
  const overallMape = summaries.length > 0 ? summaries.reduce((s, m) => s + m.rollingMape, 0) / summaries.length : 0;
  const aggMape = aggSummaries.length > 0 ? aggSummaries.reduce((s, m) => s + m.rollingMape, 0) / aggSummaries.length : 0;
  const bankMape = bankSummaries.length > 0 ? bankSummaries.reduce((s, m) => s + m.rollingMape, 0) / bankSummaries.length : 0;

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Live Scorecard</h1>
          <p className="text-sm text-gray-400 dark:text-slate-500 mt-0.5">Audited accuracy: forecasts scored against first-release actuals</p>
        </div>
        <div className="flex rounded-lg border border-gray-300 dark:border-slate-600 overflow-hidden">
          {(["all", "aggregate", "bank"] as const).map((f) => (
            <button key={f} onClick={() => setFilter(f)}
              className={`px-4 py-2 text-sm font-medium transition-colors ${filter === f ? "bg-blue-600 text-white" : "bg-white dark:bg-slate-800 text-gray-500 dark:text-slate-400 hover:bg-gray-200 dark:hover:bg-slate-700"}`}>
              {f === "all" ? "All" : f === "aggregate" ? "Aggregate" : "Bank-Level"}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-5 text-center">
          <p className="text-sm text-gray-500 dark:text-slate-400 mb-1">Overall Audited Accuracy</p>
          <p className="text-3xl font-bold text-emerald-400">{(100 - overallMape).toFixed(1)}%</p>
          <p className="text-xs text-gray-400 dark:text-slate-500 mt-1">rolling 12-month average</p>
        </div>
        <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-5 text-center">
          <p className="text-sm text-gray-500 dark:text-slate-400 mb-1">Aggregate Models</p>
          <p className="text-3xl font-bold text-blue-400">{aggSummaries.length > 0 ? (100 - aggMape).toFixed(1) + "%" : "—"}</p>
          <p className="text-xs text-gray-400 dark:text-slate-500 mt-1">{aggSummaries.length} models scored</p>
        </div>
        <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-5 text-center">
          <p className="text-sm text-gray-500 dark:text-slate-400 mb-1">Bank-Level Models</p>
          <p className="text-3xl font-bold text-blue-400">{bankSummaries.length > 0 ? (100 - bankMape).toFixed(1) + "%" : "—"}</p>
          <p className="text-xs text-gray-400 dark:text-slate-500 mt-1">{bankSummaries.length} models scored</p>
        </div>
      </div>

      <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-6">
        <h3 className="text-sm font-semibold text-gray-800 dark:text-slate-200 mb-4">Model Scorecard</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-slate-700 text-left text-gray-500 dark:text-slate-400">
                <th className="pb-3 font-medium">Model</th>
                <th className="pb-3 font-medium">Type</th>
                <th className="pb-3 text-right font-medium">Rolling Accuracy</th>
                <th className="pb-3 text-right font-medium">Rolling MAPE</th>
                <th className="pb-3 text-right font-medium">Months Scored</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((s) => (
                <>
                  <tr key={s.model}
                    className="border-b border-gray-200 dark:border-slate-700/50 last:border-0 hover:bg-gray-50 dark:hover:bg-slate-700/30 cursor-pointer"
                    onClick={() => setExpanded(expanded === s.model ? null : s.model)}>
                    <td className="py-3 font-medium text-gray-800 dark:text-slate-200">{s.label}</td>
                    <td className="py-3 text-gray-400 dark:text-slate-500">{s.type === "aggregate" ? "Aggregate" : "Bank"}</td>
                    <td className="py-3 text-right">
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${accuracyColor(s.rollingAccuracy)}`}>
                        {s.rollingAccuracy.toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-3 text-right text-gray-600 dark:text-slate-300">{s.rollingMape.toFixed(2)}%</td>
                    <td className="py-3 text-right text-gray-400 dark:text-slate-500">{s.nScored}</td>
                  </tr>
                  {expanded === s.model && (
                    <tr key={s.model + "-detail"}>
                      <td colSpan={5} className="pb-4 pt-0">
                        <div className="bg-gray-50 dark:bg-slate-700/20 rounded-lg p-4 ml-4">
                          <table className="w-full text-xs">
                            <thead>
                              <tr className="text-gray-400 dark:text-slate-500 text-left">
                                <th className="pb-2 font-medium">Month</th>
                                <th className="pb-2 text-right font-medium">Forecast</th>
                                <th className="pb-2 text-right font-medium">Actual</th>
                                <th className="pb-2 text-right font-medium">APE</th>
                              </tr>
                            </thead>
                            <tbody>
                              {s.months.slice(0, 12).map((m) => (
                                <tr key={m.forecast_month} className="border-t border-gray-200 dark:border-slate-600/30">
                                  <td className="py-1.5 text-gray-600 dark:text-slate-300">
                                    {new Date(m.forecast_month).toLocaleDateString("en-IN", { month: "short", year: "numeric" })}
                                  </td>
                                  <td className="py-1.5 text-right text-gray-600 dark:text-slate-300">{m.forecast_value.toLocaleString("en-IN", { maximumFractionDigits: 1 })}</td>
                                  <td className="py-1.5 text-right text-gray-600 dark:text-slate-300">{m.actual_value.toLocaleString("en-IN", { maximumFractionDigits: 1 })}</td>
                                  <td className={`py-1.5 text-right font-medium ${apeColor(m.ape)}`}>{m.ape.toFixed(2)}%</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </td>
                    </tr>
                  )}
                </>
              ))}
            </tbody>
          </table>
        </div>
        {filtered.length === 0 && (
          <p className="text-gray-400 dark:text-slate-500 text-center py-4">No models match this filter.</p>
        )}
      </div>
    </div>
  );
}
