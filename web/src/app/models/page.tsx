"use client";

import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";
import type { ModelMetadata } from "@/lib/types";

function accuracyFromMape(mape: number | null): number | null {
  if (mape == null) return null;
  return Math.max(0, 100 - mape);
}

function accuracyColor(acc: number | null): string {
  if (acc == null) return "";
  if (acc >= 93) return "text-emerald-700 bg-emerald-100 dark:text-emerald-400 dark:bg-emerald-900/30";
  if (acc >= 85) return "text-amber-700 bg-amber-100 dark:text-amber-400 dark:bg-amber-900/30";
  return "text-red-700 bg-red-100 dark:text-red-400 dark:bg-red-900/30";
}

function accuracyBadge(mape: number | null): string {
  const acc = accuracyFromMape(mape);
  if (acc == null) return "—";
  return acc.toFixed(1) + "% accurate";
}

function median(arr: number[]): number {
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

const ALLOWED_CC = new Set([
  "HDFC Bank", "State Bank of India", "ICICI Bank", "Axis Bank",
  "Kotak Mahindra Bank", "IndusInd Bank", "Bank of Baroda",
  "Yes Bank", "Canara Bank", "HSBC",
]);
const ALLOWED_DC = new Set([
  "State Bank of India", "Bank of Baroda", "Canara Bank", "HDFC Bank",
  "Union Bank of India", "Punjab National Bank", "Axis Bank",
  "Bank of India", "Kotak Mahindra Bank", "Indian Bank",
  "Central Bank of India", "UCO Bank", "ICICI Bank",
  "Indian Overseas Bank", "Paytm Payments Bank",
]);

export default function ModelPerformancePage() {
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<"all" | "CC" | "DC">("all");

  useEffect(() => {
    async function load() {
      const { data, error: err } = await supabase
        .from("model_metadata")
        .select("*")
        .order("bank_name", { ascending: true });
      if (err) { setError(err.message); setLoading(false); return; }
      if (data) setModels(data);
      setLoading(false);
    }
    load();
  }, []);

  if (loading) return <div className="flex items-center justify-center h-64"><div className="text-gray-400 dark:text-slate-500">Loading...</div></div>;

  if (error) return (
    <div className="bg-red-100 border border-red-300 dark:bg-red-900/30 dark:border-red-700/50 rounded-lg p-6 text-center my-8">
      <p className="text-red-700 dark:text-red-400 font-medium">Failed to load model data</p>
      <p className="text-red-500 text-sm mt-1">{error}</p>
    </div>
  );

  const filtered = filter === "all" ? models : models.filter((m) => m.card_type === filter);
  const bankModels = filtered.filter((m) => {
    if (!m.bank_name) return false;
    if (m.card_type === "CC") return ALLOWED_CC.has(m.bank_name);
    if (m.card_type === "DC") return ALLOWED_DC.has(m.bank_name);
    return false;
  });
  const aggModels = filtered.filter((m) => !m.bank_name && m.metric);

  const ccBankMapes = models.filter((m) => m.bank_name && m.card_type === "CC" && m.cv_mape != null && ALLOWED_CC.has(m.bank_name)).map((m) => m.cv_mape!);
  const dcBankMapes = models.filter((m) => m.bank_name && m.card_type === "DC" && m.cv_mape != null && ALLOWED_DC.has(m.bank_name)).map((m) => m.cv_mape!);
  const ccMedianAcc = ccBankMapes.length > 0 ? (100 - median(ccBankMapes)).toFixed(1) : null;
  const dcMedianAcc = dcBankMapes.length > 0 ? (100 - median(dcBankMapes)).toFixed(1) : null;
  const allMapes = [...ccBankMapes, ...dcBankMapes];
  const overallAcc = allMapes.length > 0 ? (100 - median(allMapes)).toFixed(1) : null;

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Model Performance</h1>
          <p className="text-sm text-gray-400 dark:text-slate-500 mt-0.5">Accuracy = 100% minus forecast error (MAPE)</p>
        </div>
        <div className="flex rounded-lg border border-gray-300 dark:border-slate-600 overflow-hidden">
          {(["all", "CC", "DC"] as const).map((f) => (
            <button key={f} onClick={() => setFilter(f)}
              className={`px-4 py-2 text-sm font-medium transition-colors ${filter === f ? "bg-blue-600 text-white" : "bg-white dark:bg-slate-800 text-gray-500 dark:text-slate-400 hover:bg-gray-200 dark:hover:bg-slate-700"}`}>
              {f === "all" ? "All" : f === "CC" ? "Credit" : "Debit"}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-5 text-center">
          <p className="text-sm text-gray-500 dark:text-slate-400 mb-1">Overall Median Accuracy</p>
          <p className="text-3xl font-bold text-emerald-400">{overallAcc ? overallAcc + "%" : "—"}</p>
          <p className="text-xs text-gray-400 dark:text-slate-500 mt-1">across all bank models</p>
        </div>
        <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-5 text-center">
          <p className="text-sm text-gray-500 dark:text-slate-400 mb-1">Credit Card Models</p>
          <p className="text-3xl font-bold text-blue-400">{ccMedianAcc ? ccMedianAcc + "%" : "—"}</p>
          <p className="text-xs text-gray-400 dark:text-slate-500 mt-1">median accuracy (10 banks)</p>
        </div>
        <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-5 text-center">
          <p className="text-sm text-gray-500 dark:text-slate-400 mb-1">Debit Card Models</p>
          <p className="text-3xl font-bold text-blue-400">{dcMedianAcc ? dcMedianAcc + "%" : "—"}</p>
          <p className="text-xs text-gray-400 dark:text-slate-500 mt-1">median accuracy (15 banks)</p>
        </div>
      </div>

      {bankModels.length > 0 && (
        <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-6 mb-6">
          <h3 className="text-sm font-semibold text-gray-800 dark:text-slate-200 mb-4">Bank-Level Models</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-slate-700 text-left text-gray-500 dark:text-slate-400">
                  <th className="pb-3 font-medium">Bank</th>
                  <th className="pb-3 font-medium">Type</th>
                  <th className="pb-3 font-medium">Model</th>
                  <th className="pb-3 text-right font-medium">Accuracy (CV)</th>
                  <th className="pb-3 text-right font-medium">Accuracy (OOS)</th>
                  <th className="pb-3 text-right font-medium">Last Trained</th>
                </tr>
              </thead>
              <tbody>
                {bankModels.map((m) => (
                  <tr key={`${m.bank_name}-${m.card_type}`} className="border-b border-gray-200 dark:border-slate-700/50 last:border-0 hover:bg-gray-50 dark:hover:bg-slate-700/30">
                    <td className="py-3 font-medium text-gray-800 dark:text-slate-200">{m.bank_name}</td>
                    <td className="py-3 text-gray-500 dark:text-slate-400">{m.card_type === "CC" ? "Credit" : "Debit"}</td>
                    <td className="py-3 text-gray-400 dark:text-slate-500">{m.model_type}</td>
                    <td className="py-3 text-right"><span className={`px-2 py-0.5 rounded text-xs font-medium ${accuracyColor(accuracyFromMape(m.cv_mape))}`}>{accuracyBadge(m.cv_mape)}</span></td>
                    <td className="py-3 text-right"><span className={`px-2 py-0.5 rounded text-xs font-medium ${accuracyColor(accuracyFromMape(m.oos_mape))}`}>{accuracyBadge(m.oos_mape)}</span></td>
                    <td className="py-3 text-right text-gray-400 dark:text-slate-500">{m.last_trained ? new Date(m.last_trained).toLocaleDateString("en-IN") : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {aggModels.length > 0 && (
        <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-6">
          <h3 className="text-sm font-semibold text-gray-800 dark:text-slate-200 mb-4">Aggregate Models</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-slate-700 text-left text-gray-500 dark:text-slate-400">
                  <th className="pb-3 font-medium">Metric</th>
                  <th className="pb-3 font-medium">Model</th>
                  <th className="pb-3 text-right font-medium">Accuracy (CV)</th>
                  <th className="pb-3 text-right font-medium">Last Trained</th>
                </tr>
              </thead>
              <tbody>
                {aggModels.map((m) => (
                  <tr key={m.metric} className="border-b border-gray-200 dark:border-slate-700/50 last:border-0 hover:bg-gray-50 dark:hover:bg-slate-700/30">
                    <td className="py-3 font-medium text-gray-800 dark:text-slate-200">{m.metric?.replace(/_/g, " ")}</td>
                    <td className="py-3 text-gray-400 dark:text-slate-500">{m.model_type}</td>
                    <td className="py-3 text-right"><span className={`px-2 py-0.5 rounded text-xs font-medium ${accuracyColor(accuracyFromMape(m.cv_mape))}`}>{accuracyBadge(m.cv_mape)}</span></td>
                    <td className="py-3 text-right text-gray-400 dark:text-slate-500">{m.last_trained ? new Date(m.last_trained).toLocaleDateString("en-IN") : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {models.length === 0 && (
        <p className="text-gray-400 dark:text-slate-500 text-center py-8">No model metadata available yet.</p>
      )}
    </div>
  );
}
