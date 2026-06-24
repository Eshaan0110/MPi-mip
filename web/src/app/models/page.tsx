"use client";

import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";
import type { ModelMetadata } from "@/lib/types";

function mapeColor(mape: number | null): string {
  if (mape == null) return "";
  if (mape <= 7) return "text-green-700 bg-green-50";
  if (mape <= 15) return "text-amber-700 bg-amber-50";
  return "text-red-700 bg-red-50";
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

  if (loading) {
    return <div className="flex items-center justify-center h-64"><div className="text-gray-400">Loading...</div></div>;
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center my-8">
        <p className="text-red-700 font-medium">Failed to load model data</p>
        <p className="text-red-600 text-sm mt-1">{error}</p>
      </div>
    );
  }

  const filtered = filter === "all" ? models : models.filter((m) => m.card_type === filter);
  const bankModels = filtered.filter((m) => {
    if (!m.bank_name) return false;
    if (m.card_type === "CC") return ALLOWED_CC.has(m.bank_name);
    if (m.card_type === "DC") return ALLOWED_DC.has(m.bank_name);
    return false;
  });
  const aggModels = filtered.filter((m) => !m.bank_name && m.metric);

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-brand-700">Model Performance</h1>
        <div className="flex rounded-md border border-gray-300 overflow-hidden">
          {(["all", "CC", "DC"] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1.5 text-sm font-medium ${
                filter === f ? "bg-brand-500 text-white" : "bg-white text-gray-600 hover:bg-gray-50"
              }`}
            >
              {f === "all" ? "All" : f === "CC" ? "Credit" : "Debit"}
            </button>
          ))}
        </div>
      </div>

      {bankModels.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-5 shadow-sm mb-6">
          <h3 className="text-sm font-semibold text-gray-700 mb-4">Bank-Level Models</h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-gray-500">
                <th className="pb-2">Bank</th>
                <th className="pb-2">Type</th>
                <th className="pb-2">Model</th>
                <th className="pb-2 text-right">CV MAPE</th>
                <th className="pb-2 text-right">OOS MAPE</th>
                <th className="pb-2 text-right">Last Trained</th>
              </tr>
            </thead>
            <tbody>
              {bankModels.map((m) => (
                <tr key={`${m.bank_name}-${m.card_type}`} className="border-b last:border-0">
                  <td className="py-2 font-medium">{m.bank_name}</td>
                  <td className="py-2">{m.card_type}</td>
                  <td className="py-2 text-gray-500">{m.model_type}</td>
                  <td className="py-2 text-right"><span className={`px-2 py-0.5 rounded text-xs font-medium ${mapeColor(m.cv_mape)}`}>{m.cv_mape != null ? m.cv_mape.toFixed(1) + "%" : "—"}</span></td>
                  <td className="py-2 text-right"><span className={`px-2 py-0.5 rounded text-xs font-medium ${mapeColor(m.oos_mape)}`}>{m.oos_mape != null ? m.oos_mape.toFixed(1) + "%" : "—"}</span></td>
                  <td className="py-2 text-right text-gray-400">
                    {m.last_trained ? new Date(m.last_trained).toLocaleDateString("en-IN") : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {aggModels.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-5 shadow-sm">
          <h3 className="text-sm font-semibold text-gray-700 mb-4">Aggregate Models</h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-gray-500">
                <th className="pb-2">Metric</th>
                <th className="pb-2">Model</th>
                <th className="pb-2 text-right">CV MAPE</th>
                <th className="pb-2 text-right">Last Trained</th>
              </tr>
            </thead>
            <tbody>
              {aggModels.map((m) => (
                <tr key={m.metric} className="border-b last:border-0">
                  <td className="py-2 font-medium">{m.metric?.replace(/_/g, " ")}</td>
                  <td className="py-2 text-gray-500">{m.model_type}</td>
                  <td className="py-2 text-right">{m.cv_mape != null ? m.cv_mape.toFixed(1) + "%" : "—"}</td>
                  <td className="py-2 text-right text-gray-400">
                    {m.last_trained ? new Date(m.last_trained).toLocaleDateString("en-IN") : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {models.length === 0 && (
        <p className="text-gray-400 text-center py-8">No model metadata available yet.</p>
      )}
    </div>
  );
}
