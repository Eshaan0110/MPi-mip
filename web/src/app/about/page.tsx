"use client";

import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";
import type { ModelMetadata } from "@/lib/types";

function median(arr: number[]): number {
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

export default function AboutPage() {
  const [ccMape, setCcMape] = useState<string>("—");
  const [dcMape, setDcMape] = useState<string>("—");
  const [ccAggMape, setCcAggMape] = useState<string>("—");
  const [dcAggMape, setDcAggMape] = useState<string>("—");

  useEffect(() => {
    async function load() {
      const { data } = await supabase.from("model_metadata").select("*");
      if (!data) return;
      const ccBanks = data.filter((m: ModelMetadata) => m.bank_name && m.card_type === "CC" && m.cv_mape != null);
      const dcBanks = data.filter((m: ModelMetadata) => m.bank_name && m.card_type === "DC" && m.cv_mape != null);
      if (ccBanks.length > 0) setCcMape(median(ccBanks.map((m: ModelMetadata) => m.cv_mape!)).toFixed(1) + "%");
      if (dcBanks.length > 0) setDcMape(median(dcBanks.map((m: ModelMetadata) => m.cv_mape!)).toFixed(1) + "%");
      const ccAgg = data.find((m: ModelMetadata) => !m.bank_name && m.metric === "cc_outstanding" && m.cv_mape != null);
      const dcAgg = data.find((m: ModelMetadata) => !m.bank_name && m.metric === "dc_outstanding" && m.cv_mape != null);
      if (ccAgg) setCcAggMape(ccAgg.cv_mape!.toFixed(1) + "%");
      if (dcAgg) setDcAggMape(dcAgg.cv_mape!.toFixed(1) + "%");
    }
    load();
  }, []);

  return (
    <div className="max-w-3xl">
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">About MIP</h1>

      <section className="mb-8">
        <h2 className="text-lg font-semibold text-gray-800 dark:text-slate-200 mb-3">What is MIP?</h2>
        <p className="text-gray-500 dark:text-slate-400 leading-relaxed">
          MPi Market Intelligence Platform (MIP) is a forecasting system for India&apos;s
          credit card, debit card, and digital payments market. It provides 24-month forward-looking
          estimates at both the national aggregate level and for individual banks — enabling card manufacturers,
          banking partners, and strategy teams to plan production, assess market share, and anticipate demand shifts.
        </p>
      </section>

      <section className="mb-8">
        <h2 className="text-lg font-semibold text-gray-800 dark:text-slate-200 mb-3">Data Sources</h2>
        <ul className="space-y-2 text-sm text-gray-500 dark:text-slate-400">
          <li><strong className="text-gray-700 dark:text-slate-300">RBI Bankwise PSI:</strong> Monthly bank-level cards outstanding, ATMs, PoS terminals, and transaction volumes — sourced directly from the Reserve Bank of India</li>
          <li><strong className="text-gray-700 dark:text-slate-300">RBI Payment System Indicators:</strong> Aggregate CC/DC/UPI volumes published monthly by RBI</li>
          <li><strong className="text-gray-700 dark:text-slate-300">NPCI UPI:</strong> Monthly UPI transaction volume and value from the National Payments Corporation of India</li>
          <li><strong className="text-gray-700 dark:text-slate-300">RBI Repo Rate:</strong> Policy rate used as a macroeconomic regressor to capture monetary policy effects</li>
        </ul>
      </section>

      <section className="mb-8">
        <h2 className="text-lg font-semibold text-gray-800 dark:text-slate-200 mb-3">Methodology</h2>
        <ul className="space-y-2 text-sm text-gray-500 dark:text-slate-400">
          <li><strong className="text-gray-700 dark:text-slate-300">Ensemble Forecasting:</strong> Prophet + ARIMA(1,1,1) + damped ETS with per-series cross-validation-optimized weights</li>
          <li><strong className="text-gray-700 dark:text-slate-300">Bank-Level Models:</strong> Top 10 credit card and top 15 debit card issuers modelled individually; remaining banks aggregated into a residual bucket</li>
          <li><strong className="text-gray-700 dark:text-slate-300">Ground-Up Approach:</strong> Individual bank forecasts + residual = India total, cross-checked against aggregate for consistency</li>
          <li><strong className="text-gray-700 dark:text-slate-300">Validation:</strong> Walk-forward cross-validation (48-month initial window, 6-month horizon, 6-month step) with out-of-sample testing</li>
          <li><strong className="text-gray-700 dark:text-slate-300">Confidence Intervals:</strong> 90% conformal prediction intervals from walk-forward CV residual quantiles</li>
        </ul>
      </section>

      <section className="mb-8">
        <h2 className="text-lg font-semibold text-gray-800 dark:text-slate-200 mb-3">Forecast Accuracy</h2>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div className="bg-white dark:bg-slate-800/50 border border-gray-200 dark:border-slate-700/50 rounded-xl p-4">
            <p className="text-gray-400 dark:text-slate-500">CC Bank Median CV MAPE</p>
            <p className="text-xl font-bold text-gray-900 dark:text-white">{ccMape}</p>
          </div>
          <div className="bg-white dark:bg-slate-800/50 border border-gray-200 dark:border-slate-700/50 rounded-xl p-4">
            <p className="text-gray-400 dark:text-slate-500">DC Bank Median CV MAPE</p>
            <p className="text-xl font-bold text-gray-900 dark:text-white">{dcMape}</p>
          </div>
          <div className="bg-white dark:bg-slate-800/50 border border-gray-200 dark:border-slate-700/50 rounded-xl p-4">
            <p className="text-gray-400 dark:text-slate-500">CC Aggregate OOS MAPE</p>
            <p className="text-xl font-bold text-gray-900 dark:text-white">{ccAggMape}</p>
          </div>
          <div className="bg-white dark:bg-slate-800/50 border border-gray-200 dark:border-slate-700/50 rounded-xl p-4">
            <p className="text-gray-400 dark:text-slate-500">DC Aggregate OOS MAPE</p>
            <p className="text-xl font-bold text-gray-900 dark:text-white">{dcAggMape}</p>
          </div>
        </div>
      </section>

      <section className="mb-8">
        <h2 className="text-lg font-semibold text-gray-800 dark:text-slate-200 mb-3">Coverage</h2>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div className="bg-white dark:bg-slate-800/50 border border-gray-200 dark:border-slate-700/50 rounded-xl p-4">
            <p className="text-gray-400 dark:text-slate-500">Credit Card Banks Modelled</p>
            <p className="text-xl font-bold text-gray-900 dark:text-white">10 + Residual</p>
            <p className="text-xs text-gray-400 dark:text-slate-500 mt-1">~91% of India total outstanding</p>
          </div>
          <div className="bg-white dark:bg-slate-800/50 border border-gray-200 dark:border-slate-700/50 rounded-xl p-4">
            <p className="text-gray-400 dark:text-slate-500">Debit Card Banks Modelled</p>
            <p className="text-xl font-bold text-gray-900 dark:text-white">15 + Residual</p>
            <p className="text-xs text-gray-400 dark:text-slate-500 mt-1">~83% of India total outstanding</p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-gray-800 dark:text-slate-200 mb-3">Limitations</h2>
        <ul className="space-y-1 text-sm text-gray-400 dark:text-slate-500">
          <li>Forecasts assume historical patterns continue — structural breaks (new regulations, bank mergers) are not predicted</li>
          <li>Jun 2025 CC dip confirmed as an RBI reporting anomaly and excluded from model assessment</li>
          <li>NPCI UPI data undergoes occasional revisions; forecasts use the latest available figures</li>
          <li>Forward-looking variables (bank news, policy announcements) planned for Phase 3</li>
        </ul>
      </section>
    </div>
  );
}
