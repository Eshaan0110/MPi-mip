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

  useEffect(() => {
    async function load() {
      const { data } = await supabase.from("model_metadata").select("*");
      if (!data) return;
      const ccBanks = data.filter((m: ModelMetadata) => m.bank_name && m.card_type === "CC" && m.cv_mape != null);
      const dcBanks = data.filter((m: ModelMetadata) => m.bank_name && m.card_type === "DC" && m.cv_mape != null);
      if (ccBanks.length > 0) setCcMape(median(ccBanks.map((m: ModelMetadata) => m.cv_mape!)).toFixed(1) + "%");
      if (dcBanks.length > 0) setDcMape(median(dcBanks.map((m: ModelMetadata) => m.cv_mape!)).toFixed(1) + "%");
    }
    load();
  }, []);

  return (
    <div className="max-w-3xl">
      <h1 className="text-2xl font-bold text-brand-700 mb-6">About MIP</h1>

      <section className="mb-8">
        <h2 className="text-lg font-semibold text-brand-600 mb-3">What is MIP?</h2>
        <p className="text-gray-600 leading-relaxed">
          MPi Market Intelligence Platform (MIP) is an internal forecasting tool for India&apos;s
          credit card, debit card, and digital payments market. It provides 24-month forward-looking
          estimates at both aggregate (India) and individual bank levels.
        </p>
      </section>

      <section className="mb-8">
        <h2 className="text-lg font-semibold text-brand-600 mb-3">Data Sources</h2>
        <ul className="space-y-2 text-sm text-gray-600">
          <li><strong>RBI Bankwise:</strong> Monthly bank-level cards outstanding, ATMs, PoS terminals, transaction volumes</li>
          <li><strong>RBI PSI:</strong> Payment System Indicators — aggregate CC/DC/UPI volumes</li>
          <li><strong>NPCI UPI:</strong> Monthly UPI transaction volume and value</li>
          <li><strong>RBI Repo Rate:</strong> Policy rate used as macroeconomic regressor</li>
        </ul>
      </section>

      <section className="mb-8">
        <h2 className="text-lg font-semibold text-brand-600 mb-3">Methodology</h2>
        <ul className="space-y-2 text-sm text-gray-600">
          <li><strong>Models:</strong> Facebook Prophet (with logistic growth caps) and Holt-Winters ETS for stable-growth banks</li>
          <li><strong>Validation:</strong> Walk-forward cross-validation (36m initial, 6m horizon, 6m step) + out-of-sample testing on held-out months</li>
          <li><strong>Variable Selection:</strong> Granger causality testing on first-differenced series to validate regressors</li>
          <li><strong>Ground-Up:</strong> Individual bank models summed + residual adjustment = India total</li>
          <li><strong>Confidence Intervals:</strong> 90% prediction intervals from Prophet/ETS simulation</li>
        </ul>
      </section>

      <section className="mb-8">
        <h2 className="text-lg font-semibold text-brand-600 mb-3">Accuracy</h2>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <p className="text-gray-500">CC Bank Median CV MAPE</p>
            <p className="text-xl font-bold text-brand-700">{ccMape}</p>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <p className="text-gray-500">DC Bank Median CV MAPE</p>
            <p className="text-xl font-bold text-brand-700">{dcMape}</p>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <p className="text-gray-500">CC Aggregate OOS MAPE</p>
            <p className="text-xl font-bold text-brand-700">1.6%</p>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <p className="text-gray-500">DC Aggregate OOS MAPE</p>
            <p className="text-xl font-bold text-brand-700">1.0%</p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-lg font-semibold text-brand-600 mb-3">Limitations</h2>
        <ul className="space-y-1 text-sm text-gray-500">
          <li>Forecasts assume historical patterns continue; structural breaks (new regulations, mergers) are not predicted</li>
          <li>Jun 2025 CC dip confirmed as RBI reporting blip — excluded from model assessment</li>
          <li>NPCI UPI data has occasional revisions; forecasts use latest available figures</li>
          <li>Bank-exclusive regressors (branches, CASA, NPA) not yet incorporated — planned for Phase 3</li>
        </ul>
      </section>
    </div>
  );
}
