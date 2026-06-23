import { NextResponse } from "next/server";
import { supabase, isConfigured } from "@/lib/supabase";

export const dynamic = "force-dynamic";

export async function GET() {
  if (!isConfigured) {
    return NextResponse.json(
      { status: "not_configured", message: "Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY" },
      { status: 503 }
    );
  }

  try {
    const [{ count }, { data: lastPipeline }] = await Promise.all([
      supabase.from("forecasts_bank").select("*", { count: "exact", head: true }),
      supabase.from("pipeline_runs").select("started_at, status").order("started_at", { ascending: false }).limit(1).single(),
    ]);

    return NextResponse.json({
      status: "healthy",
      database: "connected",
      forecast_rows: count ?? 0,
      last_pipeline: lastPipeline ?? null,
      checked_at: new Date().toISOString(),
    });
  } catch (e) {
    return NextResponse.json({ status: "error", message: String(e) }, { status: 503 });
  }
}
