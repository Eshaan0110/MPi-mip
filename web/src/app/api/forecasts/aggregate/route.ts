import { NextRequest, NextResponse } from "next/server";
import { supabase, isConfigured } from "@/lib/supabase";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  if (!isConfigured) return NextResponse.json([], { status: 200 });

  const month = request.nextUrl.searchParams.get("month");
  let query = supabase.from("forecasts_aggregate").select("*").order("forecast_month", { ascending: true });
  if (month) query = query.eq("forecast_month", month);

  const { data, error } = await query;
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  return NextResponse.json(data, {
    headers: { "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=1800" },
  });
}
