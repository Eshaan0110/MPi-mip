import { createClient, SupabaseClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL ?? "";
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? "";

export const supabase: SupabaseClient = createClient(
  supabaseUrl || "https://localhost",
  supabaseAnonKey || "placeholder"
);

export const isConfigured = Boolean(supabaseUrl && supabaseAnonKey);

export const CONFIG_ERROR_MESSAGE =
  "Supabase is not configured on this deployment. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY in the environment and redeploy.";
