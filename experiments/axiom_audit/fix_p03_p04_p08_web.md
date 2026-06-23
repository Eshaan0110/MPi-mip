# Web App Fixes: P03, P04, P08

## P03 — Config Check (web/src/lib/supabase.ts)

Replace placeholder fallback with explicit error:

```ts
import { createClient, SupabaseClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL ?? "";
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? "";

export const isConfigured = Boolean(supabaseUrl && supabaseAnonKey);

export const supabase: SupabaseClient = createClient(
  supabaseUrl || "https://placeholder.supabase.co",
  supabaseAnonKey || "placeholder"
);
```

Add a ConfigWarning component:

```tsx
// web/src/components/ConfigWarning.tsx
import { isConfigured } from "@/lib/supabase";

export function ConfigWarning() {
  if (isConfigured) return null;
  return (
    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
      <p className="text-yellow-800 font-medium">Supabase not configured</p>
      <p className="text-yellow-700 text-sm mt-1">
        Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY in .env.local
      </p>
    </div>
  );
}
```

## P04 — Error States

Every page should handle the error case. Pattern:

```tsx
const [error, setError] = useState<string | null>(null);

// In load():
const { data, error: err } = await supabase.from("table").select("*");
if (err) { setError(err.message); setLoading(false); return; }

// In render:
if (error) return (
  <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
    <p className="text-red-700 font-medium">Failed to load data</p>
    <p className="text-red-600 text-sm mt-1">{error}</p>
  </div>
);
```

## P08 — Next.js Link

In `web/src/app/layout.tsx`, change:
```tsx
import Link from "next/link";
// ...
<a href={l.href} ...>  →  <Link href={l.href} ...>
```
