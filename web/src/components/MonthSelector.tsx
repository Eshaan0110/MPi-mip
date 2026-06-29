"use client";

interface MonthSelectorProps {
  months: string[];
  selected: string;
  onChange: (month: string) => void;
}

export function MonthSelector({ months, selected, onChange }: MonthSelectorProps) {
  return (
    <select
      value={selected}
      onChange={(e) => onChange(e.target.value)}
      className="border border-gray-300 dark:border-slate-600 rounded-md px-3 py-2 text-sm bg-white dark:bg-slate-800 text-gray-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
    >
      {months.map((m) => (
        <option key={m} value={m}>
          {new Date(m.length === 7 ? m + "-01" : m).toLocaleDateString("en-IN", {
            month: "long",
            year: "numeric",
          })}
        </option>
      ))}
    </select>
  );
}
