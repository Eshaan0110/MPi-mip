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
      className="border border-gray-300 rounded-md px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-brand-500"
    >
      {months.map((m) => (
        <option key={m} value={m}>
          {new Date(m + "-01").toLocaleDateString("en-IN", {
            month: "long",
            year: "numeric",
          })}
        </option>
      ))}
    </select>
  );
}
