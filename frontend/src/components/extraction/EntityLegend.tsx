import { ENTITY_COLORS } from "./colors";

interface EntityLegendProps {
  labels: string[];
}

export function EntityLegend({ labels }: EntityLegendProps) {
  if (labels.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-2">
      {labels.map((label) => (
        <span
          key={label}
          className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-medium"
          style={{ backgroundColor: ENTITY_COLORS[label.toUpperCase()] ?? ENTITY_COLORS.DEFAULT }}
        >
          <span
            className="w-2 h-2 rounded-full"
            style={{ backgroundColor: ENTITY_COLORS[label.toUpperCase()] ?? ENTITY_COLORS.DEFAULT, filter: "brightness(0.7)" }}
          />
          {label}
        </span>
      ))}
    </div>
  );
}
