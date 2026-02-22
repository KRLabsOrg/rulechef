export const ENTITY_COLORS: Record<string, string> = {
  PER: "#e8d0f0",
  PERSON: "#e8d0f0",
  ORG: "#cce5ff",
  ORGANIZATION: "#cce5ff",
  LOC: "#d4edda",
  LOCATION: "#d4edda",
  GPE: "#d4edda",
  DATE: "#fff3cd",
  TIME: "#fff3cd",
  MONEY: "#f8d7da",
  MISC: "#f8d7da",
  EVENT: "#d1ecf1",
  PRODUCT: "#e2e3e5",
  DEFAULT: "#fef9c3",
};

export function getEntityColor(type: string | undefined | null): string {
  if (!type) return ENTITY_COLORS.DEFAULT;
  return ENTITY_COLORS[type.toUpperCase()] ?? ENTITY_COLORS.DEFAULT;
}
