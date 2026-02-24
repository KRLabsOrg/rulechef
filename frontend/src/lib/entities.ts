import type { Dispatch, SetStateAction } from "react";
import type { Entity } from "@/api/types";

export const DEFAULT_LABELS = ["PER", "ORG", "LOC", "DATE", "MISC"];
export const LABELS_STORAGE_KEY = "rulechef_labels";

// --- Normalization ---

/**
 * Normalize raw API/unknown data into typed Entity[].
 * Handles overlap deduplication (longest-first wins).
 */
export function normalizeEntities(raw: unknown): Entity[] {
  if (!Array.isArray(raw)) {
    return [];
  }

  const parsed = raw
    .map((item) => {
      if (!item || typeof item !== "object") {
        return null;
      }
      const entity = item as Record<string, unknown>;
      return {
        text: String(entity.text ?? ""),
        start: Number(entity.start ?? 0),
        end: Number(entity.end ?? 0),
        type: String(entity.type ?? entity.label ?? "ENTITY").trim().toUpperCase(),
        ...(entity.rule_id ? { rule_id: String(entity.rule_id) } : {}),
        ...(entity.rule_name ? { rule_name: String(entity.rule_name) } : {}),
      };
    })
    .filter(
      (entity): entity is Entity =>
        !!entity &&
        Number.isFinite(entity.start) &&
        Number.isFinite(entity.end) &&
        entity.end > entity.start &&
        entity.type.length > 0,
    )
    .sort(
      (a, b) =>
        a.start - b.start ||
        b.end - b.start - (a.end - a.start) ||
        a.type.localeCompare(b.type),
    );

  // Deduplicate overlapping spans (keep first = longest)
  const selected: Entity[] = [];
  for (const entity of parsed) {
    const overlaps = selected.some(
      (picked) => entity.start < picked.end && entity.end > picked.start,
    );
    if (!overlaps) {
      selected.push(entity);
    }
  }
  return selected.sort((a, b) => a.start - b.start || a.end - b.end);
}

/**
 * Parse entities from an output dict (e.g. example.expected_output).
 */
export function parseEntities(output: Record<string, unknown>): Entity[] {
  const raw = output.entities;
  if (!Array.isArray(raw)) {
    return [];
  }
  return normalizeEntities(raw);
}

// --- Labels ---

export function parseLabels(raw: string): string[] {
  const unique = new Set(
    raw
      .split(",")
      .map((part) => part.trim())
      .filter(Boolean)
      .map((label) => label.toUpperCase()),
  );
  const parsed = Array.from(unique);
  return parsed.length > 0 ? parsed : DEFAULT_LABELS;
}

export function labelOptionsFor(entityType: string, labels: string[]): string[] {
  return Array.from(
    new Set([entityType.toUpperCase(), ...labels.map((label) => label.toUpperCase())]),
  );
}

export function loadStoredLabels(): string[] {
  try {
    const raw = localStorage.getItem(LABELS_STORAGE_KEY);
    if (!raw) return DEFAULT_LABELS;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return DEFAULT_LABELS;
    const labels = parsed
      .map((entry) => String(entry).trim().toUpperCase())
      .filter(Boolean);
    return labels.length > 0 ? labels : DEFAULT_LABELS;
  } catch {
    return DEFAULT_LABELS;
  }
}

export function loadStoredLabelsString(): string {
  return loadStoredLabels().join(", ");
}

// --- Entity mutations ---

export type SetEntities = Dispatch<SetStateAction<Entity[]>>;

export function removeEntityAt(setEntities: SetEntities, index: number): void {
  setEntities((prev) => prev.filter((_, i) => i !== index));
}

export function updateEntityLabelAt(
  setEntities: SetEntities,
  index: number,
  newType: string,
): void {
  const normalized = newType.toUpperCase();
  setEntities((prev) =>
    prev.map((entity, i) => (i === index ? { ...entity, type: normalized } : entity)),
  );
}

export function updateEntityRangeAt(
  setEntities: SetEntities,
  text: string,
  index: number,
  nextStart: number,
  nextEnd: number,
): void {
  const textLength = text.length;
  if (textLength === 0) return;
  const clampedStart = Math.max(0, Math.min(Math.floor(nextStart), textLength - 1));
  const clampedEnd = Math.max(clampedStart + 1, Math.min(Math.floor(nextEnd), textLength));

  setEntities((prev) =>
    prev.map((entity, i) =>
      i === index
        ? {
            ...entity,
            start: clampedStart,
            end: clampedEnd,
            text: text.slice(clampedStart, clampedEnd),
          }
        : entity,
    ),
  );
}

// --- Helpers ---

export function getExampleText(input: Record<string, unknown>): string {
  const text = input.text;
  if (typeof text === "string") return text;
  for (const value of Object.values(input)) {
    if (typeof value === "string" && value.trim()) return value;
  }
  return "";
}

/**
 * Deep-compare two entity arrays by value (start, end, type).
 */
export function entitiesEqual(a: Entity[], b: Entity[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i].start !== b[i].start || a[i].end !== b[i].end || a[i].type !== b[i].type) {
      return false;
    }
  }
  return true;
}
