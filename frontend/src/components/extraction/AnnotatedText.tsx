import { useCallback, useEffect, useRef, useState } from "react";
import type { Entity } from "@/api/types";
import { getEntityColor } from "./colors";
import { Button } from "@/components/ui/button";

interface AnnotatedTextProps {
  text: string;
  entities: Entity[];
  labels: string[];
  editable?: boolean;
  onChange?: (entities: Entity[]) => void;
}

interface Segment {
  text: string;
  start: number;
  end: number;
  entity?: Entity;
}

function collapseOverlaps(entities: Entity[]): Entity[] {
  const sorted = [...entities].sort(
    (a, b) =>
      a.start - b.start ||
      (b.end - b.start) - (a.end - a.start) ||
      a.type.localeCompare(b.type)
  );
  const selected: Entity[] = [];

  for (const entity of sorted) {
    const overlaps = selected.some(
      (picked) => entity.start < picked.end && entity.end > picked.start
    );
    if (!overlaps) {
      selected.push(entity);
    }
  }

  return selected.sort((a, b) => a.start - b.start || a.end - b.end);
}

function normalizeEntityBounds(entities: Entity[], textLength: number): Entity[] {
  if (textLength <= 0) {
    return [];
  }
  return entities
    .map((entity) => {
      const start = Math.max(0, Math.min(Math.floor(entity.start), textLength - 1));
      const end = Math.max(start + 1, Math.min(Math.floor(entity.end), textLength));
      return {
        ...entity,
        start,
        end,
      };
    })
    .filter((entity) => entity.end > entity.start);
}

function buildSegments(text: string, entities: Entity[]): Segment[] {
  const bounded = normalizeEntityBounds(entities, text.length);
  const sorted = collapseOverlaps(bounded);
  const segments: Segment[] = [];
  let cursor = 0;

  for (const ent of sorted) {
    if (ent.start < cursor) {
      continue;
    }
    if (ent.start > cursor) {
      segments.push({ text: text.slice(cursor, ent.start), start: cursor, end: ent.start });
    }
    segments.push({ text: text.slice(ent.start, ent.end), start: ent.start, end: ent.end, entity: ent });
    cursor = ent.end;
  }
  if (cursor < text.length) {
    segments.push({ text: text.slice(cursor), start: cursor, end: text.length });
  }
  return segments;
}

/**
 * Convert a DOM selection within the annotated-text container to plain-text character offsets.
 */
function selectionToOffsets(container: HTMLElement): { start: number; end: number } | null {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed || !sel.rangeCount) return null;

  const range = sel.getRangeAt(0);
  if (!container.contains(range.startContainer) || !container.contains(range.endContainer)) {
    return null;
  }

  const computeOffset = (node: Node, offset: number): number => {
    const prefixRange = document.createRange();
    prefixRange.selectNodeContents(container);
    prefixRange.setEnd(node, offset);
    const fragment = prefixRange.cloneContents();
    fragment
      .querySelectorAll("[data-annotation-meta='true']")
      .forEach((meta) => meta.remove());
    return (fragment.textContent ?? "").length;
  };

  const start = computeOffset(range.startContainer, range.startOffset);
  const end = computeOffset(range.endContainer, range.endOffset);

  if (start >= 0 && end >= 0) {
    const normalizedStart = Math.min(start, end);
    const normalizedEnd = Math.max(start, end);
    if (normalizedEnd > normalizedStart) {
      return { start: normalizedStart, end: normalizedEnd };
    }
  }
  return null;
}

export function AnnotatedText({
  text,
  entities,
  labels,
  editable = false,
  onChange,
}: AnnotatedTextProps) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [pickerAnchor, setPickerAnchor] = useState<{ top: number; left: number } | null>(null);
  const [pendingRange, setPendingRange] = useState<{ start: number; end: number } | null>(null);
  const [clickedEntity, setClickedEntity] = useState<Entity | null>(null);
  const [entityPopoverPos, setEntityPopoverPos] = useState<{ top: number; left: number } | null>(null);

  const segments = buildSegments(text, entities);

  useEffect(() => {
    const closeFloatingMenus = () => {
      setPickerAnchor(null);
      setPendingRange(null);
      setClickedEntity(null);
      setEntityPopoverPos(null);
    };
    window.addEventListener("resize", closeFloatingMenus);
    window.addEventListener("scroll", closeFloatingMenus, true);
    return () => {
      window.removeEventListener("resize", closeFloatingMenus);
      window.removeEventListener("scroll", closeFloatingMenus, true);
    };
  }, []);

  const handleMouseUp = useCallback(() => {
    if (!editable || !containerRef.current || !wrapperRef.current) return;

    const offsets = selectionToOffsets(containerRef.current);
    if (!offsets) {
      setPickerAnchor(null);
      setPendingRange(null);
      return;
    }

    const sel = window.getSelection();
    if (!sel || !sel.rangeCount) return;
    const rect = sel.getRangeAt(0).getBoundingClientRect();
    const wrapperRect = wrapperRef.current.getBoundingClientRect();
    const anchorLeft = rect.left + rect.width / 2 - wrapperRect.left;
    const anchorTop = rect.top - 8 - wrapperRect.top;

    setPendingRange(offsets);
    setPickerAnchor({ top: anchorTop, left: anchorLeft });
    setClickedEntity(null);
  }, [editable]);

  const handleLabelPick = useCallback(
    (label: string) => {
      if (!pendingRange || !onChange) return;
      const newEntity: Entity = {
        text: text.slice(pendingRange.start, pendingRange.end),
        start: pendingRange.start,
        end: pendingRange.end,
        type: label,
      };
      // Remove any overlapping entities
      const filtered = entities.filter(
        (e) => !(e.start < newEntity.end && e.end > newEntity.start)
      );
      onChange([...filtered, newEntity]);
      setPickerAnchor(null);
      setPendingRange(null);
      window.getSelection()?.removeAllRanges();
    },
    [pendingRange, entities, text, onChange]
  );

  const handleEntityClick = useCallback(
    (entity: Entity, event: React.MouseEvent) => {
      if (!editable || !wrapperRef.current) return;
      const sel = window.getSelection();
      if (sel && !sel.isCollapsed) return;
      event.stopPropagation();
      const wrapperRect = wrapperRef.current.getBoundingClientRect();
      const anchorLeft = event.clientX - wrapperRect.left;
      const anchorTop = event.clientY - 8 - wrapperRect.top;
      setClickedEntity(entity);
      setEntityPopoverPos({ top: anchorTop, left: anchorLeft });
      setPickerAnchor(null);
    },
    [editable]
  );

  const handleRemoveEntity = useCallback(() => {
    if (!clickedEntity || !onChange) return;
    onChange(entities.filter((e) => e.start !== clickedEntity.start || e.end !== clickedEntity.end));
    setClickedEntity(null);
    setEntityPopoverPos(null);
  }, [clickedEntity, entities, onChange]);

  const handleChangeLabel = useCallback(
    (newLabel: string) => {
      if (!clickedEntity || !onChange) return;
      onChange(
        entities.map((e) =>
          e.start === clickedEntity.start && e.end === clickedEntity.end
            ? { ...e, type: newLabel }
            : e
        )
      );
      setClickedEntity(null);
      setEntityPopoverPos(null);
    },
    [clickedEntity, entities, onChange]
  );

  return (
    <div ref={wrapperRef} className="relative">
      <div
        ref={containerRef}
        className="p-4 border rounded-md leading-7 text-base select-text min-h-[100px]"
        onMouseUp={handleMouseUp}
      >
        {segments.map((seg) =>
          seg.entity ? (
            <mark
              key={`entity-${seg.start}-${seg.end}-${seg.entity.type}`}
              className="relative rounded px-0.5 cursor-pointer font-medium"
              style={{ backgroundColor: getEntityColor(seg.entity.type) }}
              onClick={(e) => handleEntityClick(seg.entity!, e)}
              title={seg.entity.type}
            >
              {seg.text}
              <sup
                data-annotation-meta="true"
                className="text-[9px] ml-0.5 opacity-70 select-none pointer-events-none"
              >
                {seg.entity.type}
              </sup>
            </mark>
          ) : (
            <span key={`text-${seg.start}-${seg.end}`}>{seg.text}</span>
          )
        )}
        {text.length === 0 && (
          <span className="text-muted-foreground">Text will appear here after extraction...</span>
        )}
      </div>

      {/* Label picker for new selections */}
      {editable && pickerAnchor && (
        <div
          className="absolute z-[95] -translate-x-1/2 -translate-y-full rounded-md border bg-popover p-2 shadow-md"
          style={{ top: pickerAnchor.top, left: pickerAnchor.left }}
          onMouseDown={(event) => event.preventDefault()}
        >
          <div className="flex flex-wrap gap-1">
            {labels.map((label) => (
              <Button
                key={label}
                variant="outline"
                size="sm"
                className="text-xs"
                style={{ backgroundColor: getEntityColor(label) }}
                onClick={() => handleLabelPick(label)}
              >
                {label}
              </Button>
            ))}
          </div>
        </div>
      )}

      {/* Entity click popover for edit/remove */}
      {editable && clickedEntity && entityPopoverPos && (
        <div
          className="absolute z-[95] -translate-x-1/2 -translate-y-full rounded-md border bg-popover p-2 shadow-md"
          style={{ top: entityPopoverPos.top, left: entityPopoverPos.left }}
          onMouseDown={(event) => event.preventDefault()}
        >
          <div className="space-y-2">
            <p className="text-xs text-muted-foreground">
              "{clickedEntity.text}" — {clickedEntity.type}
            </p>
            <div className="flex flex-wrap gap-1">
              {labels
                .filter((l) => l !== clickedEntity.type)
                .map((label) => (
                  <Button
                    key={label}
                    variant="outline"
                    size="sm"
                    className="text-xs"
                    style={{ backgroundColor: getEntityColor(label) }}
                    onClick={() => handleChangeLabel(label)}
                  >
                    {label}
                  </Button>
                ))}
              <Button
                variant="destructive"
                size="sm"
                className="text-xs"
                onClick={handleRemoveEntity}
              >
                Remove
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
