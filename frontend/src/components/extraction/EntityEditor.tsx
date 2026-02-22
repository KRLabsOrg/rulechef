import type { Entity } from "@/api/types";
import type { SetEntities } from "@/lib/entities";
import {
  labelOptionsFor,
  removeEntityAt,
  updateEntityLabelAt,
  updateEntityRangeAt,
} from "@/lib/entities";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface EntityEditorProps {
  entities: Entity[];
  setEntities: SetEntities;
  text: string;
  labels: string[];
  disabled?: boolean;
  showRuleName?: boolean;
}

export function EntityEditor({
  entities,
  setEntities,
  text,
  labels,
  disabled = false,
  showRuleName = false,
}: EntityEditorProps) {
  if (entities.length === 0) return null;

  return (
    <div className="space-y-1 rounded-md border p-2">
      <p className="text-xs text-muted-foreground">Entities</p>
      {entities.map((entity, index) => (
        <div
          key={`${entity.start}-${entity.end}-${entity.type}-${index}`}
          className="flex items-center justify-between text-sm"
        >
          <span className="flex items-center gap-2">
            <span>
              <strong>{entity.type}</strong> [{entity.start},{entity.end}) "{entity.text}"
            </span>
            {showRuleName && entity.rule_name && (
              <span className="inline-flex items-center rounded bg-muted px-1.5 py-0.5 text-[10px] font-mono text-muted-foreground">
                {entity.rule_name}
              </span>
            )}
          </span>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              <Label className="text-xs text-muted-foreground">Start</Label>
              <Input
                type="number"
                min={0}
                max={Math.max(0, text.length - 1)}
                value={entity.start}
                className="h-8 w-20"
                onChange={(event) => {
                  const parsed = Number(event.target.value);
                  if (!Number.isFinite(parsed)) return;
                  updateEntityRangeAt(setEntities, text, index, parsed, entity.end);
                }}
                disabled={disabled}
              />
            </div>
            <div className="flex items-center gap-1">
              <Label className="text-xs text-muted-foreground">End</Label>
              <Input
                type="number"
                min={1}
                max={text.length}
                value={entity.end}
                className="h-8 w-20"
                onChange={(event) => {
                  const parsed = Number(event.target.value);
                  if (!Number.isFinite(parsed)) return;
                  updateEntityRangeAt(setEntities, text, index, entity.start, parsed);
                }}
                disabled={disabled}
              />
            </div>
            <Select
              value={entity.type}
              onValueChange={(value) => updateEntityLabelAt(setEntities, index, value)}
              disabled={disabled}
            >
              <SelectTrigger className="h-8 w-28">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {labelOptionsFor(entity.type, labels).map((label) => (
                  <SelectItem key={label} value={label}>
                    {label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => removeEntityAt(setEntities, index)}
              disabled={disabled}
            >
              Remove
            </Button>
          </div>
        </div>
      ))}
    </div>
  );
}
