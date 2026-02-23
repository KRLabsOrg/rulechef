import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useExamples, useDeleteExample } from "@/api/client";
import type { ExampleItem } from "@/api/types";
import { toast } from "sonner";

interface ExampleTableProps {
  onAnnotate?: (example: ExampleItem) => void;
}

type StatusVariant = "secondary" | "outline" | "default";

function isBufferedId(id: string): boolean {
  return id.startsWith("buffer:");
}

function isBufferedCorrectionId(id: string): boolean {
  return id.startsWith("buffer-correction:");
}

function getExampleStatus(example: ExampleItem): { label: string; variant: StatusVariant } {
  if (example.id.startsWith("draft:") || String(example.source).toLowerCase() === "draft") {
    return { label: "needs annotation", variant: "outline" };
  }
  if (isBufferedId(example.id)) {
    return { label: "pending training", variant: "secondary" };
  }
  return { label: "in current rules", variant: "default" };
}

function getCorrectionStatus(id: string): { label: string; variant: StatusVariant } {
  if (isBufferedCorrectionId(id)) {
    return { label: "pending training", variant: "secondary" };
  }
  return { label: "in current rules", variant: "default" };
}

function getExampleText(input: Record<string, unknown>): string {
  const textValue = input.text;
  if (typeof textValue === "string") {
    return textValue;
  }

  for (const value of Object.values(input)) {
    if (typeof value === "string" && value.trim()) {
      return value;
    }
  }
  return JSON.stringify(input);
}

function getEntitySummary(output: Record<string, unknown>): string {
  const entities = output.entities;
  if (!Array.isArray(entities) || entities.length === 0) {
    return "0";
  }
  const counts = new Map<string, number>();
  for (const entity of entities) {
    if (!entity || typeof entity !== "object") {
      continue;
    }
    const type = String((entity as Record<string, unknown>).type ?? "").toUpperCase();
    if (!type) {
      continue;
    }
    counts.set(type, (counts.get(type) ?? 0) + 1);
  }

  const labels = Array.from(counts.entries())
    .map(([label, count]) => `${label}:${count}`)
    .slice(0, 3)
    .join(", ");
  return labels ? `${entities.length} (${labels})` : String(entities.length);
}

export function ExampleTable({ onAnnotate }: ExampleTableProps) {
  const { data, isLoading } = useExamples();
  const deleteExample = useDeleteExample();

  if (isLoading) return <p className="text-sm text-muted-foreground">Loading...</p>;

  const examples = data?.examples ?? [];
  const corrections = data?.corrections ?? [];
  const draftExamples = examples.filter(
    (example) =>
      example.id.startsWith("draft:") || String(example.source).toLowerCase() === "draft"
  ).length;
  const bufferedExamples = examples.filter((example) => isBufferedId(example.id)).length;
  const learnedExamples = Math.max(0, examples.length - draftExamples - bufferedExamples);
  const bufferedCorrections = corrections.filter((correction) =>
    isBufferedCorrectionId(correction.id)
  ).length;
  const learnedCorrections = corrections.length - bufferedCorrections;

  if (examples.length === 0 && corrections.length === 0) {
    return (
      <p className="text-sm text-muted-foreground py-4 text-center">
        No examples yet. Add text above to get started.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      <p className="text-xs text-muted-foreground">
        Status: <strong>Needs annotation</strong> means raw text still needs labels.{" "}
        <strong>Pending training</strong> means saved data waiting for the next Train run.{" "}
        <strong>In current rules</strong> means the latest training already used it.
      </p>

      {examples.length > 0 && (
        <div>
          <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
            Examples ({examples.length})
            {draftExamples > 0 && (
              <Badge variant="outline" className="text-[10px]">
                needs annotation {draftExamples}
              </Badge>
            )}
            {bufferedExamples > 0 && (
              <Badge variant="secondary" className="text-[10px]">
                pending training {bufferedExamples}
              </Badge>
            )}
            {learnedExamples > 0 && (
              <Badge className="text-[10px]">
                in current rules {learnedExamples}
              </Badge>
            )}
          </h3>
          <div className="border rounded-md">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Text</TableHead>
                  <TableHead className="w-32">Entities</TableHead>
                  <TableHead className="w-28">Status</TableHead>
                  <TableHead className="w-28">Source</TableHead>
                  <TableHead className="w-28 text-right">Action</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {examples.map((ex) => {
                  const status = getExampleStatus(ex);
                  return (
                    <TableRow key={ex.id}>
                      <TableCell className="max-w-[560px] whitespace-normal break-words">
                        {getExampleText(ex.input)}
                      </TableCell>
                      <TableCell>{getEntitySummary(ex.expected_output)}</TableCell>
                      <TableCell>
                        <Badge variant={status.variant}>{status.label}</Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant="secondary">{ex.source}</Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-1">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => onAnnotate?.(ex)}
                          >
                            Edit
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="text-destructive hover:text-destructive"
                            onClick={() =>
                              deleteExample.mutate(ex.id, {
                                onSuccess: () => toast.success("Example deleted"),
                                onError: (err: Error) => toast.error(err.message),
                              })
                            }
                            disabled={deleteExample.isPending}
                          >
                            Delete
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        </div>
      )}

      {corrections.length > 0 && (
        <div>
          <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
            Corrections ({corrections.length})
            {bufferedCorrections > 0 && (
              <Badge variant="secondary" className="text-[10px]">
                pending training {bufferedCorrections}
              </Badge>
            )}
            {learnedCorrections > 0 && (
              <Badge className="text-[10px]">
                in current rules {learnedCorrections}
              </Badge>
            )}
          </h3>
          <div className="border rounded-md">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Text</TableHead>
                  <TableHead className="w-36">Model Output</TableHead>
                  <TableHead className="w-36">Expected</TableHead>
                  <TableHead className="w-28">Status</TableHead>
                  <TableHead className="w-48">Feedback</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {corrections.map((c) => {
                  const status = getCorrectionStatus(c.id);
                  return (
                    <TableRow key={c.id}>
                      <TableCell className="max-w-[300px] whitespace-normal break-words text-sm">
                        {getExampleText(c.input)}
                      </TableCell>
                      <TableCell className="text-sm text-destructive">
                        {getEntitySummary(c.model_output)}
                      </TableCell>
                      <TableCell className="text-sm text-green-600">
                        {getEntitySummary(c.expected_output)}
                      </TableCell>
                      <TableCell>
                        <Badge variant={status.variant}>{status.label}</Badge>
                      </TableCell>
                      <TableCell className="max-w-xs truncate text-sm text-muted-foreground">
                        {c.feedback || "-"}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        </div>
      )}
    </div>
  );
}
