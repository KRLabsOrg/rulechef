import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useLearnStatus } from "@/api/client";
import type { ClassMetrics } from "@/api/types";

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

export function MetricsCard() {
  const { data } = useLearnStatus(false);
  const metrics = data?.metrics;

  if (!metrics) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            No metrics yet. Run learning first.
          </p>
        </CardContent>
      </Card>
    );
  }

  const perClass: ClassMetrics[] = (metrics.per_class as ClassMetrics[]) ?? [];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Metrics</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Summary row */}
        <div className="grid grid-cols-4 gap-3">
          <div>
            <p className="text-xs text-muted-foreground">Exact Match</p>
            <p className="text-xl font-bold">
              {metrics.exact_match != null ? pct(metrics.exact_match as number) : "—"}
            </p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Precision</p>
            <p className="text-xl font-bold">
              {metrics.micro_precision != null
                ? pct(metrics.micro_precision as number)
                : "—"}
            </p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Recall</p>
            <p className="text-xl font-bold">
              {metrics.micro_recall != null
                ? pct(metrics.micro_recall as number)
                : "—"}
            </p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">F1 (micro)</p>
            <p className="text-xl font-bold">
              {metrics.micro_f1 != null ? pct(metrics.micro_f1 as number) : "—"}
            </p>
          </div>
        </div>

        {/* Per-class breakdown */}
        {perClass.length > 0 && (
          <div>
            <p className="text-xs text-muted-foreground mb-2">Per-class breakdown</p>
            <div className="border rounded-md">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Class</TableHead>
                    <TableHead className="text-right">Prec</TableHead>
                    <TableHead className="text-right">Rec</TableHead>
                    <TableHead className="text-right">F1</TableHead>
                    <TableHead className="text-right">TP</TableHead>
                    <TableHead className="text-right">FP</TableHead>
                    <TableHead className="text-right">FN</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {perClass.map((c) => (
                    <TableRow key={c.label}>
                      <TableCell className="font-medium">{c.label}</TableCell>
                      <TableCell className="text-right">{pct(c.precision)}</TableCell>
                      <TableCell className="text-right">{pct(c.recall)}</TableCell>
                      <TableCell className="text-right">{pct(c.f1)}</TableCell>
                      <TableCell className="text-right">{c.tp}</TableCell>
                      <TableCell className="text-right">{c.fp}</TableCell>
                      <TableCell className="text-right">{c.fn}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>
        )}

        {/* Doc count */}
        {metrics.total_docs != null && (
          <p className="text-xs text-muted-foreground">
            Evaluated on {String(metrics.total_docs)} documents
          </p>
        )}
      </CardContent>
    </Card>
  );
}
