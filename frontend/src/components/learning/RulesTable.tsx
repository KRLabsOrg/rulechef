import { Fragment, useState } from "react";
import { Trash2, ChevronDown, ChevronRight, ChevronLeft, Send, X } from "lucide-react";
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
import { Input } from "@/components/ui/input";
import {
  useRules,
  useDeleteRule,
  useRuleMetrics,
  useAddFeedback,
  useFeedback,
  useDeleteFeedback,
} from "@/api/client";
import { toast } from "sonner";
import type { RuleMetricsItem } from "@/api/types";

// --- Sample Matches Browser ---

interface SpanItem {
  text: string;
  start: number;
  end: number;
  type?: string;
  label?: string;
}

type SegmentKind = "tp" | "fp" | "fn" | "plain";

interface Segment {
  text: string;
  kind: SegmentKind;
  label?: string;
}

function getSpanType(s: SpanItem): string {
  return (s.type ?? s.label ?? "").toUpperCase();
}

function classifySpans(
  ruleOutput: SpanItem[],
  expected: SpanItem[],
): { tp: SpanItem[]; fp: SpanItem[]; fn: SpanItem[] } {
  const matched = new Set<number>();
  const tp: SpanItem[] = [];
  const fp: SpanItem[] = [];

  for (const ro of ruleOutput) {
    // Match by text + type, same logic as the backend _entity_key_text.
    // Overlap-based matching disagrees with backend counts and can show
    // a span as green/TP even when the extracted text is wrong.
    const matchIdx = expected.findIndex(
      (ex, i) =>
        !matched.has(i) &&
        getSpanType(ro) === getSpanType(ex) &&
        ro.text === ex.text,
    );
    if (matchIdx >= 0) {
      matched.add(matchIdx);
      tp.push(ro);
    } else {
      fp.push(ro);
    }
  }

  const fn = expected.filter((_, i) => !matched.has(i));
  return { tp, fp, fn };
}

function buildSegments(text: string, tp: SpanItem[], fp: SpanItem[], fn: SpanItem[]): Segment[] {
  const markers: { pos: number; kind: SegmentKind; isStart: boolean; label: string }[] = [];

  for (const s of tp) {
    markers.push({ pos: s.start, kind: "tp", isStart: true, label: getSpanType(s) });
    markers.push({ pos: s.end, kind: "tp", isStart: false, label: getSpanType(s) });
  }
  for (const s of fp) {
    markers.push({ pos: s.start, kind: "fp", isStart: true, label: getSpanType(s) });
    markers.push({ pos: s.end, kind: "fp", isStart: false, label: getSpanType(s) });
  }
  for (const s of fn) {
    markers.push({ pos: s.start, kind: "fn", isStart: true, label: getSpanType(s) });
    markers.push({ pos: s.end, kind: "fn", isStart: false, label: getSpanType(s) });
  }

  markers.sort((a, b) => a.pos - b.pos || (a.isStart ? 0 : 1) - (b.isStart ? 0 : 1));

  const segments: Segment[] = [];
  let cursor = 0;
  const activeStack: { kind: SegmentKind; label: string }[] = [];

  for (const m of markers) {
    const pos = Math.min(m.pos, text.length);
    if (pos > cursor) {
      const current = activeStack.length > 0 ? activeStack[activeStack.length - 1] : null;
      segments.push({
        text: text.slice(cursor, pos),
        kind: current?.kind ?? "plain",
        label: current?.label,
      });
      cursor = pos;
    }
    if (m.isStart) {
      activeStack.push({ kind: m.kind, label: m.label });
    } else {
      // Remove the matching start from stack
      for (let i = activeStack.length - 1; i >= 0; i--) {
        if (activeStack[i].kind === m.kind && activeStack[i].label === m.label) {
          activeStack.splice(i, 1);
          break;
        }
      }
    }
  }

  if (cursor < text.length) {
    segments.push({ text: text.slice(cursor), kind: "plain" });
  }

  return segments;
}

const SEGMENT_STYLES: Record<SegmentKind, string> = {
  tp: "bg-emerald-100 text-emerald-900 border-b-2 border-emerald-400 rounded-sm px-0.5",
  fp: "bg-red-100 text-red-900 border-b-2 border-red-400 rounded-sm px-0.5",
  fn: "border-b-2 border-dashed border-amber-400 text-amber-800 px-0.5",
  plain: "",
};

function SampleMatchesBrowser({ samples }: { samples: Record<string, unknown>[] }) {
  const [idx, setIdx] = useState(0);
  const sample = samples[idx];
  if (!sample) return null;

  const inputData = sample.input as Record<string, unknown> | undefined;
  const text = typeof inputData?.text === "string"
    ? inputData.text
    : Object.values(inputData ?? {}).find((v) => typeof v === "string") as string ?? "";

  const ruleOutput = (sample.rule_output ?? []) as SpanItem[];
  const expected = (sample.expected ?? []) as SpanItem[];

  const { tp, fp, fn } = classifySpans(ruleOutput, expected);
  const segments = buildSegments(text, tp, fp, fn);
  // Use frontend-computed counts so TP/FP/FN in the footer are consistent
  // with the span colors shown above. (Backend counts use the same text+type
  // matching logic; mixing them with frontend-computed FN caused mismatches.)
  const displayTp = tp.length;
  const displayFp = fp.length;
  const displayFn = fn.length;

  return (
    <div
      className="rounded-md border border-slate-200 bg-background"
      onClick={(e) => e.stopPropagation()}
    >
      {/* Header bar */}
      <div className="flex items-center justify-between border-b border-slate-200 px-3 py-1.5">
        <p className="text-[11px] font-medium tracking-wide text-muted-foreground uppercase">
          Eval Samples
        </p>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            disabled={idx === 0}
            onClick={() => setIdx((i) => i - 1)}
          >
            <ChevronLeft className="h-3.5 w-3.5" />
          </Button>
          <span className="text-xs tabular-nums text-muted-foreground min-w-[3.5rem] text-center">
            {idx + 1} / {samples.length}
          </span>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            disabled={idx === samples.length - 1}
            onClick={() => setIdx((i) => i + 1)}
          >
            <ChevronRight className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {/* Annotated text */}
      <div className="px-3 py-2.5 text-sm leading-relaxed">
        {segments.map((seg, i) =>
          seg.kind === "plain" ? (
            <span key={i}>{seg.text}</span>
          ) : (
            <span key={i} className={SEGMENT_STYLES[seg.kind]} title={`${seg.kind.toUpperCase()}: ${seg.label ?? ""}`}>
              {seg.text}
            </span>
          ),
        )}
      </div>

      {/* Footer: legend + counts */}
      <div className="flex items-center justify-between border-t border-slate-100 px-3 py-1.5">
        <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full bg-emerald-400" /> TP
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full bg-red-400" /> FP
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-3 border-b-2 border-dashed border-amber-400" /> Missed
          </span>
        </div>
        <div className="flex items-center gap-2 text-[10px] tabular-nums">
          <span className="text-emerald-600 font-medium">{displayTp} TP</span>
          <span className="text-red-600 font-medium">{displayFp} FP</span>
          {displayFn > 0 && <span className="text-amber-600 font-medium">{displayFn} FN</span>}
        </div>
      </div>
    </div>
  );
}

function pct(v: number) {
  return `${(v * 100).toFixed(0)}%`;
}

function healthColor(f1: number): string {
  if (f1 >= 0.7) return "text-emerald-600";
  if (f1 >= 0.4) return "text-amber-600";
  return "text-red-600";
}

function healthBg(f1: number): string {
  if (f1 >= 0.7) return "bg-emerald-500/10 text-emerald-700 border-emerald-200";
  if (f1 >= 0.4) return "bg-amber-500/10 text-amber-700 border-amber-200";
  return "bg-red-500/10 text-red-700 border-red-200";
}

const QUICK_FEEDBACK = [
  "Too broad - matches things it shouldn't",
  "Too narrow - misses valid matches",
  "Wrong entity type assigned",
  "Pattern is correct but needs refinement",
];

export function RulesTable() {
  const { data, isLoading } = useRules();
  const { data: metricsData } = useRuleMetrics();
  const { data: feedbackData } = useFeedback();
  const deleteRule = useDeleteRule();
  const addFeedback = useAddFeedback();
  const deleteFeedback = useDeleteFeedback();
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [feedbackText, setFeedbackText] = useState<Record<string, string>>({});

  if (isLoading)
    return <p className="text-sm text-muted-foreground">Loading rules...</p>;

  const rules = data?.rules ?? [];

  if (rules.length === 0) {
    return (
      <p className="text-sm text-muted-foreground py-4 text-center">
        No rules yet. Add examples and run learning.
      </p>
    );
  }

  const metricsMap = new Map<string, RuleMetricsItem>();
  for (const rm of metricsData?.rule_metrics ?? []) {
    metricsMap.set(rm.rule_id, rm);
  }

  const toggle = (id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const submitFeedback = (ruleId: string, text?: string) => {
    const fbText = text ?? feedbackText[ruleId]?.trim();
    if (!fbText) return;
    addFeedback.mutate(
      { text: fbText, level: "rule", target_id: ruleId },
      {
        onSuccess: () => {
          toast.success("Feedback saved");
          if (!text) setFeedbackText((prev) => ({ ...prev, [ruleId]: "" }));
        },
        onError: (err: Error) => toast.error(err.message),
      },
    );
  };

  const ruleFeedback = (ruleId: string) =>
    (feedbackData?.feedback ?? []).filter(
      (f) => f.level === "rule" && f.target_id === ruleId,
    );

  return (
    <div>
      <div className="border rounded-md">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-8" />
              <TableHead>Name</TableHead>
              <TableHead>Format</TableHead>
              <TableHead className="text-right">Prec</TableHead>
              <TableHead className="text-right">Rec</TableHead>
              <TableHead className="text-right">F1</TableHead>
              <TableHead className="text-right">TP/FP</TableHead>
              <TableHead className="w-12" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {rules.map((rule) => {
              const rm = metricsMap.get(rule.id);
              const fb = ruleFeedback(rule.id);
              return (
                <Fragment key={rule.id}>
                  <TableRow
                    className="cursor-pointer"
                    onClick={() => toggle(rule.id)}
                  >
                    <TableCell>
                      {expanded.has(rule.id) ? (
                        <ChevronDown className="h-4 w-4" />
                      ) : (
                        <ChevronRight className="h-4 w-4" />
                      )}
                    </TableCell>
                    <TableCell className="font-medium">
                      <span className="flex items-center gap-2">
                        {rule.name}
                        {fb.length > 0 && (
                          <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                            {fb.length} feedback
                          </Badge>
                        )}
                      </span>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline">{rule.format}</Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      {rm ? (
                        <span className={healthColor(rm.precision)}>
                          {pct(rm.precision)}
                        </span>
                      ) : "—"}
                    </TableCell>
                    <TableCell className="text-right">
                      {rm ? (
                        <span className={healthColor(rm.recall)}>
                          {pct(rm.recall)}
                        </span>
                      ) : "—"}
                    </TableCell>
                    <TableCell className="text-right">
                      {rm ? (
                        <Badge
                          variant="outline"
                          className={`font-mono text-xs ${healthBg(rm.f1)}`}
                        >
                          {pct(rm.f1)}
                        </Badge>
                      ) : "—"}
                    </TableCell>
                    <TableCell className="text-right text-xs text-muted-foreground">
                      {rm ? `${rm.true_positives}/${rm.false_positives}` : "—"}
                    </TableCell>
                    <TableCell>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteRule.mutate(rule.id, {
                            onSuccess: () => toast.success("Rule deleted"),
                            onError: (err: Error) => toast.error(err.message),
                          });
                        }}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </TableCell>
                  </TableRow>
                  {expanded.has(rule.id) && (
                    <TableRow key={`${rule.id}-detail`}>
                      <TableCell colSpan={8} className="bg-muted/50">
                        <div className="p-3 space-y-3">
                          <p className="text-sm">{rule.description}</p>
                          {rule.content && (
                            <pre className="text-xs bg-background p-2 rounded border whitespace-pre-wrap break-all">
                              {rule.content}
                            </pre>
                          )}

                          {/* Per-class breakdown */}
                          {rm && rm.per_class.length > 0 && (
                            <div className="flex flex-wrap gap-2">
                              {rm.per_class.map((c) => (
                                <Badge
                                  key={c.label}
                                  variant="outline"
                                  className={`text-xs ${healthBg(c.f1)}`}
                                >
                                  {c.label}: F1 {pct(c.f1)}
                                </Badge>
                              ))}
                            </div>
                          )}

                          {/* Sample matches browser */}
                          {rm && rm.sample_matches && rm.sample_matches.length > 0 && (
                            <SampleMatchesBrowser samples={rm.sample_matches} />
                          )}

                          {/* Existing feedback */}
                          {fb.length > 0 && (
                            <div className="space-y-1">
                              <p className="text-xs text-muted-foreground font-medium">Feedback:</p>
                              {fb.map((f) => (
                                <div
                                  key={f.id}
                                  className="flex items-center gap-1 text-sm text-muted-foreground"
                                >
                                  <span className="flex-1">{f.text}</span>
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    className="h-5 w-5 shrink-0"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      deleteFeedback.mutate(f.id);
                                    }}
                                  >
                                    <X className="h-3 w-3" />
                                  </Button>
                                </div>
                              ))}
                            </div>
                          )}

                          {/* Quick feedback chips */}
                          <div className="flex flex-wrap gap-1.5">
                            {QUICK_FEEDBACK.map((text) => (
                              <button
                                key={text}
                                className="text-xs px-2 py-1 rounded-full border border-dashed hover:bg-muted transition-colors"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  submitFeedback(rule.id, text);
                                }}
                              >
                                {text}
                              </button>
                            ))}
                          </div>

                          {/* Custom feedback input */}
                          <div className="flex gap-2 items-center">
                            <Input
                              placeholder="Custom feedback..."
                              className="text-sm h-8"
                              value={feedbackText[rule.id] ?? ""}
                              onChange={(e) =>
                                setFeedbackText((prev) => ({
                                  ...prev,
                                  [rule.id]: e.target.value,
                                }))
                              }
                              onKeyDown={(e) => {
                                if (e.key === "Enter") submitFeedback(rule.id);
                              }}
                              onClick={(e) => e.stopPropagation()}
                            />
                            <Button
                              variant="outline"
                              size="icon"
                              className="h-8 w-8 shrink-0"
                              onClick={(e) => {
                                e.stopPropagation();
                                submitFeedback(rule.id);
                              }}
                            >
                              <Send className="h-3 w-3" />
                            </Button>
                          </div>
                        </div>
                      </TableCell>
                    </TableRow>
                  )}
                </Fragment>
              );
            })}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
