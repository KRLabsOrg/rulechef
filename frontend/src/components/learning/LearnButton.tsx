import { useRef, useState } from "react";
import { Loader2, Play, BarChart3, ChevronRight, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTriggerLearning, useLearnStatus, useEvaluate } from "@/api/client";
import { toast } from "sonner";
import type { LearnStatusResponse } from "@/api/types";
import { useQueryClient } from "@tanstack/react-query";

export function LearnButton() {
  const [polling, setPolling] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [maxIterations, setMaxIterations] = useState(3);
  const [incrementalOnly, setIncrementalOnly] = useState(true);
  const [useAgentic, setUseAgentic] = useState(false);
  const trigger = useTriggerLearning();
  const evaluate = useEvaluate();
  const prevRunning = useRef(false);
  const qc = useQueryClient();

  const status = useLearnStatus(polling, (data: LearnStatusResponse) => {
    if (prevRunning.current && !data.running) {
      setPolling(false);
      qc.invalidateQueries({ queryKey: ["rules"] });
      qc.invalidateQueries({ queryKey: ["project"] });
      qc.invalidateQueries({ queryKey: ["data"] });
      qc.invalidateQueries({ queryKey: ["learn"] });
      if (data.error) {
        toast.error("Learning failed: " + data.error);
      } else if (data.progress) {
        toast.success(data.progress);
      }
    }
    prevRunning.current = data.running;
  });

  const isRunning = status.data?.running ?? false;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Button
          onClick={() => {
            trigger.mutate(
              {
                max_iterations: maxIterations,
                incremental_only: incrementalOnly,
                use_agentic: useAgentic,
              },
              {
                onSuccess: () => {
                  prevRunning.current = true;
                  setPolling(true);
                },
                onError: (err: Error) => toast.error(err.message),
              }
            );
          }}
          disabled={isRunning || trigger.isPending}
        >
          {isRunning ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Learning...
            </>
          ) : (
            <>
              <Play className="h-4 w-4 mr-2" />
              Learn Rules
            </>
          )}
        </Button>
        <Button
          variant="outline"
          onClick={() => {
            evaluate.mutate(undefined, {
              onSuccess: () => {
                qc.invalidateQueries({ queryKey: ["rules", "metrics"] });
                toast.success("Evaluation complete");
              },
              onError: (err: Error) => toast.error(err.message),
            });
          }}
          disabled={isRunning || evaluate.isPending}
        >
          {evaluate.isPending ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Evaluating...
            </>
          ) : (
            <>
              <BarChart3 className="h-4 w-4 mr-2" />
              Evaluate
            </>
          )}
        </Button>
      </div>

      <button
        type="button"
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        {showAdvanced ? (
          <ChevronDown className="h-3 w-3" />
        ) : (
          <ChevronRight className="h-3 w-3" />
        )}
        Advanced
      </button>

      {showAdvanced && (
        <div className="space-y-2 pl-4 text-sm">
          <div className="flex items-center gap-2">
            <label htmlFor="max-iterations" className="text-muted-foreground">
              Iterations:
            </label>
            <select
              id="max-iterations"
              value={maxIterations}
              onChange={(e) => setMaxIterations(Number(e.target.value))}
              className="rounded border border-input bg-background px-2 py-1 text-sm"
            >
              {Array.from({ length: 15 }, (_, i) => i + 1).map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </div>
          <label className="flex items-center gap-2 text-muted-foreground">
            <input
              type="checkbox"
              checked={incrementalOnly}
              onChange={(e) => setIncrementalOnly(e.target.checked)}
            />
            Incremental only
          </label>
          <label className="flex items-center gap-2 text-muted-foreground">
            <input
              type="checkbox"
              checked={useAgentic}
              onChange={(e) => setUseAgentic(e.target.checked)}
            />
            Agentic coordinator
          </label>
        </div>
      )}

      {isRunning && status.data?.progress && (
        <p className="text-sm text-muted-foreground">{status.data.progress}</p>
      )}
    </div>
  );
}
