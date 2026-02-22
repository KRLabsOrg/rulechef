import { useState } from "react";
import { Link } from "react-router-dom";
import { Send, X } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  useProjectStatus,
  useRules,
  useAddFeedback,
  useFeedback,
  useDeleteFeedback,
} from "@/api/client";
import { LearnButton } from "@/components/learning/LearnButton";
import { MetricsCard } from "@/components/learning/MetricsCard";
import { RulesTable } from "@/components/learning/RulesTable";
import { toast } from "sonner";

export function LearnPage() {
  const status = useProjectStatus();
  const { data: rulesData } = useRules();
  const hasRules = (rulesData?.rules?.length ?? 0) > 0;
  const addFeedback = useAddFeedback();
  const deleteFeedback = useDeleteFeedback();
  const { data: feedbackData } = useFeedback();
  const [taskFeedback, setTaskFeedback] = useState("");

  const submitTaskFeedback = () => {
    const text = taskFeedback.trim();
    if (!text) return;
    addFeedback.mutate(
      { text, level: "task" },
      {
        onSuccess: () => {
          toast.success("Feedback saved");
          setTaskFeedback("");
        },
        onError: (err: Error) => toast.error(err.message),
      },
    );
  };

  const taskFeedbackItems = (feedbackData?.feedback ?? []).filter(
    (f) => f.level === "task",
  );

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Learn Rules</h2>
        <p className="text-muted-foreground">
          Train from your examples and corrections. No technical setup required.
        </p>
      </div>

      {/* Learning controls + workspace stats in one row */}
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Learning</CardTitle>
          </CardHeader>
          <CardContent>
            <LearnButton />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Workspace</CardTitle>
          </CardHeader>
          <CardContent>
            {status.data?.stats ? (
              <div className="grid grid-cols-3 gap-3">
                {Object.entries(status.data.stats)
                  .filter(([k]) =>
                    [
                      "examples",
                      "corrections",
                      "rules",
                      "pending_examples",
                      "pending_corrections",
                      "feedback",
                    ].includes(k)
                  )
                  .map(([key, value]) => (
                    <div key={key}>
                      <p className="text-xs text-muted-foreground capitalize">
                        {key.replace(/_/g, " ")}
                      </p>
                      <p className="text-lg font-bold">{String(value)}</p>
                    </div>
                  ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                Preparing workspace...
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Metrics - full width */}
      <MetricsCard />

      {/* Rules table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Learned Rules</CardTitle>
          <p className="text-xs text-muted-foreground">
            Click a rule to see details, per-class metrics, and give feedback.
            Feedback is used in the next learning run.
          </p>
        </CardHeader>
        <CardContent>
          <RulesTable />
        </CardContent>
      </Card>

      {/* Task-level guidance — below rules, less prominent */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Task Guidance</CardTitle>
          <p className="text-xs text-muted-foreground">
            High-level instructions for the learner (e.g. "always extract
            dosages with units", "ignore brand names").
          </p>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="flex gap-2">
            <Input
              placeholder="Add guidance..."
              className="h-8 text-sm"
              value={taskFeedback}
              onChange={(e) => setTaskFeedback(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") submitTaskFeedback();
              }}
            />
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8 shrink-0"
              onClick={submitTaskFeedback}
            >
              <Send className="h-3 w-3" />
            </Button>
          </div>
          {taskFeedbackItems.length > 0 && (
            <div className="space-y-1">
              {taskFeedbackItems.map((f) => (
                <div
                  key={f.id}
                  className="flex items-center gap-1 text-sm text-muted-foreground"
                >
                  <span className="flex-1">{f.text}</span>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-5 w-5 shrink-0"
                    onClick={() => deleteFeedback.mutate(f.id)}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {hasRules && (
        <div className="rounded-lg border border-primary/20 bg-primary/5 px-4 py-3 text-sm">
          Rules are ready!{" "}
          <Link
            to="/extract"
            className="font-medium text-primary underline underline-offset-4 hover:text-primary/80"
          >
            Next: Test extraction on new text
          </Link>
        </div>
      )}
    </div>
  );
}
