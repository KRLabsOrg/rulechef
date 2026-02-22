import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { RotateCcw } from "lucide-react";
import { toast } from "sonner";

import type { Entity, ExampleItem } from "@/api/types";
import {
  useAddExample,
  useAddRawLines,
  useAnnotateExample,
  useExamples,
  useResetProject,
} from "@/api/client";
import {
  LABELS_STORAGE_KEY,
  getExampleText,
  loadStoredLabelsString,
  normalizeEntities,
  parseEntities,
  parseLabels,
} from "@/lib/entities";

import { ExampleTable } from "@/components/data/ExampleTable";
import { FileUpload } from "@/components/data/FileUpload";
import { AnnotatedText } from "@/components/extraction/AnnotatedText";
import { EntityEditor } from "@/components/extraction/EntityEditor";
import { EntityLegend } from "@/components/extraction/EntityLegend";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export function DataPage() {
  const addExample = useAddExample();
  const addRawLines = useAddRawLines();
  const annotateExample = useAnnotateExample();
  const resetProject = useResetProject();
  const { data: examplesData } = useExamples();

  // Labels
  const [labelsInput, setLabelsInput] = useState(loadStoredLabelsString);
  const labels = useMemo(() => parseLabels(labelsInput), [labelsInput]);

  // New annotation flow: paste text → annotate → save
  const [inputText, setInputText] = useState("");
  const [isAnnotating, setIsAnnotating] = useState(false);
  const [newEntities, setNewEntities] = useState<Entity[]>([]);

  // Bulk import
  const [rawLinesText, setRawLinesText] = useState("");

  // Edit dialog for existing examples
  const [editingExample, setEditingExample] = useState<ExampleItem | null>(null);
  const [editEntities, setEditEntities] = useState<Entity[]>([]);

  // Reset confirmation dialog
  const [showResetDialog, setShowResetDialog] = useState(false);

  const hasExamples =
    (examplesData?.examples?.some((ex) => String(ex.source).toLowerCase() !== "draft") ??
      false);
  const editingText = editingExample ? getExampleText(editingExample.input) : "";

  const saveLabels = () => {
    localStorage.setItem(LABELS_STORAGE_KEY, JSON.stringify(labels));
    toast.success("Labels saved");
  };

  // --- New annotation flow ---

  const startAnnotating = () => {
    if (!inputText.trim()) {
      toast.error("Type or paste some text first");
      return;
    }
    setNewEntities([]);
    setIsAnnotating(true);
  };

  const saveNewExample = () => {
    addExample.mutate(
      {
        input_data: { text: inputText.trim() },
        output_data: { entities: normalizeEntities(newEntities) },
      },
      {
        onSuccess: () => {
          toast.success("Example saved with annotations");
          setInputText("");
          setNewEntities([]);
          setIsAnnotating(false);
        },
        onError: (err: Error) => toast.error(err.message),
      }
    );
  };

  const cancelAnnotation = () => {
    setIsAnnotating(false);
    setNewEntities([]);
  };

  // --- Bulk import ---

  const addRawLinesBatch = () => {
    const lines = rawLinesText
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);

    if (lines.length === 0) {
      toast.error("Paste one or more lines");
      return;
    }

    addRawLines.mutate(
      { lines },
      {
        onSuccess: (data) => {
          toast.success(data.message);
          setRawLinesText("");
        },
        onError: (err: Error) => toast.error(err.message),
      }
    );
  };

  // --- Edit existing example (dialog) ---

  const openEditDialog = (example: ExampleItem) => {
    const text = getExampleText(example.input);
    if (!text) {
      toast.error("This example does not contain a text field");
      return;
    }
    setIsAnnotating(false);
    setNewEntities([]);
    setEditingExample(example);
    setEditEntities(parseEntities(example.expected_output));
  };

  const saveEditAnnotation = () => {
    if (!editingExample) return;
    annotateExample.mutate(
      {
        exampleId: editingExample.id,
        req: { output_data: { entities: normalizeEntities(editEntities) } },
      },
      {
        onSuccess: () => {
          toast.success("Annotation updated");
          setEditingExample(null);
          setEditEntities([]);
        },
        onError: (err: Error) => toast.error(err.message),
      }
    );
  };

  // --- Reset ---

  const handleReset = () => {
    resetProject.mutate(undefined, {
      onSuccess: () => {
        toast.success("Workspace reset. You now have a clean dataset.");
        setShowResetDialog(false);
        setInputText("");
        setNewEntities([]);
        setIsAnnotating(false);
        setEditingExample(null);
      },
      onError: (err: Error) => toast.error(err.message),
    });
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Training Data</h2>
          <p className="text-muted-foreground">
            Add examples and label entities by highlighting spans.
          </p>
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="text-muted-foreground"
          onClick={() => setShowResetDialog(true)}
        >
          <RotateCcw className="h-3.5 w-3.5 mr-1.5" />
          Reset
        </Button>
      </div>

      {/* Entity Labels (compact) */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Entity Labels</CardTitle>
          <CardDescription>
            Labels used by the annotation picker. Edit and save.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-center gap-2">
            <Label htmlFor="labels" className="sr-only">Labels</Label>
            <Input
              id="labels"
              value={labelsInput}
              onChange={(event) => setLabelsInput(event.target.value)}
              onBlur={saveLabels}
              placeholder="PER, ORG, LOC, DATE, MISC"
              className="flex-1"
            />
            <Button variant="outline" size="sm" onClick={saveLabels}>
              Save
            </Button>
          </div>
          <EntityLegend labels={labels} />
        </CardContent>
      </Card>

      {/* Add Examples */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Add Examples</CardTitle>
          <CardDescription>
            Annotate a single text or import many drafts at once.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="annotate" className="gap-4">
            <TabsList>
              <TabsTrigger value="annotate">Annotate Text</TabsTrigger>
              <TabsTrigger value="bulk">Bulk Import</TabsTrigger>
            </TabsList>

            <TabsContent value="annotate" className="space-y-4">
              {!isAnnotating ? (
                <>
                  <Textarea
                    value={inputText}
                    onChange={(event) => setInputText(event.target.value)}
                    rows={4}
                    placeholder="Type or paste a sentence, then click 'Start Annotating' to highlight entities..."
                  />
                  <Button
                    onClick={startAnnotating}
                    disabled={!inputText.trim()}
                  >
                    Start Annotating
                  </Button>
                </>
              ) : (
                <>
                  <EntityLegend labels={labels} />
                  <AnnotatedText
                    text={inputText.trim()}
                    entities={newEntities}
                    labels={labels}
                    editable
                    onChange={setNewEntities}
                  />
                  <p className="text-xs text-muted-foreground">
                    Select text to add an entity. Click a highlight to change its label or remove it.
                  </p>
                  <EntityEditor
                    entities={newEntities}
                    setEntities={setNewEntities}
                    text={inputText.trim()}
                    labels={labels}
                    disabled={addExample.isPending}
                  />
                  <div className="flex items-center gap-2">
                    <Button
                      onClick={saveNewExample}
                      disabled={addExample.isPending}
                    >
                      {addExample.isPending ? "Saving..." : "Save Example"}
                    </Button>
                    <Button variant="outline" onClick={cancelAnnotation}>
                      Cancel
                    </Button>
                  </div>
                </>
              )}
            </TabsContent>

            <TabsContent value="bulk" className="space-y-4">
              <Textarea
                value={rawLinesText}
                onChange={(event) => setRawLinesText(event.target.value)}
                rows={6}
                placeholder={"One example per line:\nKR Labs is a company.\nAcme Corp opened an office in Berlin.\nOpenAI released a new model."}
              />
              <Button onClick={addRawLinesBatch} disabled={addRawLines.isPending}>
                {addRawLines.isPending ? "Saving..." : "Add Lines as Drafts"}
              </Button>
              <div className="border-t pt-4">
                <p className="text-xs text-muted-foreground mb-2">
                  Or upload a file: <code>.txt</code> for raw lines, <code>.json</code>/<code>.jsonl</code>/<code>.csv</code> for pre-annotated data.
                </p>
                <FileUpload accept=".txt,.csv,.json,.jsonl" prompt="Drag and drop a file, or" />
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Training Data table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Training Data</CardTitle>
        </CardHeader>
        <CardContent>
          <ExampleTable onAnnotate={openEditDialog} />
        </CardContent>
      </Card>

      {/* Next step prompt */}
      {hasExamples && (
        <div className="rounded-lg border border-primary/20 bg-primary/5 px-4 py-3 text-sm">
          You have training examples.{" "}
          <Link
            to="/learn"
            className="font-medium text-primary underline underline-offset-4 hover:text-primary/80"
          >
            Next: Learn rules from your data
          </Link>
        </div>
      )}

      {/* Edit annotation dialog */}
      <Dialog
        open={!!editingExample}
        onOpenChange={(open) => {
          if (!open) {
            setEditingExample(null);
            setEditEntities([]);
          }
        }}
      >
        <DialogContent className="sm:max-w-2xl bg-white dark:bg-slate-900">
          <DialogHeader>
            <DialogTitle>Edit Annotation</DialogTitle>
            <DialogDescription>
              Select text to add entities. Click a highlight to change or remove it.
            </DialogDescription>
          </DialogHeader>
          {editingExample && (
            <div className="space-y-3">
              <EntityLegend labels={labels} />
              <AnnotatedText
                text={editingText}
                entities={editEntities}
                labels={labels}
                editable
                onChange={setEditEntities}
              />
              <EntityEditor
                entities={editEntities}
                setEntities={setEditEntities}
                text={editingText}
                labels={labels}
                disabled={annotateExample.isPending}
              />
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditingExample(null)}>
              Cancel
            </Button>
            <Button onClick={saveEditAnnotation} disabled={annotateExample.isPending}>
              {annotateExample.isPending ? "Saving..." : "Save"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Reset confirmation dialog */}
      <Dialog open={showResetDialog} onOpenChange={setShowResetDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Reset Workspace?</DialogTitle>
            <DialogDescription>
              This will delete all examples, corrections, and learned rules. This cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowResetDialog(false)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleReset}
              disabled={resetProject.isPending}
            >
              {resetProject.isPending ? "Resetting..." : "Reset Everything"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
