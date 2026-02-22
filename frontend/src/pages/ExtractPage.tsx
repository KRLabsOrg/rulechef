import { useCallback, useMemo, useRef, useState } from "react";
import { Loader2, Search } from "lucide-react";
import { Link } from "react-router-dom";
import { toast } from "sonner";

import { useAddCorrection, useAddExample, useExtract, useRules } from "@/api/client";
import type { Entity } from "@/api/types";
import {
  entitiesEqual,
  loadStoredLabels,
  normalizeEntities,
} from "@/lib/entities";

import { AnnotatedText } from "@/components/extraction/AnnotatedText";
import { CorrectionToolbar } from "@/components/extraction/CorrectionToolbar";
import { EntityEditor } from "@/components/extraction/EntityEditor";
import { EntityLegend } from "@/components/extraction/EntityLegend";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";

export function ExtractPage() {
  const extract = useExtract();
  const addCorrection = useAddCorrection();
  const addExample = useAddExample();
  const { data: rulesData, isLoading: rulesLoading } = useRules();

  const labels = useMemo(() => loadStoredLabels(), []);

  const [inputText, setInputText] = useState("");
  const [extractedText, setExtractedText] = useState("");
  const [originalEntities, setOriginalEntities] = useState<Entity[]>([]);
  const [editedEntities, setEditedEntities] = useState<Entity[]>([]);
  const [correctionFeedback, setCorrectionFeedback] = useState("");
  const correctionFeedbackRef = useRef("");

  const hasChanges = !entitiesEqual(originalEntities, editedEntities);
  const hasRules = (rulesData?.rules?.length ?? 0) > 0;

  const handleExtract = useCallback(() => {
    if (!inputText.trim()) {
      return;
    }

    extract.mutate(
      { input_data: { text: inputText } },
      {
        onSuccess: (data) => {
          setExtractedText(inputText);
          const entities = normalizeEntities(data.result.entities ?? data.result.spans ?? []);
          setOriginalEntities(entities);
          setEditedEntities(entities);
        },
        onError: (err: Error) => toast.error(err.message),
      }
    );
  }, [extract, inputText]);

  const handleConfirm = useCallback(() => {
    const normalized = normalizeEntities(editedEntities);
    addExample.mutate(
      {
        input_data: { text: extractedText },
        output_data: { entities: normalized },
      },
      {
        onSuccess: () => toast.success("Saved as a correct example."),
        onError: (err: Error) => toast.error(err.message),
      }
    );
  }, [addExample, editedEntities, extractedText]);

  const handleSubmitCorrection = useCallback(() => {
    const normalized = normalizeEntities(editedEntities);

    // Use ref to guarantee we read the latest feedback value
    let feedback = correctionFeedbackRef.current.trim();
    if (!feedback) {
      const added = normalized.filter(
        (e) => !originalEntities.some((o) => o.start === e.start && o.end === e.end && o.type === e.type),
      );
      const removed = originalEntities.filter(
        (o) => !normalized.some((e) => e.start === o.start && e.end === o.end && e.type === o.type),
      );
      const parts: string[] = [];
      if (added.length) parts.push(`Added ${added.map((e) => `${e.type}:"${e.text}"`).join(", ")}`);
      if (removed.length) parts.push(`Removed ${removed.map((e) => `${e.type}:"${e.text}"`).join(", ")}`);
      feedback = parts.join(". ") || "No entity changes";
    }

    addCorrection.mutate(
      {
        input_data: { text: extractedText },
        model_output: { entities: originalEntities },
        expected_output: { entities: normalized },
        feedback,
      },
      {
        onSuccess: () => {
          toast.success("Correction submitted. Re-run learning to improve rules.");
          setOriginalEntities(normalized);
          setCorrectionFeedback("");
          correctionFeedbackRef.current = "";
        },
        onError: (err: Error) => toast.error(err.message),
      }
    );
  }, [addCorrection, editedEntities, extractedText, originalEntities]);

  const clearAllAnnotations = useCallback(() => {
    setEditedEntities([]);
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Extract</h2>
        <p className="text-muted-foreground">
          Run extraction on new text, then fix highlights directly.
        </p>
      </div>

      {!rulesLoading && !hasRules && (
        <Card>
          <CardContent className="py-8 text-center">
            <p className="text-sm text-muted-foreground mb-2">
              No rules learned yet. Extraction works best after you train rules from examples.
            </p>
            <Link
              to="/learn"
              className="text-sm font-medium text-primary underline underline-offset-4 hover:text-primary/80"
            >
              Go to Step 2: Learn Rules
            </Link>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Input Text</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <Textarea
            value={inputText}
            onChange={(event) => setInputText(event.target.value)}
            placeholder="Enter text to extract entities from..."
            rows={4}
          />
          <Button onClick={handleExtract} disabled={extract.isPending || !inputText.trim()}>
            {extract.isPending ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Extracting...
              </>
            ) : (
              <>
                <Search className="h-4 w-4 mr-2" />
                Extract
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {extractedText && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Results</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <EntityLegend labels={labels} />

            <AnnotatedText
              text={extractedText}
              entities={editedEntities}
              labels={labels}
              editable
              onChange={setEditedEntities}
            />

            <EntityEditor
              entities={editedEntities}
              setEntities={setEditedEntities}
              text={extractedText}
              labels={labels}
              disabled={addCorrection.isPending || addExample.isPending}
              showRuleName
            />

            <CorrectionToolbar
              hasChanges={hasChanges}
              onConfirm={handleConfirm}
              onSubmitCorrection={handleSubmitCorrection}
              isSubmitting={addCorrection.isPending || addExample.isPending}
            />

            {editedEntities.length > 0 && (
              <Button
                variant="outline"
                size="sm"
                onClick={clearAllAnnotations}
                disabled={addCorrection.isPending || addExample.isPending}
              >
                Clear All Annotations
              </Button>
            )}

            {hasChanges && (
              <div className="space-y-2">
                <Label htmlFor="correction-feedback">Optional feedback</Label>
                <Textarea
                  id="correction-feedback"
                  rows={2}
                  value={correctionFeedback}
                  onChange={(event) => {
                    setCorrectionFeedback(event.target.value);
                    correctionFeedbackRef.current = event.target.value;
                  }}
                  placeholder="What was wrong? e.g. ORG label too broad for names."
                />
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
