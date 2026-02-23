import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type {
  AddCorrectionRequest,
  AddExampleRequest,
  AddFeedbackRequest,
  AddRawLinesRequest,
  AnnotateExampleRequest,
  ConfigureRequest,
  EvalResult,
  ExamplesResponse,
  ExtractRequest,
  ExtractResponse,
  FeedbackResponse,
  LearnRequest,
  LearnStatusResponse,
  RuleMetricsResponse,
  RulesResponse,
  StatusResponse,
} from "./types";

const BASE = "/api";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers ?? {});
  if (!headers.has("Content-Type") && !(init?.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }

  const res = await fetch(`${BASE}${path}`, {
    credentials: "include",
    ...init,
    headers,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

// --- Project ---

export function useProjectStatus() {
  return useQuery<StatusResponse>({
    queryKey: ["project", "status"],
    queryFn: () => apiFetch("/project/status"),
  });
}

export function useConfigure() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: ConfigureRequest) =>
      apiFetch<StatusResponse>("/project/configure", {
        method: "POST",
        body: JSON.stringify(req),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["project"] }),
  });
}

export function useResetProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      apiFetch<StatusResponse>("/project/reset", {
        method: "POST",
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["project"] });
      qc.invalidateQueries({ queryKey: ["data"] });
      qc.invalidateQueries({ queryKey: ["rules"] });
      qc.invalidateQueries({ queryKey: ["learn"] });
    },
  });
}

// --- Data ---

export function useExamples() {
  return useQuery<ExamplesResponse>({
    queryKey: ["data", "examples"],
    queryFn: () => apiFetch("/data/examples"),
  });
}

export function useAddExample() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: AddExampleRequest) =>
      apiFetch("/data/example", { method: "POST", body: JSON.stringify(req) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["data"] });
      qc.invalidateQueries({ queryKey: ["project"] });
    },
  });
}

export function useAddCorrection() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: AddCorrectionRequest) =>
      apiFetch("/data/correction", {
        method: "POST",
        body: JSON.stringify(req),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["data"] });
      qc.invalidateQueries({ queryKey: ["project"] });
    },
  });
}

export function useAddRawLines() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: AddRawLinesRequest) =>
      apiFetch<{ count: number; message: string }>("/data/raw-lines", {
        method: "POST",
        body: JSON.stringify(req),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["data"] });
      qc.invalidateQueries({ queryKey: ["project"] });
    },
  });
}

export function useAnnotateExample() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (args: { exampleId: string; req: AnnotateExampleRequest }) =>
      apiFetch(`/data/example/${args.exampleId}/annotation`, {
        method: "POST",
        body: JSON.stringify(args.req),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["data"] });
      qc.invalidateQueries({ queryKey: ["project"] });
    },
  });
}

export function useDeleteExample() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (exampleId: string) =>
      apiFetch(`/data/example/${exampleId}`, { method: "DELETE" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["data"] });
      qc.invalidateQueries({ queryKey: ["project"] });
    },
  });
}

export function useUploadFile() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (file: File) => {
      const form = new FormData();
      form.append("file", file);
      return apiFetch<{ count: number; message: string }>("/data/upload", {
        method: "POST",
        body: form,
      });
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["data"] });
      qc.invalidateQueries({ queryKey: ["project"] });
    },
  });
}

// --- Learning ---

export function useTriggerLearning() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: LearnRequest = {}) =>
      apiFetch("/learn", { method: "POST", body: JSON.stringify(req) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["learn"] });
      qc.invalidateQueries({ queryKey: ["project"] });
    },
  });
}

export function useLearnStatus(
  polling: boolean,
  onData?: (data: LearnStatusResponse) => void,
) {
  return useQuery<LearnStatusResponse>({
    queryKey: ["learn", "status"],
    queryFn: async () => {
      const data = await apiFetch<LearnStatusResponse>("/learn/status");
      onData?.(data);
      return data;
    },
    refetchInterval: polling ? 2000 : false,
  });
}

// --- Extraction ---

export function useExtract() {
  return useMutation({
    mutationFn: (req: ExtractRequest) =>
      apiFetch<ExtractResponse>("/extract", {
        method: "POST",
        body: JSON.stringify(req),
      }),
  });
}

// --- Rules ---

export function useRules() {
  return useQuery<RulesResponse>({
    queryKey: ["rules"],
    queryFn: () => apiFetch("/rules"),
  });
}

export function useDeleteRule() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (ruleId: string) =>
      apiFetch(`/rules/${ruleId}`, { method: "DELETE" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["rules"] });
      qc.invalidateQueries({ queryKey: ["project"] });
    },
  });
}

export function useRuleMetrics() {
  return useQuery<RuleMetricsResponse>({
    queryKey: ["rules", "metrics"],
    queryFn: () => apiFetch("/rules/metrics"),
  });
}

// --- Evaluation ---

export function useEvaluate() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => apiFetch<EvalResult>("/learn/evaluate"),
    onSuccess: (data) => {
      // Store the eval result in the learn status metrics so MetricsCard picks it up
      qc.setQueryData<LearnStatusResponse>(["learn", "status"], (old) =>
        old ? { ...old, metrics: data as unknown as Record<string, unknown> } : old,
      );
      qc.invalidateQueries({ queryKey: ["rules", "metrics"] });
    },
  });
}

// --- Feedback ---

export function useFeedback() {
  return useQuery<FeedbackResponse>({
    queryKey: ["feedback"],
    queryFn: () => apiFetch("/learn/feedback"),
  });
}

export function useAddFeedback() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: AddFeedbackRequest) =>
      apiFetch("/learn/feedback", {
        method: "POST",
        body: JSON.stringify(req),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["feedback"] });
      qc.invalidateQueries({ queryKey: ["project"] });
    },
  });
}

export function useDeleteFeedback() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (feedbackId: string) =>
      apiFetch(`/learn/feedback/${feedbackId}`, { method: "DELETE" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["feedback"] });
      qc.invalidateQueries({ queryKey: ["project"] });
    },
  });
}
