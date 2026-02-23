// --- Entities ---
export interface Entity {
  text: string;
  start: number;
  end: number;
  type: string;
  score?: number;
  rule_id?: string;
  rule_name?: string;
}

// --- Project ---
export interface ConfigureRequest {
  task_name: string;
  task_description: string;
  task_type: "ner" | "extraction" | "classification" | "transformation";
  input_schema: Record<string, string>;
  output_schema: Record<string, string>;
  entity_labels: string[];
  openai_api_key?: string;
  openai_base_url?: string;
  model: string;
  text_field: string;
  allowed_formats?: string[];
}

export interface StatusResponse {
  configured: boolean;
  task_name?: string;
  task_type?: string;
  stats?: Record<string, unknown>;
}

// --- Data ---
export interface AddExampleRequest {
  input_data: Record<string, unknown>;
  output_data: Record<string, unknown>;
}

export interface AddCorrectionRequest {
  input_data: Record<string, unknown>;
  model_output: Record<string, unknown>;
  expected_output: Record<string, unknown>;
  feedback?: string;
}

export interface AddRawLinesRequest {
  lines: string[];
}

export interface AnnotateExampleRequest {
  output_data: Record<string, unknown>;
}

export interface ExampleItem {
  id: string;
  input: Record<string, unknown>;
  expected_output: Record<string, unknown>;
  source: string;
}

export interface CorrectionItem {
  id: string;
  input: Record<string, unknown>;
  model_output: Record<string, unknown>;
  expected_output: Record<string, unknown>;
  feedback?: string;
}

export interface ExamplesResponse {
  examples: ExampleItem[];
  corrections: CorrectionItem[];
}

// --- Learning ---
export interface LearnRequest {
  sampling_strategy?: string;
  max_iterations?: number;
  incremental_only?: boolean;
  use_agentic?: boolean;
}

export interface LearnStatusResponse {
  running: boolean;
  progress: string;
  error?: string;
  metrics?: Record<string, unknown>;
}

// --- Extraction ---
export interface ExtractRequest {
  input_data: Record<string, unknown>;
}

export interface ExtractResponse {
  result: Record<string, unknown>;
}

// --- Rules ---
export interface RuleSummary {
  id: string;
  name: string;
  description: string;
  format: string;
  priority: number;
  confidence: number;
  times_applied: number;
  success_rate: number;
  content?: string;
}

export interface RulesResponse {
  rules: RuleSummary[];
}

// --- Evaluation ---
export interface ClassMetrics {
  label: string;
  precision: number;
  recall: number;
  f1: number;
  tp: number;
  fp: number;
  fn: number;
}

export interface EvalResult {
  micro_precision: number;
  micro_recall: number;
  micro_f1: number;
  macro_f1: number;
  exact_match: number;
  total_tp: number;
  total_fp: number;
  total_fn: number;
  total_docs: number;
  per_class: ClassMetrics[];
  failures: Record<string, unknown>[];
}

export interface RuleMetricsItem {
  rule_id: string;
  rule_name: string;
  precision: number;
  recall: number;
  f1: number;
  matches: number;
  true_positives: number;
  false_positives: number;
  covered_expected: number;
  total_expected: number;
  per_class: ClassMetrics[];
  sample_matches: Record<string, unknown>[];
}

export interface RuleMetricsResponse {
  rule_metrics: RuleMetricsItem[];
}

// --- Feedback ---
export interface AddFeedbackRequest {
  text: string;
  level: "task" | "example" | "rule";
  target_id?: string;
}

export interface FeedbackItem {
  id: string;
  text: string;
  level: string;
  target_id: string;
  timestamp: string;
}

export interface FeedbackResponse {
  feedback: FeedbackItem[];
}
