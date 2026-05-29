export type ApprovedSample = {
  sample_id: string;
  sample_path: string;
  image_path: string | null;
  mask_path: string | null;
  width: number | null;
  height: number | null;
  status: string;
  selectable: boolean;
};

export type ApprovedSamplePage = {
  approved_source_root: string;
  page: number;
  page_size: number;
  total_count: number;
  total_pages: number;
  has_prev: boolean;
  has_next: boolean;
  items: ApprovedSample[];
};

export type DatasetVersion = {
  id: number;
  name: string;
  version: string;
  manifest_path: string;
  sample_count: number;
  label: string;
};

export type RegisteredModel = {
  key: string;
  display_name: string;
  default_encoder: string;
  trainer_backend: string;
};

export type RunSummary = {
  run_name: string;
  run_dir: string;
  status: string;
  dataset_version_id: number;
  model_name: string;
  encoder_name: string;
  trainer_backend: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  best_dice: number | null;
};

export type RunDetail = {
  summary: RunSummary;
  dataset_label: string;
  metrics: Array<{ epoch: number; dice: number }>;
  log_text: string;
};

export type CompareRow = {
  run_name: string;
  dataset_version_id: number;
  dataset_label: string;
  model_name: string;
  encoder_name: string;
  best_dice: number;
};

export type TestableRun = {
  run_name: string;
  run_dir: string;
  model_name: string;
  encoder_name: string;
  dataset_version_id: number;
};

export type TestSample = {
  sample_id: string;
  image_path: string;
  mask_path: string;
};

export type InferencePreview = {
  run_name: string;
  model_name: string;
  encoder_name: string;
  dataset_label: string;
  checkpoint_path: string;
  sample_id: string;
  original_data_url: string;
  ground_truth_data_url: string;
  prediction_data_url: string;
  overlay_data_url: string;
};

async function request<T>(input: string, init?: RequestInit): Promise<T> {
  const response = await fetch(input, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `Request failed with status ${response.status}`);
  }
  return (await response.json()) as T;
}

export const api = {
  getApprovedSamples(queryString: string) {
    return request<ApprovedSamplePage>(`/api/datasets/samples${queryString}`);
  },
  createDatasetVersion(payload: { name: string; sample_ids: string[] }) {
    return request<DatasetVersion>("/api/datasets/versions", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },
  getDatasetVersions() {
    return request<DatasetVersion[]>("/api/datasets/versions");
  },
  getModels() {
    return request<RegisteredModel[]>("/api/models");
  },
  getRuns(queryString: string) {
    return request<RunSummary[]>(`/api/runs${queryString}`);
  },
  createRun(payload: {
    dataset_version_id: number;
    model_name: string;
    epochs: number;
    batch_size: number;
    learning_rate: number;
  }) {
    return request<{ run_name: string }>("/api/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },
  getRunDetail(runName: string) {
    return request<RunDetail>(`/api/runs/${encodeURIComponent(runName)}`);
  },
  retryRun(runName: string) {
    return request<{ run_name: string }>(`/api/runs/${encodeURIComponent(runName)}/retry`, {
      method: "POST",
    });
  },
  stopRun(runName: string) {
    return request<{ run_name: string; status: string }>(
      `/api/runs/${encodeURIComponent(runName)}/stop`,
      { method: "POST" },
    );
  },
  getCompareRows(queryString: string) {
    return request<CompareRow[]>(`/api/compare${queryString}`);
  },
  getTestRuns() {
    return request<TestableRun[]>("/api/test/runs");
  },
  getTestSamples(runName: string) {
    return request<TestSample[]>(`/api/test/runs/${encodeURIComponent(runName)}/samples`);
  },
  infer(payload: { run_name: string; sample_id: string }) {
    return request<InferencePreview>("/api/test/infer", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },
};
