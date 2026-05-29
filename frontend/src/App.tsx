import { FormEvent, ReactNode, useDeferredValue, useEffect, useState } from "react";
import {
  BrowserRouter,
  Link,
  NavLink,
  Navigate,
  Route,
  Routes,
  useNavigate,
  useParams,
} from "react-router-dom";

import {
  api,
  ApprovedSample,
  ApprovedSamplePage,
  CompareRow,
  DatasetVersion,
  InferencePreview,
  RegisteredModel,
  RunDetail,
  RunSummary,
  TestSample,
  TestableRun,
} from "./api";

type NavItem = {
  label: string;
  to: string;
  hint: string;
};

const navigation: NavItem[] = [
  { label: "Datasets", to: "/datasets", hint: "Scan and version" },
  { label: "Train", to: "/train", hint: "Launch runs" },
  { label: "Runs", to: "/runs", hint: "Inspect metrics" },
  { label: "Compare", to: "/compare", hint: "Rank checkpoints" },
  { label: "Test", to: "/test", hint: "Preview masks" },
];

function App() {
  return (
    <BrowserRouter basename="/app">
      <DashboardShell>
        <Routes>
          <Route path="/" element={<Navigate to="/datasets" replace />} />
          <Route path="/datasets" element={<DatasetsPage />} />
          <Route path="/train" element={<TrainPage />} />
          <Route path="/runs" element={<RunsPage />} />
          <Route path="/runs/:runName" element={<RunDetailPage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="/test" element={<TestPage />} />
        </Routes>
      </DashboardShell>
    </BrowserRouter>
  );
}

function DashboardShell({ children }: { children: ReactNode }) {
  return (
    <div className="dashboard-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">TM</div>
          <div>
            <div className="brand-title">Training Manager</div>
            <div className="brand-subtitle">Model training and review</div>
          </div>
        </div>

        <nav className="sidebar-nav" aria-label="Primary">
          {navigation.map((item) => (
            <NavLink
              key={item.label}
              to={item.to}
              className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
            >
              <span>{item.label}</span>
              <small>{item.hint}</small>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <span className="status-dot" />
          <div>
            <strong>React dashboard</strong>
            <p>FastAPI-backed control room</p>
          </div>
        </div>
      </aside>

      <main className="workspace">{children}</main>
    </div>
  );
}

function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow: string;
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {actions ? <div className="header-actions">{actions}</div> : null}
    </header>
  );
}

function SectionCard({
  title,
  description,
  children,
  actions,
}: {
  title: string;
  description?: string;
  children: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <section className="section-card">
      <div className="section-card-header">
        <div>
          <h2>{title}</h2>
          {description ? <p>{description}</p> : null}
        </div>
        {actions ? <div className="section-card-actions">{actions}</div> : null}
      </div>
      {children}
    </section>
  );
}

function PaginationControls({
  page,
  totalPages,
  onPageChange,
}: {
  page: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}) {
  if (totalPages <= 1) {
    return null;
  }
  return (
    <div className="pagination">
      <button
        type="button"
        className="ghost-button"
        disabled={page <= 1}
        onClick={() => onPageChange(page - 1)}
      >
        Previous
      </button>
      <span>
        Page {page} / {totalPages}
      </span>
      <button
        type="button"
        className="ghost-button"
        disabled={page >= totalPages}
        onClick={() => onPageChange(page + 1)}
      >
        Next
      </button>
    </div>
  );
}

function StatusBadge({ value }: { value: string }) {
  return <span className={`status-badge status-${value}`}>{value}</span>;
}

function DatasetsPage() {
  const [query, setQuery] = useState("");
  const [resolutionBucket, setResolutionBucket] = useState("");
  const [status, setStatus] = useState("");
  const [sort, setSort] = useState("sample_id");
  const [widthMin, setWidthMin] = useState("");
  const [widthMax, setWidthMax] = useState("");
  const [heightMin, setHeightMin] = useState("");
  const [heightMax, setHeightMax] = useState("");
  const [pageSize, setPageSize] = useState(25);
  const [page, setPage] = useState(1);
  const [datasetName, setDatasetName] = useState("");
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [datasetPage, setDatasetPage] = useState<ApprovedSamplePage | null>(null);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const deferredQuery = useDeferredValue(query);

  const fetchPage = async (nextPage: number, nextPageSize: number) => {
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams();
      if (deferredQuery) params.set("query", deferredQuery);
      if (resolutionBucket) params.set("resolution_bucket", resolutionBucket);
      if (status) params.set("status", status);
      if (sort) params.set("sort", sort);
      if (widthMin) params.set("width_min", widthMin);
      if (widthMax) params.set("width_max", widthMax);
      if (heightMin) params.set("height_min", heightMin);
      if (heightMax) params.set("height_max", heightMax);
      params.set("page", String(nextPage));
      params.set("page_size", String(nextPageSize));
      const payload = await api.getApprovedSamples(`?${params.toString()}`);
      setDatasetPage(payload);
    } catch (requestError) {
      setError((requestError as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setPage(1);
  }, [deferredQuery, resolutionBucket, status, sort, widthMin, widthMax, heightMin, heightMax, pageSize]);

  useEffect(() => {
    void fetchPage(page, pageSize);
  }, [page, pageSize, deferredQuery, resolutionBucket, status, sort, widthMin, widthMax, heightMin, heightMax]);

  const visibleIds = new Set((datasetPage?.items ?? []).map((item) => item.sample_id));
  const allVisibleSelected =
    datasetPage?.items.filter((item) => item.selectable).every((item) => selectedIds.includes(item.sample_id)) ??
    false;

  const togglePageSelection = () => {
    const selectableIds = (datasetPage?.items ?? [])
      .filter((item) => item.selectable)
      .map((item) => item.sample_id);
    if (allVisibleSelected) {
      setSelectedIds((current) => current.filter((sampleId) => !visibleIds.has(sampleId)));
      return;
    }
    setSelectedIds((current) => Array.from(new Set([...current, ...selectableIds])));
  };

  const selectAllFiltered = async () => {
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams();
      if (deferredQuery) params.set("query", deferredQuery);
      if (resolutionBucket) params.set("resolution_bucket", resolutionBucket);
      if (status) params.set("status", status);
      if (sort) params.set("sort", sort);
      if (widthMin) params.set("width_min", widthMin);
      if (widthMax) params.set("width_max", widthMax);
      if (heightMin) params.set("height_min", heightMin);
      if (heightMax) params.set("height_max", heightMax);
      params.set("page", "1");
      params.set("page_size", "5000");
      const payload = await api.getApprovedSamples(`?${params.toString()}`);
      setSelectedIds(payload.items.filter((item) => item.selectable).map((item) => item.sample_id));
    } catch (requestError) {
      setError((requestError as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const submitDataset = async (event: FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    setMessage("");
    try {
      const result = await api.createDatasetVersion({
        name: datasetName,
        sample_ids: selectedIds,
      });
      setMessage(`Created ${result.label} with ${result.sample_count} samples.`);
      setDatasetName("");
      setSelectedIds([]);
    } catch (requestError) {
      setError((requestError as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Datasets"
        title="Version the approved annotation pool."
        description="Filter by resolution, inspect sample health, and create clean manifest versions without losing the source structure."
        actions={
          <button type="button" className="ghost-button" onClick={() => void fetchPage(page, pageSize)}>
            Rescan
          </button>
        }
      />

      <div className="two-column-grid">
        <SectionCard
          title="Create dataset version"
          description={datasetPage ? `Approved source: ${datasetPage.approved_source_root}` : "Loading source root..."}
        >
          <form className="stack-form" onSubmit={(event) => void submitDataset(event)}>
            <div className="form-grid compact">
              <label>
                <span>Dataset name</span>
                <input value={datasetName} onChange={(event) => setDatasetName(event.target.value)} placeholder="baseline" />
              </label>
              <label>
                <span>Selected samples</span>
                <input value={String(selectedIds.length)} disabled />
              </label>
            </div>
            <div className="action-row">
              <button type="button" className="ghost-button" onClick={togglePageSelection}>
                {allVisibleSelected ? "Clear page" : "Select page"}
              </button>
              <button type="button" className="ghost-button" onClick={() => void selectAllFiltered()}>
                Select all filtered
              </button>
              <button type="submit" className="primary-button" disabled={submitting}>
                {submitting ? "Creating..." : "Create dataset version"}
              </button>
            </div>
          </form>
          {message ? <p className="success-text">{message}</p> : null}
          {error ? <p className="error-text">{error}</p> : null}
        </SectionCard>

        <SectionCard title="Filters" description="Narrow the sample pool before creating a dataset version.">
          <div className="form-grid">
            <label>
              <span>Search sample id</span>
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="sample_001" />
            </label>
            <label>
              <span>Resolution bucket</span>
              <select value={resolutionBucket} onChange={(event) => setResolutionBucket(event.target.value)}>
                <option value="">All</option>
                <option value="<=256">&lt;=256</option>
                <option value="257-512">257-512</option>
                <option value="513-1024">513-1024</option>
                <option value="1025+">1025+</option>
              </select>
            </label>
            <label>
              <span>Status</span>
              <select value={status} onChange={(event) => setStatus(event.target.value)}>
                <option value="">All</option>
                <option value="ready">Ready</option>
                <option value="missing_image">Missing image</option>
                <option value="missing_mask">Missing mask</option>
              </select>
            </label>
            <label>
              <span>Sort</span>
              <select value={sort} onChange={(event) => setSort(event.target.value)}>
                <option value="sample_id">Sample id</option>
                <option value="width_desc">Width desc</option>
                <option value="width_asc">Width asc</option>
                <option value="height_desc">Height desc</option>
                <option value="height_asc">Height asc</option>
              </select>
            </label>
            <label>
              <span>Width min</span>
              <input value={widthMin} onChange={(event) => setWidthMin(event.target.value)} type="number" />
            </label>
            <label>
              <span>Width max</span>
              <input value={widthMax} onChange={(event) => setWidthMax(event.target.value)} type="number" />
            </label>
            <label>
              <span>Height min</span>
              <input value={heightMin} onChange={(event) => setHeightMin(event.target.value)} type="number" />
            </label>
            <label>
              <span>Height max</span>
              <input value={heightMax} onChange={(event) => setHeightMax(event.target.value)} type="number" />
            </label>
            <label>
              <span>Page size</span>
              <select
                value={pageSize}
                onChange={(event) => {
                  setPageSize(Number(event.target.value));
                  setPage(1);
                }}
              >
                <option value={25}>25</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </select>
            </label>
          </div>
        </SectionCard>
      </div>

      <SectionCard
        title="Approved samples"
        description={datasetPage ? `${datasetPage.total_count} sample(s) matched the current filters.` : "Loading samples..."}
        actions={
          <PaginationControls page={datasetPage?.page ?? 1} totalPages={datasetPage?.total_pages ?? 0} onPageChange={setPage} />
        }
      >
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th aria-label="Select">Pick</th>
                <th>Sample</th>
                <th>Resolution</th>
                <th>Status</th>
                <th>Image</th>
              </tr>
            </thead>
            <tbody>
              {(datasetPage?.items ?? []).map((item) => (
                <tr key={item.sample_id}>
                  <td>
                    <input
                      type="checkbox"
                      checked={selectedIds.includes(item.sample_id)}
                      disabled={!item.selectable}
                      onChange={() => {
                        setSelectedIds((current) =>
                          current.includes(item.sample_id)
                            ? current.filter((sampleId) => sampleId !== item.sample_id)
                            : [...current, item.sample_id],
                        );
                      }}
                    />
                  </td>
                  <td>
                    <div className="cell-stack">
                      <strong>{item.sample_id}</strong>
                      <small>{item.sample_path}</small>
                    </div>
                  </td>
                  <td>{item.width && item.height ? `${item.width} × ${item.height}` : "Unknown"}</td>
                  <td>
                    <StatusBadge value={item.status} />
                  </td>
                  <td className="truncate">{item.image_path ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {loading ? <p className="muted-text">Loading samples...</p> : null}
      </SectionCard>
    </div>
  );
}

function TrainPage() {
  const [datasetVersions, setDatasetVersions] = useState<DatasetVersion[]>([]);
  const [models, setModels] = useState<RegisteredModel[]>([]);
  const [datasetVersionId, setDatasetVersionId] = useState("");
  const [modelName, setModelName] = useState("");
  const [epochs, setEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(2);
  const [learningRate, setLearningRate] = useState(0.001);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    void Promise.all([api.getDatasetVersions(), api.getModels()])
      .then(([nextDatasets, nextModels]) => {
        setDatasetVersions(nextDatasets);
        setModels(nextModels);
        setDatasetVersionId(nextDatasets[0] ? String(nextDatasets[0].id) : "");
        setModelName(nextModels[0]?.key ?? "");
      })
      .catch((requestError) => setError((requestError as Error).message));
  }, []);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError("");
    setMessage("");
    try {
      const result = await api.createRun({
        dataset_version_id: Number(datasetVersionId),
        model_name: modelName,
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
      });
      setMessage(`Queued run ${result.run_name}.`);
    } catch (requestError) {
      setError((requestError as Error).message);
    }
  };

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Train"
        title="Launch a controlled training run."
        description="Pick a dataset version, choose a registered model, and push a run into the managed pipeline."
      />
      <SectionCard title="Run configuration" description="All models stay pre-registered and reproducible.">
        <form className="stack-form" onSubmit={(event) => void submit(event)}>
          <div className="form-grid">
            <label>
              <span>Dataset version</span>
              <select value={datasetVersionId} onChange={(event) => setDatasetVersionId(event.target.value)}>
                {datasetVersions.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Model</span>
              <select value={modelName} onChange={(event) => setModelName(event.target.value)}>
                {models.map((model) => (
                  <option key={model.key} value={model.key}>
                    {model.display_name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Epochs</span>
              <input type="number" min={1} value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} />
            </label>
            <label>
              <span>Batch size</span>
              <input type="number" min={1} value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} />
            </label>
            <label>
              <span>Learning rate</span>
              <input
                type="number"
                min={0}
                step={0.0001}
                value={learningRate}
                onChange={(event) => setLearningRate(Number(event.target.value))}
              />
            </label>
          </div>
          <div className="action-row">
            <button type="submit" className="primary-button">
              Start training
            </button>
          </div>
        </form>
        {message ? <p className="success-text">{message}</p> : null}
        {error ? <p className="error-text">{error}</p> : null}
      </SectionCard>
    </div>
  );
}

function RunsPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [status, setStatus] = useState("");
  const [modelName, setModelName] = useState("");
  const [query, setQuery] = useState("");
  const [error, setError] = useState("");

  const loadRuns = async () => {
    try {
      const params = new URLSearchParams();
      if (status) params.set("status", status);
      if (modelName) params.set("model_name", modelName);
      if (query) params.set("query", query);
      setRuns(await api.getRuns(params.toString() ? `?${params.toString()}` : ""));
    } catch (requestError) {
      setError((requestError as Error).message);
    }
  };

  useEffect(() => {
    void loadRuns();
  }, [status, modelName, query]);

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Runs"
        title="Track every experiment in one place."
        description="Filter by model and state, then jump into logs, metrics, and checkpoint-backed run history."
      />
      <SectionCard title="Run filters" description="Trim the run list before diving into the details.">
        <div className="form-grid">
          <label>
            <span>Status</span>
            <select value={status} onChange={(event) => setStatus(event.target.value)}>
              <option value="">All</option>
              <option value="queued">Queued</option>
              <option value="running">Running</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
              <option value="stopped">Stopped</option>
            </select>
          </label>
          <label>
            <span>Model name</span>
            <input value={modelName} onChange={(event) => setModelName(event.target.value)} placeholder="unet" />
          </label>
          <label>
            <span>Run name</span>
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="run_2026..." />
          </label>
        </div>
      </SectionCard>
      <SectionCard title="Run list" description={`${runs.length} run(s) matched the current filters.`}>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Model</th>
                <th>Status</th>
                <th>Dataset</th>
                <th>Best dice</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.run_name}>
                  <td>
                    <div className="cell-stack">
                      <strong>{run.run_name}</strong>
                      <small>{run.trainer_backend}</small>
                    </div>
                  </td>
                  <td>{run.model_name}</td>
                  <td>
                    <StatusBadge value={run.status} />
                  </td>
                  <td>{run.dataset_version_id}</td>
                  <td>{run.best_dice ?? "-"}</td>
                  <td>
                    <div className="button-cluster">
                      <Link to={`/runs/${run.run_name}`} className="ghost-link">
                        View
                      </Link>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {error ? <p className="error-text">{error}</p> : null}
      </SectionCard>
    </div>
  );
}

function RunDetailPage() {
  const { runName = "" } = useParams();
  const navigate = useNavigate();
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [error, setError] = useState("");
  const [working, setWorking] = useState(false);

  const loadDetail = async () => {
    try {
      setDetail(await api.getRunDetail(runName));
    } catch (requestError) {
      setError((requestError as Error).message);
    }
  };

  useEffect(() => {
    void loadDetail();
  }, [runName]);

  const retry = async () => {
    setWorking(true);
    try {
      await api.retryRun(runName);
      navigate("/runs");
    } catch (requestError) {
      setError((requestError as Error).message);
    } finally {
      setWorking(false);
    }
  };

  const stop = async () => {
    setWorking(true);
    try {
      await api.stopRun(runName);
      await loadDetail();
    } catch (requestError) {
      setError((requestError as Error).message);
    } finally {
      setWorking(false);
    }
  };

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Run detail"
        title={runName}
        description={detail ? `${detail.dataset_label} · ${detail.summary.model_name}` : "Loading run detail..."}
        actions={
          <div className="button-cluster">
            <button type="button" className="ghost-button" onClick={() => void retry()} disabled={working}>
              Retry
            </button>
            <button type="button" className="ghost-button danger" onClick={() => void stop()} disabled={working}>
              Stop
            </button>
          </div>
        }
      />
      {detail ? (
        <>
          <div className="metric-strip">
            <div className="metric-card">
              <span>Status</span>
              <strong>{detail.summary.status}</strong>
            </div>
            <div className="metric-card">
              <span>Best dice</span>
              <strong>{detail.summary.best_dice ?? "-"}</strong>
            </div>
            <div className="metric-card">
              <span>Epochs</span>
              <strong>{detail.summary.epochs}</strong>
            </div>
          </div>
          <SectionCard title="Epoch metrics" description="Dice history by epoch.">
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Epoch</th>
                    <th>Dice</th>
                  </tr>
                </thead>
                <tbody>
                  {detail.metrics.map((metric) => (
                    <tr key={metric.epoch}>
                      <td>{metric.epoch}</td>
                      <td>{metric.dice}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
          <SectionCard title="Training log" description="Device, per-epoch timings, and completion summary.">
            <pre className="log-block">{detail.log_text || "No log yet."}</pre>
          </SectionCard>
        </>
      ) : null}
      {error ? <p className="error-text">{error}</p> : null}
    </div>
  );
}

function ComparePage() {
  const [rows, setRows] = useState<CompareRow[]>([]);
  const [modelName, setModelName] = useState("");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState("best_dice_desc");
  const [error, setError] = useState("");

  useEffect(() => {
    const params = new URLSearchParams();
    if (modelName) params.set("model_name", modelName);
    if (query) params.set("query", query);
    if (sort) params.set("sort", sort);
    void api
      .getCompareRows(params.toString() ? `?${params.toString()}` : "")
      .then(setRows)
      .catch((requestError) => setError((requestError as Error).message));
  }, [modelName, query, sort]);

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Compare"
        title="Rank model runs side by side."
        description="Use the same visual system to sort checkpoints, compare datasets, and spot the strongest candidates."
      />
      <SectionCard title="Compare filters" description="Keep the ranking table focused.">
        <div className="form-grid">
          <label>
            <span>Model</span>
            <input value={modelName} onChange={(event) => setModelName(event.target.value)} placeholder="unet" />
          </label>
          <label>
            <span>Run name</span>
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="run_2026..." />
          </label>
          <label>
            <span>Sort</span>
            <select value={sort} onChange={(event) => setSort(event.target.value)}>
              <option value="best_dice_desc">Best dice desc</option>
              <option value="best_dice_asc">Best dice asc</option>
            </select>
          </label>
        </div>
      </SectionCard>
      <SectionCard title="Ranking" description={`${rows.length} run(s) in the comparison set.`}>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Dataset</th>
                <th>Model</th>
                <th>Best dice</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.run_name}>
                  <td>{row.run_name}</td>
                  <td>{row.dataset_label}</td>
                  <td>{row.model_name}</td>
                  <td>{row.best_dice}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {error ? <p className="error-text">{error}</p> : null}
      </SectionCard>
    </div>
  );
}

function TestPage() {
  const [runs, setRuns] = useState<TestableRun[]>([]);
  const [samples, setSamples] = useState<TestSample[]>([]);
  const [runName, setRunName] = useState("");
  const [sampleId, setSampleId] = useState("");
  const [preview, setPreview] = useState<InferencePreview | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    void api
      .getTestRuns()
      .then((nextRuns) => {
        setRuns(nextRuns);
        setRunName(nextRuns[0]?.run_name ?? "");
      })
      .catch((requestError) => setError((requestError as Error).message));
  }, []);

  useEffect(() => {
    if (!runName) {
      setSamples([]);
      return;
    }
    void api
      .getTestSamples(runName)
      .then((nextSamples) => {
        setSamples(nextSamples);
        setSampleId(nextSamples[0]?.sample_id ?? "");
      })
      .catch((requestError) => setError((requestError as Error).message));
  }, [runName]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError("");
    try {
      setPreview(await api.infer({ run_name: runName, sample_id: sampleId }));
    } catch (requestError) {
      setError((requestError as Error).message);
    }
  };

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Test"
        title="Compare original, GT, prediction, and overlay."
        description="Select a completed run, stay inside its dataset version, and judge checkpoint quality on the exact samples it learned from."
      />
      <SectionCard title="Inference controls" description="Choose a run and one sample from that run's dataset version.">
        <form className="stack-form" onSubmit={(event) => void submit(event)}>
          <div className="form-grid">
            <label>
              <span>Run</span>
              <select value={runName} onChange={(event) => setRunName(event.target.value)}>
                {runs.map((run) => (
                  <option key={run.run_name} value={run.run_name}>
                    {run.run_name} · {run.model_name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Sample</span>
              <select value={sampleId} onChange={(event) => setSampleId(event.target.value)}>
                {samples.map((sample) => (
                  <option key={sample.sample_id} value={sample.sample_id}>
                    {sample.sample_id}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="action-row">
            <button type="submit" className="primary-button">
              Run test
            </button>
          </div>
        </form>
        {error ? <p className="error-text">{error}</p> : null}
      </SectionCard>
      {preview ? (
        <SectionCard
          title={`${preview.model_name} · ${preview.sample_id}`}
          description={`${preview.dataset_label} · ${preview.checkpoint_path}`}
        >
          <div className="preview-grid">
            <figure className="preview-card">
              <img src={preview.original_data_url} alt="Original" />
              <figcaption>Original</figcaption>
            </figure>
            <figure className="preview-card">
              <img src={preview.ground_truth_data_url} alt="Ground truth" />
              <figcaption>Ground truth</figcaption>
            </figure>
            <figure className="preview-card">
              <img src={preview.prediction_data_url} alt="Prediction" />
              <figcaption>Prediction</figcaption>
            </figure>
            <figure className="preview-card">
              <img src={preview.overlay_data_url} alt="Overlay" />
              <figcaption>Overlay</figcaption>
            </figure>
          </div>
        </SectionCard>
      ) : null}
    </div>
  );
}

export default App;
