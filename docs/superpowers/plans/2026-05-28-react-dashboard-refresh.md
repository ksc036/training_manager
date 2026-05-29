# React Dashboard Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a React-based dashboard under `/app`, unify the UI across key pages, redesign the `Datasets` experience with resolution-aware filtering and pagination, and enrich training logs with per-epoch elapsed time while preserving the current FastAPI training backend.

**Architecture:** Keep FastAPI as the backend and JSON API provider. Add a React frontend bundle served by FastAPI. Deliver the refresh incrementally: first the React shell and Datasets page, then the remaining pages, then logging improvements and compatibility cleanup.

**Tech Stack:** FastAPI, Python, React, TypeScript, Vite, PyTorch, SQLite, Pytest

---

### Task 1: Scaffold the React Frontend and `/app` Mount

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/styles.css`
- Modify: `app/main.py`
- Create: `tests/test_app_shell_routes.py`

- [ ] **Step 1: Write the failing test**

Add a route test asserting `GET /app` returns `200` and includes a frontend mount container.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest -q tests/test_app_shell_routes.py -k app_shell`
Expected: FAIL because `/app` does not exist

- [ ] **Step 3: Add minimal frontend scaffold**

Create a Vite React app with a root element and minimal `Dashboard coming soon` shell.

- [ ] **Step 4: Mount the built frontend in FastAPI**

Update `app/main.py` so `/app` serves the frontend entry and static assets.

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest -q tests/test_app_shell_routes.py -k app_shell`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add frontend app/main.py tests/test_app_shell_routes.py
git commit -m "feat: add react dashboard shell"
```

### Task 2: Add Shared React Dashboard Layout

**Files:**
- Create: `frontend/src/layout/DashboardShell.tsx`
- Create: `frontend/src/components/Sidebar.tsx`
- Create: `frontend/src/components/PageHeader.tsx`
- Create: `frontend/src/components/SectionCard.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Create: `tests/test_app_shell_routes.py`

- [ ] **Step 1: Write the failing test**

Extend the shell test to assert sidebar menu labels exist:

- `Datasets`
- `Train`
- `Runs`
- `Compare`
- `Test`

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest -q tests/test_app_shell_routes.py -k sidebar`
Expected: FAIL because menu labels are missing

- [ ] **Step 3: Implement the shared layout**

Build a fixed sidebar, top header, and content container using one visual system.

- [ ] **Step 4: Verify styling locally**

Check that sidebar, content spacing, and page shell render correctly in the browser.

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest -q tests/test_app_shell_routes.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add frontend tests/test_app_shell_routes.py
git commit -m "feat: add unified react dashboard layout"
```

### Task 3: Expose Dataset Sample List API with Resolution Metadata and Pagination

**Files:**
- Modify: `app/services/datasets.py`
- Create: `app/api/routes_datasets.py`
- Modify: `app/main.py`
- Modify: `trainer/scanner.py` if needed
- Create: `tests/test_dataset_api.py`

- [ ] **Step 1: Write the failing tests**

Add API tests for:

- `GET /api/datasets/samples`
- filtering by sample id
- filtering by resolution bucket
- pagination metadata

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest -q tests/test_dataset_api.py`
Expected: FAIL because the API does not exist

- [ ] **Step 3: Implement dataset list API**

Support query params:

- `query`
- `resolution_bucket`
- `width_min`
- `width_max`
- `height_min`
- `height_max`
- `status`
- `sort`
- `page`
- `page_size`

Response shape:

- `items`
- `total_count`
- `page`
- `page_size`

- [ ] **Step 4: Make sure width/height come from scanner metadata**

Do not reopen images on every request if scanner metadata is already enough.

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest -q tests/test_dataset_api.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/services/datasets.py app/api/routes_datasets.py app/main.py tests/test_dataset_api.py
git commit -m "feat: add dataset sample list api"
```

### Task 4: Build the React Datasets Page

**Files:**
- Create: `frontend/src/pages/DatasetsPage.tsx`
- Create: `frontend/src/components/FilterBar.tsx`
- Create: `frontend/src/components/DataTable.tsx`
- Create: `frontend/src/components/Pagination.tsx`
- Create: `frontend/src/lib/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Create: `tests/test_app_shell_routes.py`

- [ ] **Step 1: Write the failing test**

Add an integration-level route test that confirms the frontend shell includes a datasets view mount and required labels such as:

- `Dataset name`
- `Select all`
- `Create dataset version`

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest -q tests/test_app_shell_routes.py -k datasets`
Expected: FAIL

- [ ] **Step 3: Implement the Datasets page**

Include:

- page header
- source root summary
- creation card
- selected count
- filter card
- paginated results table
- row checkboxes
- page size control

- [ ] **Step 4: Fix selection behavior**

At minimum:

- page-level `Select all`
- selection count stays in sync
- manual row uncheck works

Preferred:

- filtered-set `Select all`

- [ ] **Step 5: Run tests and browser smoke check**

Run:

- `python3 -m pytest -q tests/test_app_shell_routes.py`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add frontend tests/test_app_shell_routes.py
git commit -m "feat: add react datasets page"
```

### Task 5: Add Dataset Version Creation API and Wire Create Flow

**Files:**
- Create: `app/api/routes_dataset_versions.py`
- Modify: `app/services/datasets.py`
- Modify: `app/main.py`
- Modify: `frontend/src/pages/DatasetsPage.tsx`
- Create: `tests/test_dataset_api.py`

- [ ] **Step 1: Write the failing test**

Add API test for `POST /api/datasets/versions` with selected sample ids and dataset name.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest -q tests/test_dataset_api.py -k create_dataset_version`
Expected: FAIL

- [ ] **Step 3: Implement the create endpoint**

Preserve current manifest creation logic and validation.

- [ ] **Step 4: Connect the React submit flow**

On success:

- clear or preserve selection intentionally
- show success message
- allow rescan or continued browsing

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest -q tests/test_dataset_api.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/api/routes_dataset_versions.py app/services/datasets.py app/main.py frontend tests/test_dataset_api.py
git commit -m "feat: wire react dataset creation flow"
```

### Task 6: Add Shared APIs for Runs, Compare, Train, and Test

**Files:**
- Create: `app/api/routes_models.py`
- Create: `app/api/routes_runs.py`
- Create: `app/api/routes_compare.py`
- Create: `app/api/routes_test.py`
- Modify: `app/main.py`
- Create: `tests/test_runs_api.py`
- Create: `tests/test_compare_api.py`
- Create: `tests/test_test_api.py`

- [ ] **Step 1: Write the failing tests**

Add API coverage for:

- runs list
- run detail
- compare list
- testable runs
- test samples by run

- [ ] **Step 2: Run tests to verify they fail**

Run:

- `python3 -m pytest -q tests/test_runs_api.py tests/test_compare_api.py tests/test_test_api.py`

Expected: FAIL

- [ ] **Step 3: Implement JSON APIs**

Reuse current services rather than duplicating business logic in routes.

- [ ] **Step 4: Run tests**

Run:

- `python3 -m pytest -q tests/test_runs_api.py tests/test_compare_api.py tests/test_test_api.py`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/api app/main.py tests/test_runs_api.py tests/test_compare_api.py tests/test_test_api.py
git commit -m "feat: add dashboard api routes"
```

### Task 7: Build React Pages for Train, Runs, Compare, and Test

**Files:**
- Create: `frontend/src/pages/TrainPage.tsx`
- Create: `frontend/src/pages/RunsPage.tsx`
- Create: `frontend/src/pages/ComparePage.tsx`
- Create: `frontend/src/pages/TestPage.tsx`
- Create: `frontend/src/components/MetricCard.tsx`
- Create: `frontend/src/components/EmptyState.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`

- [ ] **Step 1: Build Train page**

Use shared cards and form grid, keeping model selection and run creation clean.

- [ ] **Step 2: Build Runs page**

Support existing filters, search, pagination, and detail linking.

- [ ] **Step 3: Build Compare page**

Preserve current compare behavior while matching the unified visual system.

- [ ] **Step 4: Build Test page**

Preserve the 4-preview inference layout and make it fit the new shell.

- [ ] **Step 5: Browser smoke check**

Confirm all major pages render and navigation works.

- [ ] **Step 6: Commit**

```bash
git add frontend
git commit -m "feat: add react dashboard pages"
```

### Task 8: Record Per-Epoch Elapsed Time in Training Logs

**Files:**
- Modify: `trainer/external_adapter.py`
- Modify: `trainer/worker.py` if parity is trivial
- Modify: `tests/test_training_runner.py`

- [ ] **Step 1: Write the failing test**

Add a test asserting real training logs include per-epoch `elapsed=` and final `total_elapsed=`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest -q tests/test_training_runner.py -k elapsed`
Expected: FAIL

- [ ] **Step 3: Implement timing logs**

Record:

- device
- epoch elapsed seconds
- total elapsed seconds
- best epoch summary

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest -q tests/test_training_runner.py -k elapsed`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add trainer/external_adapter.py trainer/worker.py tests/test_training_runner.py
git commit -m "feat: add elapsed timing to training logs"
```

### Task 9: Final Integration, Compatibility Pass, and Verification

**Files:**
- Modify: `app/web/templates/base.html` only if legacy links should point to `/app`
- Modify: any docs or README files needed

- [ ] **Step 1: Decide legacy route behavior**

Either:

- keep old Jinja routes intact for compatibility

or:

- add visible link or redirect path to `/app`

- [ ] **Step 2: Run full test suite**

Run: `python3 -m pytest -q`
Expected: PASS

- [ ] **Step 3: Run frontend build verification**

Run the frontend build command and ensure FastAPI can serve the bundle.

- [ ] **Step 4: Manual browser smoke test**

Check:

- `/app`
- `/app/datasets`
- `/app/train`
- `/app/runs`
- `/app/compare`
- `/app/test`

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "feat: ship react dashboard refresh"
```
