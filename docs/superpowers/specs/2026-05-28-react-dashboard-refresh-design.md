# React Dashboard Refresh Design

## Summary

Replace the current server-rendered HTML admin UI with a React-based dashboard served by the existing FastAPI app.

The backend remains in FastAPI. The frontend becomes a React multi-page dashboard mounted under `/app`, with the existing Jinja pages kept only as temporary compatibility routes during migration.

The refresh focuses on:

- a unified sidebar-driven dashboard UI
- a redesigned `Datasets` page with cleaner layout
- dataset filtering by image resolution and sample metadata
- dataset pagination for large sample pools
- consistent visual design across `Datasets`, `Train`, `Runs`, `Compare`, and `Test`
- richer training logs including per-epoch elapsed time

## Goals

- Keep the current FastAPI backend and training workflow intact.
- Introduce a React UI without forcing a full backend rewrite.
- Make the dashboard feel like one cohesive product instead of separate forms.
- Fix the current `Datasets` page layout issues where the dataset name and selection controls drift too far below the top of the page.
- Support browsing large approved datasets with filters and pagination.
- Preserve current features:
  - dataset creation
  - run creation
  - run monitoring
  - compare view
  - test inference view
- Improve training log usefulness by recording epoch duration and run timing summaries.

## Non-Goals

- Replacing FastAPI with Node or a separate frontend deployment
- Reworking the training database schema unless strictly needed for new list APIs
- Adding realtime websocket streaming
- Replacing the existing training engine
- Adding advanced role-based permissions

## Frontend Architecture

Create a React dashboard mounted under `/app`.

Recommended structure:

- `frontend/`
- `frontend/src/`
- `frontend/src/pages/`
- `frontend/src/components/`
- `frontend/src/lib/`

FastAPI responsibilities:

- serve the React bundle
- expose JSON endpoints for dashboard pages
- keep existing training/run logic unchanged

React responsibilities:

- page layout
- sidebar navigation
- filters, tables, pagination
- form interactions
- run/test browsing UI

Migration path:

- add `/app`
- keep existing Jinja routes temporarily:
  - `/datasets`
  - `/train`
  - `/runs`
  - `/compare`
  - `/test`
- once React pages are stable, optionally redirect those legacy routes to `/app/...`

## Navigation and Layout

Use a shared dashboard shell with:

- fixed left sidebar
- compact top page header
- consistent page content container

Sidebar entries:

- `Datasets`
- `Train`
- `Runs`
- `Compare`
- `Test`

Shared layout components:

- `Sidebar`
- `PageHeader`
- `SectionCard`
- `FilterBar`
- `DataTable`
- `MetricCard`
- `Pagination`
- `EmptyState`

## Visual Direction

UI should feel operational, clean, and consistent.

Guidelines:

- light neutral base
- one primary accent color
- consistent form control heights
- consistent spacing scale
- consistent card borders, shadows, and radius
- responsive two-column and multi-column layouts where needed

The app should avoid the current mixed form spacing and stacked controls that make actions feel disconnected.

## Datasets Page Refresh

The `Datasets` page becomes the first-class large-list management screen.

### Layout

Top area:

- page title
- approved source root summary
- `Rescan` action

Creation card:

- `Dataset name` input
- `Select all` toggle
- selected item count
- `Create dataset version` button

Filter card:

- sample id search
- resolution bucket filter
- width min/max
- height min/max
- status filter
- sort selector
- page size selector

Results area:

- paginated table
- selection checkbox per row
- thumbnail preview
- sample id
- width x height
- basic status

### Filters

Support:

- sample id substring search
- resolution buckets:
  - `<=256`
  - `257-512`
  - `513-1024`
  - `1025+`
- width range
- height range
- valid pair status
- missing image
- missing mask

### Pagination

Support page sizes:

- `25`
- `50`
- `100`

Pagination should preserve all filter query parameters.

### Selection Behavior

- `Select all` applies to the currently filtered result set, not just currently visible rows
- selection summary must be visible near the create action
- users can still uncheck individual rows before creating the dataset version

If implementing filtered full-set selection server-side is too large for the first iteration, the first React version may define `Select all` as “all rows in the current page,” but the UI must label that behavior clearly. The preferred behavior remains full filtered-set selection.

## Other Page Refreshes

### Train

- use the same page shell and cards
- dataset selector and model selector in a clean form grid
- primary action aligned consistently
- show helpful run configuration summary

### Runs

- unify filters into one top card
- cleaner status badges
- clearer row density
- preserve current search, filter, pagination, and detail linking

### Compare

- consistent table/card presentation with Runs
- preserve model filter, run search, sorting, and pagination
- stronger visual treatment for best metrics

### Test

- use the same dashboard shell
- organize run selector, sample selector, and submit action into a compact control card
- present the 4-image comparison in a consistent responsive grid:
  - original
  - ground truth
  - prediction
  - overlay

## API Additions

Add JSON endpoints for React consumption.

Suggested endpoints:

- `GET /api/datasets/samples`
- `POST /api/datasets/versions`
- `GET /api/datasets/versions`
- `GET /api/models`
- `POST /api/runs`
- `GET /api/runs`
- `GET /api/runs/{run_name}`
- `GET /api/compare`
- `GET /api/test/runs`
- `GET /api/test/runs/{run_name}/samples`
- `POST /api/test/infer`

`GET /api/datasets/samples` should support:

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

Response should include:

- `items`
- `total_count`
- `page`
- `page_size`
- `selected_filter_summary`

## Backend Service Changes

Add service helpers so list pages can support filtering and pagination without pushing all logic into routes.

Expected backend work:

- extend dataset sample service to return filterable metadata
- compute width/height from scanner metadata
- add paginated result helpers
- preserve existing manifest creation behavior

If dataset sample metadata is already available from scanner output, reuse it instead of recalculating image sizes on every request.

## Training Log Improvements

Training logs should include:

- selected device
- run start time context
- per-epoch elapsed time
- total training elapsed time
- best epoch summary at the end

Expected log shape:

- `starting actual torch training with model=unet`
- `device=mps`
- `epoch 1/20 - train_loss=... val_loss=... dice=... elapsed=3.42s`
- `epoch 2/20 - train_loss=... val_loss=... dice=... elapsed=3.35s`
- `training complete - total_elapsed=72.81s best_epoch=14 best_dice=0.9132`

The same elapsed timing does not need to be added to the simulated fallback path unless it is trivial to keep parity.

## Error Handling

- If a dataset filter yields no results, show a proper empty state instead of a blank table.
- If sample metadata is incomplete, expose it in row status rather than failing the page.
- If React API requests fail, show page-local error messages and retry affordances.
- If `/test` inference fails for a sample, keep the dashboard alive and surface the error in the result area.

## Testing

Add coverage for:

- dataset sample list filtering
- dataset sample pagination
- dataset creation with filtered or paginated selection
- runs/compare list APIs
- React bundle integration smoke test if feasible
- training log lines containing per-epoch elapsed time

Preserve or replace the current server-rendered page tests as the routes evolve.
