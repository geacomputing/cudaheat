# CUDAHeat investigation report

## Scope and data audit

- PostgreSQL was reachable throughout.  It contains 13 mutually compatible
  detection runs on one grid (`22,083` points), `5,927,930` events, and five
  numerical enrichments per event.
- Every stored event was checked: `duration_hours == stop_index - start_index`
  and `end_time == start_time + (duration_hours - 1) hours`.  Matching therefore
  uses exact half-open index intervals, rather than inferred timestamps.

## 1. Descriptor-space patterns

Implemented [`discover_event_families.py`](examples/scatterPlot/discover_event_families.py).
It draws a deterministic, equal-run stratified sample (12,000 events from each
run), preserves event/run/point/time/location IDs, uses robust IQR scaling,
PCA, correlation analysis, density plots, and a conservative density graph.
Invalid rows are retained in the assignment file rather than silently dropped.

The full 156,000-event analysis is in
[`event_family_analysis/`](examples/scatterPlot/event_family_analysis/).  The
result is negative, but scientifically useful: robust-IQR normalization finds
one broad connected component (87.0% assigned, all 13 runs), not a defensible
multi-family taxonomy.  Neighbourhood/threshold changes retain one component
(ARI 0.822/0.828; reference-family Jaccard 0.960/0.968).  A separate 78,000
event, seed-independent sample reproduced this conclusion.

Rank-uniform normalization instead produces four components (ARI 0.038,
Jaccard 0.602 relative to baseline); those components are therefore not
normalization-robust and must not be presented as physical families.  PCA PC1
and PC2 explain 67.4% and 18.3%.  Several descriptors are redundant (notably
fraction-above-threshold vs nocturnal-below-threshold, Spearman -0.881), while
nocturnal-below-climatology is exactly zero for 95.3% of sampled events.
UMAP/HDBSCAN were unavailable and deliberately not added as dependencies.

`event_assignments.csv` includes the matcher-compatible columns `event_id`,
`run_id`, and `cluster`; baseline `cluster` is intentionally the single stable
component.  Thus a non-trivial classification-robustness conclusion is not
supported by the present descriptors.

## 2. Cross-run robustness

Implemented [`cross_run_robustness.py`](examples/cross_run_robustness.py) and
database-independent unit tests.  It compares only runs with equal source,
climatology, threshold and calendar metadata; only same-point, temporally
overlapping events are candidates.  Its primary match is temporal IoU >= 0.50.
It separately reports unmatched events, sub-threshold overlaps, 80%-coverage
detection, onset/termination/duration stability, descriptor scores, and
one-to-one/split/merge/complex edges.  CSV detail export preserves both event
IDs, point IDs and event context.  An optional shared-taxonomy CSV yields
classification robustness without assuming event IDs persist.

For the deliberately contrasting `gap=6 h, join=6 h` -> `gap=0 h, join=12 h`
comparison: strict detection was 0.314 (139,956 / 445,688), reciprocal
detection 0.778; mean accepted IoU was 0.806; onset/termination/duration
scores were 0.900/0.906/0.807.  There were 29,843 temporal overlaps rejected
by the strict threshold and 275,889 events with no temporal overlap.  The
composite descriptor score was 0.952, but this hides heat-content's lower
score (0.882), so per-rule results are emitted separately.  This preliminary
result indicates stable properties conditional on matching but substantially
less stable detection when the reference is the more permissive run.

Preliminary outputs, including traceable match details, are in
[`examples/cross_run_robustness/preliminary/`](examples/cross_run_robustness/preliminary/).
The completed 78-pair compatible-run matrix is in
[`examples/cross_run_robustness/final_all_pairs/`](examples/cross_run_robustness/final_all_pairs/),
including the reproducibility manifest and per-rule descriptor summary.
Pooling both directions, strict detection ranges from 0.156 to 1.000 (median
0.778); conditional mean IoU ranges 0.743--1.000 (median 0.869), onset
stability 0.869--1.000 (median 0.934), termination stability 0.874--1.000
(median 0.936), duration similarity 0.743--1.000 (median 0.869), and
descriptor robustness 0.931--1.000 (median 0.968). The 25,256,304 strict
edges are all one-to-one; no strict split/merge edge was observed. This is
not evidence that non-matches are harmless: the scan records 2,878,640/2,870,399
directional sub-threshold-overlap events and 3,659,294/11,214,219
directional no-overlap events. Classification robustness is scientifically
undefined here: the only normalization-stable descriptor label is the one
broad population, while `-1` is noise (never counted as a shared family), not
a meaningful multi-family taxonomy. Descriptor scores are survivor-conditioned
(strictly matched events only); inspect their
per-rule CSV rather than treating the composite as a detection-robustness
measure. A threshold-sensitivity check on the most divergent pair found
98,304 strict (IoU >= 0.50), all one-to-one relations (mean IoU 0.743), but
153,755 relations and 11,039 merge edges only at permissive IoU >= 0.25
(mean IoU 0.611).  Those looser relations are deliberately excluded from the
primary robustness metric.

## 3. GPU Explorer

Fixed the loading state in [`gpu_explorer/app.py`](gpu_explorer/app.py#L222).
The actual failure was deck.gl rejecting `coordinateSystem: 0` for the
ScatterplotLayer (`Invalid coordinateSystem: 0` in Chromium DevTools).  The
app now serializes the Streamlit-compatible literal `"cartesian"` via
`pdk.types.String`, preserving the existing PostgreSQL-side density binning
and WebGL architecture.

The database path was verified with two runs / 1,179,271 source events in raw
and normalized modes.  Headless Chromium rendered a 438 x 760 canvas with
coloured points and no frontend errors or failed requests after the fix.  The
only warning was expected software-WebGL fallback in headless Chromium.

## Validation and limitations

- `pytest -q tests/test_cross_run_robustness.py tests/test_event_family_discovery.py`: 8 passed.
- Pattern discovery: 3 focused tests, `ruff`, `py_compile`, and whitespace
  checks passed.
- Explorer: real database, browser rendering, `ruff`, and `compileall` passed.
- The full existing suite reached 28 passing tests, then stopped at the
  unchanged `test_engine_dayofyear_mode_uses_hourwave_bins_and_metadata`:
  a 3-day synthetic cycle rejects the requested 11-day window.  This unrelated
  core detection/climatology failure was not modified.
- The present descriptor set does not justify a physical event taxonomy.
  Geographic/time-evolution validation and additional independent physical
  descriptors are needed before assigning mechanisms to events.
