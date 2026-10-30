# CUDAHeat GPU Explorer

Minimal Streamlit/deck.gl explorer for CUDAHeat enrichment results in PostgreSQL.

## Install and run

```bash
conda env create -f gpu_explorer/environment.yml
conda activate cudaheat-gpu-explorer
streamlit run gpu_explorer/app.py
```

The app reads the existing `config/postgres.ini` and resolves the password from
its configured environment variable or standard libpq/`.pgpass` configuration.
It never embeds credentials.

## Data model

Datasets are CUDAHeat runs. Numerical axes are `duration_hours` and the
persisted `event_enrichments.result.value` rule outputs. The explorer uses a
Cartesian `OrthographicView`, not a geographic map.

The WebGL view draws numerical axes, tick marks, and variable names. Click a
density cell to inspect its run, mean coordinates, population, and mean event
duration. `Generate PNG image` recreates the current binned view in Python and
offers the resulting image for download; it does not re-query raw events.
The enlarged dark canvas uses bright high-contrast axes and labels.

Each axis can be independently transformed to `log10` before PostgreSQL bins
the data; non-positive values on that axis are excluded explicitly. Marker
size can represent density or the mean value of any available third descriptor
within each density cell.

When exactly one run is selected, `Group over time` can aggregate by event
onset month, calendar season, or half-year (each bin remains tied to its
year). It plots descriptor means with a chronological colourbar and arrowed
streamline. This is a trajectory summary, not a transfer of raw events.

`Temporal analysis` supports one run or a three/four-run comparison. It
aggregates month-year, season-year, or year centroids and independently maps
numerical fields to X, Y, marker size, and colour/severity. Its controls rerun
the cached PostgreSQL query automatically; optional thin dashed semicircular
arcs are qualitative positional guides on the displayed descriptor plane.

## Six-million-point behaviour

Current PyDeck binary transport is documented as Jupyter-only, so a Streamlit
PyDeck chart cannot reliably accept millions of raw points without serializing
millions of JSON records. This app instead groups every selected event in
PostgreSQL into run-specific X/Y density cells before transfer. A cell retains
the mean coordinates, mean duration, and event count; all input events
contribute. `Density resolution` controls the number of cells and therefore
the visual detail and browser payload.
