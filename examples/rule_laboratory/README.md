# Rule laboratory

Evaluate candidate YAML enrichment rules against an existing run without
writing anything to PostgreSQL. PostgreSQL provides the selected run's
metadata and sparse events; the metadata points the laboratory to the
optimized SST file and climatology cache needed by the rules.

Run the included candidate rule against up to 5 events:

```bash
python examples/rule_laboratory/rule_lab.py \
  --run-id YOUR-RUN-UUID
```

Test particular events or points:

```bash
python examples/rule_laboratory/rule_lab.py \
  --run-id YOUR-RUN-UUID \
  --event-id 12 --event-id 19 \
  --point-index 384 \
  --max-events 0
```

Put candidate `.yml` or `.yaml` rules in `rules/`. The package discovers,
validates, and evaluates every rule there together. The JSON output contains
the rule outputs and timing; pass `--output laboratory-results.json` to save
it. Treat candidate rule directories as trusted code.

One PNG per event is written to `plots/<run-id>/`. Each plot contains SST,
threshold, climatology, and 24 hours of context around the shaded event. Its
right-side text panel lists the run configuration and every result grouped by
rule. Set
`--plot-padding-hours 0` for the event interval only, or use
`--plot-directory PATH` to place images elsewhere. Install the optional
Matplotlib dependency if needed with `pip install -e '.[postgres,plot]'`.
The dashed daily-mean threshold line is enabled by default; disable it with
`--no-daily-mean-threshold`.

## TikZ event figure export

`latex_event_figure.py` writes a CSV file plus an importable `figure*` TikZ
fragment for one point-event. The fragment spans both document columns, keeps
the line labels below the plotting area, and places detection settings and all
rule outputs in a side panel. It includes 12 h of context by default. If `--event-id` is omitted, the latest event at
the requested point is exported.

```bash
PYTHONPATH=src python examples/rule_laboratory/latex_event_figure.py \
  --run-id YOUR-RUN-UUID \
  --point-index 384 \
  --event-id 12
```

Files are written to `LaTex/figures/generated/` by default and can be included
from `cudaHeat.tex` with `\input{figures/generated/event-000012_point-384.tex}`.
The document palette defines the semantic TikZ colours `SSTLine`,
`ThresholdLine`, `ClimatologyLine`, and `EventShade` centrally in
`LaTex/class/cudaheat-colors.tex`.

The supplied rules demonstrate threshold-fraction and local-time enrichment
calculations using context fields plus `context.sum(...)` and
`context.count(...)`. They are not persisted and do not affect production
`rules/`.
