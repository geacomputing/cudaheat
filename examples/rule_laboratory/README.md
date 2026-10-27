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

Put candidate `.yml` or `.yaml` rules in `rules/`. All files there are
schema-validated and evaluated together. The JSON output contains the rule
outputs and timing; pass `--output laboratory-results.json` to save it.

One PNG per event is written to `plots/<run-id>/`. Each plot contains SST,
threshold, climatology, and 24 hours of context around the shaded event. Its
right-side text panel lists the run configuration and every result grouped by
rule. Set
`--plot-padding-hours 0` for the event interval only, or use
`--plot-directory PATH` to place images elsewhere. Install the optional
Matplotlib dependency if needed with `pip install -e '.[postgres,plot]'`.

The supplied `mean_event_anomaly.yml` demonstrates vectorized event means
using `context.sum(...)` and `context.count(...)`. It is not persisted and
does not affect production `rules/`.
