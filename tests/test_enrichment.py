from __future__ import annotations

from pathlib import Path
import runpy
from uuid import UUID, uuid4

import numpy as np
import pytest
import xarray as xr
import zarr

from cudaheat import CUDAHeat
from cudaheat.enrichment import RuleValidationError, discover_rules, enrich
from cudaheat.postgres import PostgresConfig, _persist_enrichment_catalogue, enrich_stored_run
from cudaheat.result import EventDetectionResult, EventGrid


def _rule_text(identifier: str, output: str) -> str:
    return f"""\
id: {identifier}
version: "1.0.0"
short_name: Test rule
long_name: Test vectorized rule
category:
  primary: test
  secondary: vectorized
description: Test rule.
formula_latex: X
symbols:
  X: Test output
output:
  name: {output}
  unit: dimensionless
  range: [0.0, 1.0]
  dimensions: [event]
python: |
  import numpy as np

  def calculate(context):
      valid = np.isfinite(context.sst) & np.isfinite(context.threshold)
      above = valid & (context.sst > context.threshold)
      true_hours = context.count(above)
      valid_hours = context.count(valid)
      with np.errstate(divide="ignore", invalid="ignore"):
          value = true_hours / valid_hours
      return {{"value": value, "true_hours": true_hours, "valid_hours": valid_hours}}
"""


def _write_rule(directory: Path, identifier: str = "fraction", output: str = "fraction") -> Path:
    path = directory / f"{identifier}.yaml"
    path.write_text(_rule_text(identifier, output), encoding="utf-8")
    return path


def _result(tmp_path: Path) -> EventDetectionResult:
    time = np.arange("2020-01-01T00", "2020-01-01T06", dtype="datetime64[h]")
    points = np.array([10, 20, 30, 40], dtype=np.int64)
    sst = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 0.0],
            [2.0, 0.0, 2.0, 0.0],
            [0.0, 2.0, 2.0, 0.0],
            [99.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    source_path = tmp_path / "sst.cudaheat.nc"
    xr.Dataset(
        {"sst": (("time", "point"), sst)},
        coords={
            "time": time,
            "point": points,
            "latitude": ("point", np.array([1.0, 2.0, 3.0, 4.0])),
            "longitude": ("point", np.array([10.0, 20.0, 30.0, 40.0])),
        },
    ).to_netcdf(source_path, engine="h5netcdf")
    cache_path = tmp_path / "cycles.zarr"
    group = zarr.open_group(str(cache_path), mode="w")
    group.attrs.update(
        {
            "cudaheat_complete": True,
            "cudaheat_climatology_hash": "a" * 64,
        }
    )
    group.create_dataset("point", data=points, chunks=(2,))
    group.create_dataset("threshold", data=np.ones((8784, 4), dtype=np.float32), chunks=(8784, 2))
    group.create_dataset("climatology", data=np.zeros((8784, 4), dtype=np.float32), chunks=(8784, 2))
    group.create_dataset("longitude", data=np.array([10.0, 20.0, 30.0, 40.0]), chunks=(2,))
    catalogue = xr.Dataset(
        {
            "point_index": ("event", np.array([10, 20, 30], dtype=np.int64)),
            "start_index": ("event", np.array([1, 0, 1], dtype=np.int64)),
            "stop_index": ("event", np.array([4, 2, 4], dtype=np.int64)),
            "duration_hours": ("event", np.array([3, 2, 3], dtype=np.int64)),
            "start_time": ("event", time[[1, 0, 1]]),
            "end_time": ("event", time[[3, 1, 3]]),
            "event_count": ("point", np.array([1, 1, 1, 0], dtype=np.uint64)),
        },
        coords={"event": np.array([1, 2, 3], dtype=np.int64), "point": points},
    )
    return EventDetectionResult(
        data=catalogue,
        telemetry=(),
        metadata={
            "source_path": str(source_path),
            "climatology_cache_path": str(cache_path),
            "climatology_hash": "a" * 64,
            "variable": "sst",
            "time_dimension": "time",
            "calendar_mode": "month_day_hour",
        },
        _grid=EventGrid(time=time, point_index=points, point_indices=points, time_dim="time"),
    )


def test_yaml_discovery_validates_schema_compiles_python_and_finds_calculate(tmp_path):
    _write_rule(tmp_path)

    rule = discover_rules(tmp_path)[0]

    assert rule.id == "fraction"
    assert callable(rule.calculate)
    assert len(rule.sha256) == 64


def test_discovery_rejects_invalid_schema_and_invalid_embedded_python(tmp_path):
    broken_schema = _write_rule(tmp_path, "schema", "schema_output")
    broken_schema.write_text(broken_schema.read_text(encoding="utf-8").replace("id: schema\n", ""))
    with pytest.raises(RuleValidationError, match="invalid enrichment rule"):
        discover_rules(tmp_path)

    broken_schema.unlink()
    broken_python = _write_rule(tmp_path, "python", "python_output")
    broken_python.write_text(
        broken_python.read_text(encoding="utf-8").replace(
            "def calculate(context):", "def calculate("
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuleValidationError, match="invalid Python"):
        discover_rules(tmp_path)


def test_enrichment_is_pre_persistence_vectorized_and_keeps_half_open_boundaries(tmp_path):
    _write_rule(tmp_path)
    _write_rule(tmp_path, "fraction_two", "fraction_two")
    result = _result(tmp_path)

    enriched = enrich(result, rule_directory=tmp_path)

    assert "fraction" not in result.data
    np.testing.assert_allclose(enriched.data["fraction"].values, [2 / 3, 1 / 2, 1.0])
    np.testing.assert_allclose(enriched.data["fraction_two"].values, [2 / 3, 1 / 2, 1.0])
    np.testing.assert_array_equal(enriched.data["fraction_true_hours"].values, [2, 1, 3])
    timing = enriched.metadata["enrichment_timing"]
    assert timing["event_count"] == 3
    assert timing["spatial_batches"] == 2
    assert timing["source_reads"] == 2
    assert timing["threshold_reads"] == 2
    assert timing["climatology_reads"] == 0
    assert timing["wall_seconds"] >= 0
    assert len(enriched.metadata["enrichment_rules"]) == 2


def test_rule_laboratory_candidates_are_evaluable(tmp_path):
    rule_directory = Path(__file__).parents[1] / "examples/rule_laboratory/rules"
    rules = discover_rules(rule_directory)

    enriched = enrich(_result(tmp_path), rule_directory=rule_directory)

    assert rules
    assert all(rule.output_name in enriched.data for rule in rules)


def test_rule_laboratory_writes_one_plot_per_event(tmp_path):
    pytest.importorskip("matplotlib")
    laboratory = runpy.run_path(
        str(Path(__file__).parents[1] / "examples/rule_laboratory/rule_lab.py")
    )
    result = _result(tmp_path)
    rule_directory = Path(__file__).parents[1] / "examples/rule_laboratory/rules"
    rules = discover_rules(rule_directory)
    enriched = enrich(result, rule_directory=rule_directory)
    run_id = uuid4()

    paths = laboratory["write_event_plots"](
        run_id,
        result.data.isel(event=slice(0, 2)),
        enriched.data.isel(event=slice(0, 2)),
        result.metadata,
        rules=rules,
        directory=tmp_path / "plots",
        padding_hours=0,
    )

    assert len(paths) == 2
    assert all(path.parent == tmp_path / "plots" / str(run_id) for path in paths)
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)
    panel = laboratory["_event_panel_text"](run_id, 0, result.metadata, enriched.data, rules)
    assert "RUN CONFIGURATION" in panel
    assert "RULE RESULTS" in panel
    assert f"{rules[0].id}: " in panel


class _Jsonb:
    def __init__(self, obj):
        self.obj = obj


class _Cursor:
    def __init__(self, event_ids: list[tuple[int, int]]) -> None:
        self.event_ids = event_ids
        self.rows: list[tuple[object, ...]] = []
        self.written: list[dict[str, object]] = []

    def execute(self, statement, _parameters=None):
        normalized = " ".join(statement.split()).upper()
        if normalized.startswith("SELECT EVENT_ID, CUDAHEAT_EVENT_INDEX"):
            self.rows = self.event_ids
        elif normalized.startswith("SELECT DISTINCT RULE_SHA256"):
            self.rows = []

    def executemany(self, _statement, rows):
        self.written.extend(rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows

    def __iter__(self):
        return iter(self.rows)


def test_enriched_values_are_persisted_as_event_enrichments(tmp_path):
    _write_rule(tmp_path)
    enriched = enrich(_result(tmp_path), rule_directory=tmp_path)
    cursor = _Cursor([(101, 1), (102, 2), (103, 3)])

    _persist_enrichment_catalogue(cursor, enriched.data, enriched.metadata, uuid4(), _Jsonb)

    assert len(cursor.written) == 3
    assert cursor.written[0]["event_id"] == 101
    assert cursor.written[0]["result"].obj == {"value": pytest.approx(2 / 3), "true_hours": 2, "valid_hours": 3}


def test_detection_can_enrich_on_the_gpu_without_a_second_source_read(tmp_path):
    _write_rule(tmp_path)
    time = np.arange("2020-01-01T00", "2020-01-04T00", dtype="datetime64[h]")
    values = np.ones((time.size, 2), dtype=np.float32)
    values[10:40, 0] = 5.0
    source_path = tmp_path / "gpu_sst.nc"
    xr.Dataset(
        {"sst": (("time", "point"), values)},
        coords={
            "time": time,
            "point": np.array([0, 1]),
            "latitude": ("point", np.array([1.0, 2.0])),
            "longitude": ("point", np.array([10.0, 20.0])),
        },
    ).to_netcdf(source_path, engine="h5netcdf")

    with CUDAHeat(
        source_path,
        variable="sst",
        batch_size=2,
        input_optimization="skip",
        timing_logging=False,
    ) as heat:
        result = heat.detect(
            percentile=0.5,
            smooth_window_hours=24,
            min_duration_hours=2,
            max_internal_gap_hours=0,
            join_gap_hours=0,
            enrich=True,
            rule_directory=tmp_path,
            enrichment_batch_size=1,
        )

    assert "fraction" in result.data
    assert result.metadata["enrichment_timing"]["source_reads"] == 0
    assert result.metadata["enrichment_timing"]["threshold_reads"] == 0
    assert result.metadata["enrichment_timing"]["spatial_subbatches"] == 1
    np.testing.assert_allclose(result.data["fraction"].values, [1.0])


def test_gpu_enrichment_matches_the_standalone_vectorized_path(monkeypatch, tmp_path):
    _write_rule(tmp_path)
    monkeypatch.setenv("CUDAHEAT_CACHE_DIR", str(tmp_path / "cache"))
    time = np.arange("2020-01-01T00", "2020-01-04T00", dtype="datetime64[h]")
    values = np.ones((time.size, 2), dtype=np.float32)
    values[10:40, 0] = 5.0
    source_path = tmp_path / "parity_sst.nc"
    xr.Dataset(
        {"sst": (("time", "point"), values)},
        coords={
            "time": time,
            "point": np.array([0, 1]),
            "latitude": ("point", np.array([1.0, 2.0])),
            "longitude": ("point", np.array([10.0, 20.0])),
        },
    ).to_netcdf(source_path, engine="h5netcdf")
    options = {
        "percentile": 0.5,
        "smooth_window_hours": 24,
        "min_duration_hours": 2,
        "max_internal_gap_hours": 0,
        "join_gap_hours": 0,
    }

    with CUDAHeat(source_path, variable="sst", batch_size=2, timing_logging=False) as heat:
        standalone_input = heat.detect(**options)
    standalone = enrich(standalone_input, rule_directory=tmp_path)
    with CUDAHeat(source_path, variable="sst", batch_size=2, timing_logging=False) as heat:
        integrated = heat.detect(
            **options,
            enrich=True,
            rule_directory=tmp_path,
            enrichment_batch_size=1,
        )

    for name in ("fraction", "fraction_true_hours", "fraction_valid_hours"):
        np.testing.assert_allclose(integrated.data[name].values, standalone.data[name].values)
    assert integrated.metadata["climatology_cache_status"] == "reused"
    assert integrated.metadata["enrichment_timing"]["source_reads"] == 0


class _RetroCursor(_Cursor):
    def __init__(self, run_id: UUID, metadata: dict[str, object]) -> None:
        super().__init__([(101, 1), (102, 2), (103, 3)])
        self.run_id = run_id
        self.metadata = metadata

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, statement, parameters=None):
        normalized = " ".join(statement.split()).upper()
        if normalized.startswith("SELECT RUN_ID, METADATA"):
            self.rows = [(self.run_id, self.metadata)]
        elif normalized.startswith("SELECT E.CUDAHEAT_EVENT_INDEX"):
            self.rows = [
                (1, 10, 1, 4, 3, np.datetime64("2020-01-01T01"), np.datetime64("2020-01-01T03")),
                (2, 20, 0, 2, 2, np.datetime64("2020-01-01T00"), np.datetime64("2020-01-01T01")),
                (3, 30, 1, 4, 3, np.datetime64("2020-01-01T01"), np.datetime64("2020-01-01T03")),
            ]
        else:
            super().execute(statement, parameters)


class _RetroConnection:
    def __init__(self, cursor: _RetroCursor) -> None:
        self.cursor_value = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):
        return self.cursor_value


class _RetroPsycopg:
    def __init__(self, connection: _RetroConnection) -> None:
        self.connection = connection

    def connect(self, **_):
        return self.connection


def test_retroactive_enrichment_reuses_the_vectorized_evaluator(monkeypatch, tmp_path):
    _write_rule(tmp_path)
    result = _result(tmp_path)
    run_id = uuid4()
    cursor = _RetroCursor(run_id, result.metadata)
    connection = _RetroConnection(cursor)
    monkeypatch.setattr("cudaheat.postgres._load_psycopg", lambda: _RetroPsycopg(connection))

    timing = enrich_stored_run(
        run_id,
        PostgresConfig("localhost", 5432, "cudaheat", "test"),
        rule_directory=tmp_path,
    )

    assert timing["source_reads"] == 2
    assert len(cursor.written) == 3
