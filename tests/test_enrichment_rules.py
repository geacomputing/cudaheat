from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from cudaheat.enrichment.context import RuleContext
import cudaheat.enrichment.execution as execution
from cudaheat.enrichment.execution import _assert_immutable_version, _normalise_results, apply_rules
from cudaheat.enrichment.models import Event, RunReference
from cudaheat.enrichment.rules import EnrichmentRule, RuleValidationError, discover_rules


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RULE_DIRECTORY = REPOSITORY_ROOT / "examples" / "enrichment_pilot" / "enrichment_rules"


class _ChunkedArray:
    def __init__(self, values: np.ndarray, point_chunk: int = 2) -> None:
        self.values = values
        self.shape = values.shape
        self.chunks = (values.shape[0], point_chunk)

    def __getitem__(self, key):
        return self.values[key]


class _CountingChunkedArray(_ChunkedArray):
    def __init__(self, values: np.ndarray, point_chunk: int = 2) -> None:
        super().__init__(values, point_chunk)
        self.reads: list[object] = []

    def __getitem__(self, key):
        self.reads.append(key)
        return super().__getitem__(key)


def _rules():
    return {rule.id: rule for rule in discover_rules(RULE_DIRECTORY)}


def _context(
    *,
    sst: np.ndarray,
    threshold: np.ndarray,
    times: np.ndarray,
    events: list[Event],
    longitude: np.ndarray | None = None,
) -> RuleContext:
    point = np.arange(sst.shape[1], dtype=np.int64)
    source = xr.Dataset(
        {"sst": (("time", "point"), sst)},
        coords={
            "time": times,
            "point": point,
            "longitude": ("point", np.zeros(sst.shape[1]) if longitude is None else longitude),
            "latitude": ("point", np.zeros(sst.shape[1])),
        },
    )
    zarr_group = {
        "point": point,
        "threshold": _ChunkedArray(threshold),
        "climatology": _ChunkedArray(np.zeros_like(threshold)),
    }
    return RuleContext(
        events=events,
        cache_columns=np.asarray([event.grid_point_index for event in events]),
        active_cache_columns=point,
        cache_start=0,
        cache_stop=sst.shape[1],
        source=source,
        zarr_group=zarr_group,
        run=RunReference(
            "run", Path("source"), "hash", "clim", "sst", {"time_dimension": "time"}, "month_day_hour"
        ),
        source_times=times,
        time_bins=np.arange(times.size),
        timings={"source_read": 0.0, "zarr_read": 0.0, "numpy_compute": 0.0},
    )


def test_discovers_two_python_rules_with_publication_metadata():
    rules = _rules()

    assert set(rules) == {"threshold_exceedance_fraction", "nighttime_retention_fraction"}
    for rule in rules.values():
        assert rule.source_path.suffix == ".py"
        assert rule.metadata["formula_latex"]
        assert rule.metadata["output"] == {
            "primary": "value",
            "unit": "dimensionless",
            "range": [0.0, 1.0],
        }
        assert len(rule.sha256) == 64


def test_threshold_exceedance_matches_the_previous_count_as_false_semantics():
    rule = _rules()["threshold_exceedance_fraction"]
    times = np.arange("2020-01-01T00", "2020-01-01T04", dtype="datetime64[h]").astype("datetime64[ns]")
    context = _context(
        sst=np.array([[300.0], [301.0], [np.nan], [299.0]]),
        threshold=np.array([[299.0], [300.0], [300.0], [300.0]]),
        times=times,
        events=[Event(1, 0, None, None, 0, 4, 4)],
    )

    result = _normalise_results(rule, context, rule.calculate(context))[0]

    assert result == {
        "value": pytest.approx(2 / 4),
        "true_hours": 2,
        "total_hours": 4,
        "valid_hours": 3,
        "missing_hours": 1,
    }


def test_nighttime_retention_uses_only_event_hours_and_excludes_missing_values():
    rule = _rules()["nighttime_retention_fraction"]
    times = np.array(
        ["2020-01-01T18", "2020-01-01T19", "2020-01-01T23", "2020-01-02T02", "2020-01-02T06", "2020-01-02T12"],
        dtype="datetime64[ns]",
    )
    context = _context(
        sst=np.array([[2.0], [2.0], [np.nan], [0.0], [2.0], [2.0]]),
        threshold=np.ones((6, 1)),
        times=times,
        events=[Event(1, 0, None, None, 1, 5, 4)],
        longitude=np.array([0.0]),
    )

    result = _normalise_results(rule, context, rule.calculate(context))[0]

    assert result == {
        "value": pytest.approx(1 / 2),
        "night_hours": 3,
        "valid_night_hours": 2,
        "retained_night_hours": 1,
        "missing_night_hours": 1,
    }


def test_half_open_event_boundaries_exclude_stop_index():
    rule = _rules()["threshold_exceedance_fraction"]
    times = np.arange("2020-01-01T00", "2020-01-01T04", dtype="datetime64[h]").astype("datetime64[ns]")
    context = _context(
        sst=np.array([[0.0], [2.0], [0.0], [99.0]]),
        threshold=np.ones((4, 1)),
        times=times,
        events=[Event(1, 0, None, None, 1, 3, 2)],
    )

    result = _normalise_results(rule, context, rule.calculate(context))[0]

    assert result["true_hours"] == 1
    assert result["total_hours"] == 2
    assert result["value"] == pytest.approx(1 / 2)


def test_context_exposes_local_solar_time_and_generic_data_access():
    times = np.array(["2020-01-01T23", "2020-01-02T00"], dtype="datetime64[ns]")
    context = _context(
        sst=np.ones((2, 2)),
        threshold=np.ones((2, 2)),
        times=times,
        events=[Event(1, 0, None, None, 0, 2, 2)],
        longitude=np.array([15.0, -30.0]),
    )

    assert context.local_hour.tolist() == [[0.0, 21.0], [1.0, 22.0]]
    assert context.netcdf("sst").shape == (2, 2)
    assert context.zarr_cycle("threshold").shape == (2, 2)
    assert context.coordinate("latitude").shape == (2,)


def test_python_rule_discovery_requires_metadata_and_complete_file_sha(tmp_path):
    rule_path = tmp_path / "simple_rule.py"
    rule_path.write_text(
        """
METADATA = {
    'id': 'simple_rule', 'version': '1.0.0', 'name': 'Simple',
    'description': 'Simple metric', 'category': 'test',
    'output': {'primary': 'value', 'unit': 'dimensionless', 'range': [0.0, 1.0]},
    'formula_latex': r'X',
    'postgres': {
        'select_sql': 'SELECT * FROM events WHERE run_id = %(run_id)s AND (grid_point_index, event_id) > (%(after_grid_point_index)s, %(after_event_id)s) ORDER BY grid_point_index, event_id LIMIT %(batch_size)s',
        'write_sql': 'INSERT INTO event_enrichments VALUES (%(event_id)s) ON CONFLICT DO NOTHING',
    },
}
def calculate(context):
    return {'value': context.duration_hours * 0.0}
""",
        encoding="utf-8",
    )
    first = discover_rules(tmp_path)[0]
    rule_path.write_text(
        rule_path.read_text(encoding="utf-8") + "\n# metadata or implementation changed\n",
        encoding="utf-8",
    )
    second = discover_rules(tmp_path)[0]

    assert first.sha256 != second.sha256

    rule_path.write_text("METADATA = {}\n", encoding="utf-8")
    with pytest.raises(RuleValidationError, match="METADATA lacks"):
        discover_rules(tmp_path)


class _Cursor:
    def __init__(self, hashes):
        self.hashes = hashes

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, *_):
        return None

    def __iter__(self):
        return iter((value,) for value in self.hashes)


class _Connection:
    def __init__(self, hashes):
        self.hashes = hashes

    def cursor(self):
        return _Cursor(self.hashes)


def test_rule_file_sha_and_version_guard_protect_persisted_provenance():
    rule = _rules()["threshold_exceedance_fraction"]

    _assert_immutable_version(_Connection([rule.sha256]), rule)
    with pytest.raises(RuleValidationError, match="increase the version"):
        _assert_immutable_version(_Connection(["0" * 64]), rule)


class _PersistenceCursor:
    def __init__(self, connection):
        self.connection = connection
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, statement, _parameters=None):
        normalized = " ".join(statement.split()).upper()
        if normalized.startswith("SELECT DISTINCT RULE_SHA256"):
            self.rows = []
        elif normalized.startswith("SELECT E.EVENT_ID"):
            self.connection.event_selects += 1
            self.rows = self.connection.event_batches.pop(0)

    def executemany(self, statement, rows):
        self.connection.write_statement = statement
        self.connection.written_rows.extend(rows)

    def fetchall(self):
        return self.rows

    def __iter__(self):
        return iter(self.rows)


class _PersistenceConnection:
    def __init__(self, event_batches):
        self.event_batches = list(event_batches)
        self.written_rows = []
        self.write_statement = ""
        self.commits = 0
        self.event_selects = 0

    def cursor(self, **_):
        return _PersistenceCursor(self)

    def commit(self):
        self.commits += 1


def test_bulk_persistence_records_python_rule_version_and_file_sha_without_database():
    rule = _rules()["threshold_exceedance_fraction"]
    times = np.arange("2020-01-01T00", "2020-01-01T03", dtype="datetime64[h]").astype("datetime64[ns]")
    context = _context(
        sst=np.array([[2.0], [0.0], [2.0]]),
        threshold=np.ones((3, 1)),
        times=times,
        events=[Event(10, 0, None, None, 0, 3, 3)],
    )
    event_rows = [
        {
            "event_id": 10,
            "grid_point_index": 0,
            "start_index": 0,
            "stop_index": 3,
            "start_time": None,
            "end_time": None,
            "duration_hours": 3,
        }
    ]
    connection = _PersistenceConnection([event_rows, []])

    summary = apply_rules(
        connection,
        [rule],
        context.run,
        context.source,
        context.zarr_group,
        batch_size=10,
        write_results=True,
    )

    assert summary[0]["events_evaluated"] == 1
    assert connection.commits == 1
    assert "ON CONFLICT" in connection.write_statement
    row = connection.written_rows[0]
    assert row["rule_id"] == rule.id
    assert row["rule_version"] == rule.version
    assert row["rule_sha256"] == rule.sha256
    assert row["applied_at"].tzinfo is not None


def test_identical_selector_rules_share_one_context_scan_and_lazy_data_reads(monkeypatch):
    rules = _rules()
    nighttime = rules["nighttime_retention_fraction"]
    threshold_rule = rules["threshold_exceedance_fraction"]
    times = np.arange("2020-01-01T00", "2020-01-01T03", dtype="datetime64[h]").astype("datetime64[ns]")
    context = _context(
        sst=np.array([[2.0], [0.0], [2.0]]),
        threshold=np.ones((3, 1)),
        times=times,
        events=[Event(10, 0, None, None, 0, 3, 3)],
    )
    threshold = _CountingChunkedArray(np.ones((3, 1)))
    climatology = _CountingChunkedArray(np.zeros((3, 1)))
    context.zarr_group["threshold"] = threshold
    context.zarr_group["climatology"] = climatology
    event_rows = [
        {
            "event_id": 10,
            "grid_point_index": 0,
            "start_index": 0,
            "stop_index": 3,
            "start_time": None,
            "end_time": None,
            "duration_hours": 3,
        }
    ]
    connection = _PersistenceConnection([event_rows, []])
    contexts: list[RuleContext] = []

    class RecordingContext(RuleContext):
        def __post_init__(self) -> None:
            super().__post_init__()
            contexts.append(self)

    monkeypatch.setattr(execution, "RuleContext", RecordingContext)
    summary = apply_rules(
        connection,
        [nighttime, threshold_rule],
        context.run,
        context.source,
        context.zarr_group,
        batch_size=10,
        write_results=True,
    )

    assert connection.event_selects == 2  # one data page plus its keyset terminator, not one per rule
    assert len(contexts) == 1
    assert len(threshold.reads) == 1
    assert not climatology.reads
    expected_context = _context(
        sst=np.array([[2.0], [0.0], [2.0]]),
        threshold=np.ones((3, 1)),
        times=times,
        events=[Event(10, 0, None, None, 0, 3, 3)],
    )
    expected = {
        rule.id: _normalise_results(rule, expected_context, rule.calculate(expected_context))[0]
        for rule in (nighttime, threshold_rule)
    }
    persisted = {row["rule_id"]: row["result"].obj for row in connection.written_rows}
    assert persisted == expected
    assert set(summary[0]["rule_ids"]) == {nighttime.id, threshold_rule.id}
    assert set(summary[0]["rule_timing_seconds"]) == {
        f"{nighttime.id}@{nighttime.version}",
        f"{threshold_rule.id}@{threshold_rule.version}",
    }
    assert summary[0]["selector_group_wall_seconds"] >= 0.0
    assert all(row["applied_at"].tzinfo is not None for row in connection.written_rows)


def test_rules_with_different_selectors_keep_independent_keyset_scans(monkeypatch):
    rule = _rules()["threshold_exceedance_fraction"]
    alternate_metadata = dict(rule.metadata)
    alternate_postgres = dict(rule.metadata["postgres"])
    alternate_postgres["select_sql"] = str(alternate_postgres["select_sql"]).replace(
        "WHERE e.run_id", "WHERE e.duration_hours >= 0 AND e.run_id"
    )
    alternate_metadata["id"] = "threshold_exceedance_fraction_alternate"
    alternate_metadata["postgres"] = alternate_postgres
    alternate = EnrichmentRule(rule.source_path, alternate_metadata, rule.calculate, rule.sha256)
    times = np.arange("2020-01-01T00", "2020-01-01T02", dtype="datetime64[h]").astype("datetime64[ns]")
    context = _context(
        sst=np.ones((2, 1)),
        threshold=np.zeros((2, 1)),
        times=times,
        events=[Event(10, 0, None, None, 0, 2, 2)],
    )
    event_rows = [
        {
            "event_id": 10,
            "grid_point_index": 0,
            "start_index": 0,
            "stop_index": 2,
            "start_time": None,
            "end_time": None,
            "duration_hours": 2,
        }
    ]
    connection = _PersistenceConnection([event_rows, [], event_rows, []])
    contexts: list[RuleContext] = []

    class RecordingContext(RuleContext):
        def __post_init__(self) -> None:
            super().__post_init__()
            contexts.append(self)

    monkeypatch.setattr(execution, "RuleContext", RecordingContext)
    summary = apply_rules(
        connection,
        [rule, alternate],
        context.run,
        context.source,
        context.zarr_group,
        batch_size=10,
        write_results=False,
    )

    assert connection.event_selects == 4
    assert len(contexts) == 2
    assert len(summary) == 2
