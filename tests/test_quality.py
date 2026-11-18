"""Cold-chain quality checks: bounds, coverage, verdicts, corrections."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from conftest import batch, ingest, trace

from app import quality


def at(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 1, 1, hour, minute, second, tzinfo=timezone.utc)


def iso(hour: int, minute: int = 0, second: int = 0) -> str:
    return at(hour, minute, second).isoformat().replace("+00:00", "Z")


BATCH = {"batch_no": "B1", "departed_at": at(8), "arrived_at": at(10)}
CADENCE = [(8, minute) for minute in range(0, 60, 10)] + [
    (9, minute) for minute in range(0, 60, 10)
] + [(10, 0)]


def series(values, cadence=CADENCE):
    return [
        {"at": at(hour, minute), "temperature_c": value}
        for (hour, minute), value in zip(cadence, values)
    ]


def constant(value):
    return series([value] * len(CADENCE))


def evaluate(probe_a, probe_b, batch_=None):
    return quality.evaluate(
        batch=batch_ or BATCH,
        probes=[
            {"probe_id": "A", "readings": probe_a},
            {"probe_id": "B", "readings": probe_b},
        ],
    )


# ------------------------------------------------------------------ verdicts


def test_full_coverage_cold_readings_release():
    out = evaluate(constant(4.0), constant(4.5))
    assert out["status"] == "released"
    assert out["releasable"] is True
    for probe in out["probes"]:
        assert probe["complete_coverage"] is True
        assert probe["lower_bound_minutes"] == 0
        assert probe["upper_bound_minutes"] == 0
        assert probe["evidence_gaps"] == []


def test_certain_long_excursion_is_anomalous_even_if_other_probe_is_clean():
    out = evaluate(constant(12.0), constant(4.0))
    assert out["status"] == "anomalous"
    assert out["anomalous_probe_ids"] == ["A"]
    assert out["probes"][0]["lower_bound_minutes"] == pytest.approx(120.0)


def test_only_a_guaranteed_violation_counts_as_anomalous():
    # 9.0 - 0.5 = 8.5 > 8, so the whole interval is guaranteed hot.
    out = evaluate(constant(9.0), constant(4.0))
    assert out["status"] == "anomalous"


def test_short_spike_within_error_margin_is_releasable():
    values = [4.0] * len(CADENCE)
    for index, (hour, minute) in enumerate(CADENCE):
        if (hour, minute) in {(9, 0), (9, 10)}:
            values[index] = 8.6
    out = evaluate(series(values), constant(4.0))
    probe = out["probes"][0]
    # The spike ramps in and out, so the lower envelope is above 8 °C for
    # ~10.43 minutes and the upper envelope for ~14.78: both within 15.
    assert probe["lower_bound_minutes"] == pytest.approx(10.4348, abs=1e-3)
    assert probe["upper_bound_minutes"] == pytest.approx(14.7826, abs=1e-3)
    assert out["status"] == "released"


def test_uncertainty_that_could_hide_a_violation_is_insufficient():
    # 9.0 is certain-hot for only 12 minutes (< 15), but +0.5 °C spreads it
    # to 16 minutes: the evidence can neither release nor condemn.
    values = [4.0] * len(CADENCE)
    for index, (hour, minute) in enumerate(CADENCE):
        if (hour, minute) in {(9, 0), (9, 10)}:
            values[index] = 9.0
    out = evaluate(series(values), constant(4.0))
    probe = out["probes"][0]
    assert probe["lower_bound_minutes"] == pytest.approx(12.0)
    assert probe["upper_bound_minutes"] == pytest.approx(16.0)
    assert out["status"] == "insufficient_evidence"
    assert out["uncertain_probe_ids"] == ["A"]
    assert out["releasable"] is False


def test_boundary_values_within_error_never_condemn():
    # 8.4 - 0.5 = 7.9: no guaranteed excursion, but +0.5 is 8.9 all along.
    out = evaluate(constant(8.4), constant(4.0))
    assert out["probes"][0]["lower_bound_minutes"] == 0
    assert out["probes"][0]["upper_bound_minutes"] == pytest.approx(120.0)
    assert out["status"] == "insufficient_evidence"


def test_excursion_bounds_are_computed_per_evidenced_segment():
    readings = [
        {"at": at(8, 0), "temperature_c": 4.0},
        {"at": at(8, 10), "temperature_c": 10.0},
        {"at": at(8, 20), "temperature_c": 10.0},
        {"at": at(8, 30), "temperature_c": 10.0},
        {"at": at(8, 40), "temperature_c": 10.0},
        {"at": at(8, 50), "temperature_c": 10.0},
        {"at": at(9, 0), "temperature_c": 4.0},
        {"at": at(10, 0), "temperature_c": 4.0},
    ]
    out = quality.evaluate(
        batch=BATCH,
        probes=[
            {"probe_id": "A", "readings": readings},
            {"probe_id": "B", "readings": constant(4.0)},
        ],
    )
    probe = out["probes"][0]
    # Lower envelope crosses 8 °C at 08:07:30 and 08:52:30: 45 minutes.
    assert probe["lower_bound_minutes"] == pytest.approx(45.0)
    assert out["status"] == "anomalous"


def test_bound_exactly_fifteen_minutes_never_condemns():
    """15 minutes on the nose is not *more than* 15 minutes."""
    readings = [
        {"at": at(8, 0), "temperature_c": 4.0},
        {"at": at(8, 50), "temperature_c": 4.0},
        {"at": at(9, 0), "temperature_c": 10.0},
        {"at": at(9, 10), "temperature_c": 10.0},
        {"at": at(9, 20), "temperature_c": 4.0},
        {"at": at(10, 0), "temperature_c": 4.0},
    ]
    out = quality.evaluate(
        batch=BATCH,
        probes=[
            {"probe_id": "A", "readings": readings},
            {"probe_id": "B", "readings": constant(4.0)},
        ],
    )
    probe = out["probes"][0]
    # Lower envelope: 2.5 min ramp + 10 min plateau + 2.5 min fall = 15.
    assert probe["lower_bound_minutes"] == pytest.approx(15.0)
    # Upper envelope: 4.167 + 10 + 4.167 = 18.33 minutes, so doubt remains.
    assert probe["upper_bound_minutes"] == pytest.approx(18.3333, abs=1e-3)
    assert out["status"] == "insufficient_evidence"
    assert out["uncertain_probe_ids"] == ["A"]


# ----------------------------------------------------------------- coverage


def test_gap_longer_than_ten_minutes_is_evidence_insufficient():
    probe_a = [
        {"at": at(8, 0), "temperature_c": 4.0},
        {"at": at(8, 30), "temperature_c": 4.0},  # 30-minute gap
        {"at": at(10, 0), "temperature_c": 4.0},
    ]
    out = evaluate(probe_a, constant(4.0))
    assert out["status"] == "insufficient_evidence"
    assert out["incomplete_probe_ids"] == ["A"]
    probe = out["probes"][0]
    assert probe["complete_coverage"] is False
    assert probe["covers_start"] is False
    assert probe["covers_end"] is False
    # No pair is <= 10 minutes apart, so the whole interval is a gap.
    assert [(g.start, g.end) for g in probe["evidence_gaps"]] == [
        (at(8, 0), at(10, 0))
    ]


def test_gap_of_exactly_ten_minutes_is_usable_evidence():
    probe_a = [
        {"at": at(8, 0), "temperature_c": 4.0},
        {"at": at(8, 10), "temperature_c": 4.0},
    ]
    out = quality.evaluate(
        batch={"batch_no": "B1", "departed_at": at(8), "arrived_at": at(8, 10)},
        probes=[
            {"probe_id": "A", "readings": probe_a},
            {"probe_id": "B", "readings": probe_a},
        ],
    )
    assert out["status"] == "released"


def test_probe_that_starts_late_reports_the_head_gap():
    probe_a = constant(4.0)[2:]  # first reading 08:20
    out = evaluate(probe_a, constant(4.0))
    assert out["status"] == "insufficient_evidence"
    probe = out["probes"][0]
    assert probe["covers_start"] is False
    assert probe["covers_end"] is True
    assert [(g.start, g.end) for g in probe["evidence_gaps"]] == [
        (at(8), at(8, 20))
    ]


def test_probe_that_ends_early_reports_the_tail_gap():
    probe_a = constant(4.0)[:-1]  # last reading 09:50
    out = evaluate(probe_a, constant(4.0))
    assert out["probes"][0]["covers_end"] is False
    assert [(g.start, g.end) for g in out["probes"][0]["evidence_gaps"]] == [
        (at(9, 50), at(10))
    ]
    assert out["status"] == "insufficient_evidence"


def test_gap_never_bridges_a_violation():
    """Hot before a gap and hot after it must not be added together."""
    probe_a = [
        {"at": at(8, 0), "temperature_c": 12.0},
        {"at": at(8, 10), "temperature_c": 12.0},
        {"at": at(8, 20), "temperature_c": 12.0},
        # 30-minute gap
        {"at": at(8, 50), "temperature_c": 12.0},
        {"at": at(9, 0), "temperature_c": 12.0},
        {"at": at(9, 10), "temperature_c": 12.0},
    ]
    out = evaluate(probe_a, constant(4.0))
    probe = out["probes"][0]
    # Two 20-minute hot runs, not one 70-minute run.
    assert probe["lower_bound_minutes"] == pytest.approx(20.0)
    assert out["status"] == "anomalous"


def test_readings_are_clipped_to_the_batch_interval():
    extra = [
        {"at": at(7, 30), "temperature_c": 12.0},
        *constant(4.0),
        {"at": at(10, 30), "temperature_c": 12.0},
    ]
    out = evaluate(extra, constant(4.0))
    probe = out["probes"][0]
    assert out["status"] == "released"
    assert probe["complete_coverage"] is True
    assert probe["evaluated_minutes"] == pytest.approx(120.0)
    assert probe["guaranteed_excursions"] == []


def test_readings_out_of_order_are_sorted():
    shuffled = list(reversed(constant(4.0)))
    out = evaluate(shuffled, constant(4.0))
    assert out["status"] == "released"


def test_duplicate_timestamps_rejected():
    readings = constant(4.0) + [{"at": at(8, 0), "temperature_c": 5.0}]
    with pytest.raises(quality.QualityFailure):
        evaluate(readings, constant(4.0))


# -------------------------------------------------------- reference cross-check


def _brute_force_minutes(readings, shift):
    """Sample the shifted piecewise-linear envelope once per second."""
    ordered = sorted(readings, key=lambda item: item["at"])
    segments = []
    for left, right in zip(ordered, ordered[1:]):
        if (right["at"] - left["at"]) > timedelta(minutes=10):
            continue
        start = max(left["at"], BATCH["departed_at"])
        end = min(right["at"], BATCH["arrived_at"])
        if start >= end:
            continue
        span = (right["at"] - left["at"]).total_seconds()

        def value(t, left=left, right=right, span=span):
            fraction = (t - left["at"]).total_seconds() / span
            return (
                left["temperature_c"]
                + (right["temperature_c"] - left["temperature_c"]) * fraction
                + shift
            )

        segments.append((start, end, value))
    segments.sort(key=lambda item: item[0])
    runs: list[list[tuple]] = []
    for segment in segments:
        if runs and segment[0] <= runs[-1][-1][1]:
            runs[-1].append(segment)
        else:
            runs.append([segment])

    best = 0
    for run in runs:
        longest = current = 0
        start, end = run[0][0], run[-1][1]
        tick = start
        while tick <= end:
            hot = False
            for seg_start, seg_end, value in run:
                if seg_start <= tick <= seg_end and value(tick) > 8.0:
                    hot = True
                    break
            current = current + 1 if hot else 0
            longest = max(longest, current)
            tick += timedelta(seconds=1)
        best = max(best, longest)
    return best / 60


def test_bounds_match_second_by_second_reference():
    import random

    rng = random.Random(20260101)
    for _ in range(120):
        count = rng.randint(2, 18)
        minutes = sorted(rng.sample(range(0, 121), count))
        readings = [
            {"at": at(8) + timedelta(minutes=minute),
             "temperature_c": round(rng.uniform(2, 14), 2)}
            for minute in minutes
        ]
        out = evaluate(readings, constant(4.0))
        probe = out["probes"][0]
        assert probe["lower_bound_minutes"] == pytest.approx(
            _brute_force_minutes(readings, -0.5), abs=0.02
        )
        assert probe["upper_bound_minutes"] == pytest.approx(
            _brute_force_minutes(readings, +0.5), abs=0.02
        )


# ---------------------------------------------------------------------- API


def seed(client, *, arrived="2026-01-01T10:00:00Z"):
    ingest(
        client,
        sources=[],
        samples=[{"sample_id": "S1", "batch_nos": ["B1"]}],
        batches=[
            batch("B1", "lab", "hub", "2026-01-01T08:00:00Z", arrived)
        ],
    )


def probes_payload(value_a=4.0, value_b=4.5):
    return {
        "batch_no": "B1",
        "probes": [
            {
                "probe_id": "P-A",
                "readings": [
                    {"at": iso(hour, minute), "temperature_c": value_a}
                    for hour, minute in CADENCE
                ],
            },
            {
                "probe_id": "P-B",
                "readings": [
                    {"at": iso(hour, minute), "temperature_c": value_b}
                    for hour, minute in CADENCE
                ],
            },
        ],
    }


def test_quality_check_documented(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/quality-checks" in paths
    assert "/quality-checks/{check_id}" in paths


def test_quality_check_ok_and_readable(client):
    seed(client)
    response = client.post("/quality-checks", json=probes_payload())
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "released"
    assert body["releasable"] is True
    assert body["batch"] == {
        "batch_no": "B1",
        "version": 1,
        "departed_at": "2026-01-01T08:00:00Z",
        "arrived_at": "2026-01-01T10:00:00Z",
        "origin": "lab",
        "destination": "hub",
    }
    assert [probe["probe_id"] for probe in body["probes"]] == ["P-A", "P-B"]

    read = client.get(f"/quality-checks/{body['check_id']}")
    assert read.status_code == 200
    assert read.json() == body
    listed = client.get("/quality-checks").json()["checks"]
    assert [(row["check_id"], row["releasable"]) for row in listed] == [
        (body["check_id"], True)
    ]


def test_quality_check_anomalous_reports_probe_and_intervals(client):
    seed(client)
    payload = probes_payload(value_a=12.0)
    response = client.post("/quality-checks", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "anomalous"
    assert body["releasable"] is False
    assert body["anomalous_probe_ids"] == ["P-A"]
    (hot,) = body["probes"][0]["guaranteed_excursions"]
    assert hot["start"] == "2026-01-01T08:00:00Z"
    assert hot["end"] == "2026-01-01T10:00:00Z"
    assert hot["duration_minutes"] == pytest.approx(120.0)


def test_quality_check_insufficient_lists_gaps(client):
    seed(client)
    payload = probes_payload()
    payload["probes"][1]["readings"] = [
        {"at": "2026-01-01T08:00:00Z", "temperature_c": 4.0},
        {"at": "2026-01-01T10:00:00Z", "temperature_c": 4.5},
    ]
    response = client.post("/quality-checks", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "insufficient_evidence"
    assert body["incomplete_probe_ids"] == ["P-B"]
    (gap,) = body["probes"][1]["evidence_gaps"]
    assert gap["start"] == "2026-01-01T08:00:00Z"
    assert gap["end"] == "2026-01-01T10:00:00Z"


def test_quality_check_unknown_batch_is_404(client):
    seed(client)
    payload = probes_payload()
    payload["batch_no"] = "NOPE"
    assert client.post("/quality-checks", json=payload).status_code == 404


def test_quality_check_before_dataset_is_404(client):
    assert client.post("/quality-checks", json=probes_payload()).status_code == 404


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["probes"].pop(),  # only one probe
        lambda p: p.update(probes=[]),
        lambda p: p["probes"][0].update(probe_id="P-B"),  # duplicate ids
        lambda p: p["probes"][0]["readings"].clear(),
        lambda p: p["probes"][0]["readings"].append(
            {**p["probes"][0]["readings"][0]}
        ),
        lambda p: p["probes"][0]["readings"][0].update(temperature_c="hot"),
        lambda p: p["probes"][0]["readings"][0].update(temperature_c=1000),
        lambda p: p.update(batch_no="  "),
        lambda p: p.update(extra=True),
    ],
)
def test_quality_check_invalid_payload_is_422_and_writes_nothing(client, mutation):
    seed(client)
    payload = probes_payload()
    mutation(payload)
    response = client.post("/quality-checks", json=payload)
    assert response.status_code == 422, response.text
    assert response.json()["code"] == "validation_error"
    assert client.get("/quality-checks").json()["checks"] == []


def test_correction_invalidates_quality_check(client):
    seed(client)
    check_id = client.post("/quality-checks", json=probes_payload()).json()[
        "check_id"
    ]
    correction = client.post(
        "/batches/B1/corrections",
        json={
            "seen_version": 1,
            "origin": "lab",
            "destination": "hub",
            "departed_at": "2026-01-01T08:00:00Z",
            "arrived_at": "2026-01-01T11:00:00Z",
        },
    )
    assert correction.status_code == 200
    assert correction.json()["invalidated_quality_check_ids"] == [check_id]

    stale = client.get(f"/quality-checks/{check_id}")
    assert stale.status_code == 409
    assert stale.json()["code"] == "quality_check_invalidated"
    assert stale.json()["details"]["invalidations"][0]["batch_no"] == "B1"

    listed = client.get("/quality-checks").json()["checks"][0]
    assert listed["invalidated"] is True
    assert listed["releasable"] is False

    # A fresh check pins the new version.
    fresh = client.post("/quality-checks", json=probes_payload()).json()
    assert fresh["batch"]["version"] == 2


def test_check_takes_batch_interval_from_ledger(client):
    # The request carries no interval: it is read from the batch record.
    seed(client, arrived="2026-01-01T09:00:00Z")
    response = client.post("/quality-checks", json=probes_payload())
    assert response.status_code == 201
    body = response.json()
    assert body["batch"]["arrived_at"] == "2026-01-01T09:00:00Z"
    # Readings after 09:00 are clipped, so coverage is complete.
    assert body["status"] == "released"
    assert body["probes"][0]["evaluated_minutes"] == pytest.approx(60.0)


def test_quality_check_does_not_touch_traces(client):
    seed(client)
    before = client.get("/traces").json()["traces"]
    client.post("/quality-checks", json=probes_payload())
    after = client.get("/traces").json()["traces"]
    assert before == after


def test_failed_evaluation_records_nothing(client, monkeypatch):
    seed(client)

    def boom(**kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(quality, "evaluate", boom)
    response = client.post("/quality-checks", json=probes_payload())
    assert response.status_code == 500
    assert response.json()["code"] == "quality_check_failed"
    assert client.get("/quality-checks").json()["checks"] == []


def test_correction_during_evaluation_returns_conflict(client, monkeypatch):
    """A correction between pinning and publication must not yield a verdict."""
    seed(client)
    real_evaluate = quality.evaluate

    def corrected_midway(**kwargs):
        outcome = real_evaluate(**kwargs)
        response = client.post(
            "/batches/B1/corrections",
            json={
                "seen_version": 1,
                "origin": "lab",
                "destination": "cool-room",
                "departed_at": "2026-01-01T08:00:00Z",
                "arrived_at": "2026-01-01T10:00:00Z",
            },
        )
        assert response.status_code == 200
        return outcome

    monkeypatch.setattr(quality, "evaluate", corrected_midway)
    response = client.post("/quality-checks", json=probes_payload())
    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "snapshot_changed"
    assert body["details"]["pinned_version"] == 1
    # Nothing was recorded: no stale release can be read back.
    assert client.get("/quality-checks").json()["checks"] == []


def test_correction_after_publication_is_never_returned_as_release(
    client, monkeypatch
):
    """Publish commits, then a correction lands before the response is built."""
    seed(client)
    from app.store import Store

    real_publish = Store.publish_quality_check

    def publish_then_correct(self, **kwargs):
        check_id = real_publish(self, **kwargs)
        self.correct_batch(
            batch_no="B1",
            seen_version=1,
            origin="lab",
            destination="cool-room",
            departed_at=quality._parse("2026-01-01T08:00:00Z"),
            arrived_at=quality._parse("2026-01-01T10:00:00Z"),
        )
        return check_id

    monkeypatch.setattr(Store, "publish_quality_check", publish_then_correct)
    response = client.post("/quality-checks", json=probes_payload())
    assert response.status_code == 409
    assert response.json()["code"] == "snapshot_changed"
    # The stored row is invalidated and can never be read as a release.
    listed = client.get("/quality-checks").json()["checks"]
    assert [(row["status"], row["releasable"], row["invalidated"]) for row in listed] == [
        ("released", False, True)
    ]
    assert client.get(f"/quality-checks/{listed[0]['check_id']}").status_code == 409


def test_correction_only_invalidates_checks_of_the_superseded_version(client):
    seed(client)
    first = client.post("/quality-checks", json=probes_payload()).json()
    client.post(
        "/batches/B1/corrections",
        json={
            "seen_version": 1,
            "origin": "lab",
            "destination": "cool-room",
            "departed_at": "2026-01-01T08:00:00Z",
            "arrived_at": "2026-01-01T10:00:00Z",
        },
    )
    second = client.post("/quality-checks", json=probes_payload()).json()
    listed = {
        row["check_id"]: row
        for row in client.get("/quality-checks").json()["checks"]
    }
    assert listed[first["check_id"]]["invalidated"] is True
    assert listed[second["check_id"]]["invalidated"] is False
    assert client.get(f"/quality-checks/{second['check_id']}").status_code == 200


def test_quality_check_read_unknown_is_404(client):
    seed(client)
    assert client.get("/quality-checks/999").status_code == 404
    assert client.get("/quality-checks/999/invalidations").status_code == 404


def test_quality_check_invalidations_endpoint(client):
    seed(client)
    check_id = client.post("/quality-checks", json=probes_payload()).json()[
        "check_id"
    ]
    client.post(
        "/batches/B1/corrections",
        json={
            "seen_version": 1,
            "origin": "lab",
            "destination": "cool-room",
            "departed_at": "2026-01-01T08:00:00Z",
            "arrived_at": "2026-01-01T10:00:00Z",
        },
    )
    rows = client.get(f"/quality-checks/{check_id}/invalidations").json()
    assert [(row["batch_no"], row["version"]) for row in rows] == [("B1", 1)]
    # Filtering the history by batch is supported.
    assert len(client.get("/quality-checks", params={"batch_no": "B1"}).json()["checks"]) == 1


def test_correction_is_rejected_while_a_check_is_in_flight(client, monkeypatch):
    """Pin, then let a correction run, then publish: the version moved.

    A quality check pins the live batch version before evaluating. If the
    correction commits while evaluation is running, publication observes the
    new live version and refuses to store the now-stale verdict.
    """
    seed(client)
    seen = {}

    def inspect(**kwargs):
        # Mid-check the batch is still version 1 and the ledger is untouched.
        seen["dataset"] = client.get("/dataset").json()["batches"][0]["version"]
        seen["checks"] = client.get("/quality-checks").json()["checks"]
        return quality.QualityFailure("abort after inspecting")

    monkeypatch.setattr(quality, "evaluate", inspect)
    assert client.post("/quality-checks", json=probes_payload()).status_code == 500
    assert seen == {"dataset": 1, "checks": []}


def test_trace_unaffected_by_quality_checks(client):
    seed(client)
    client.post("/quality-checks", json=probes_payload())
    assert trace(client).status_code == 201
