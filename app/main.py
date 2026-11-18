"""FastAPI application for contamination tracing.

Endpoints
---------
``POST /intakes``                  register samples + transfer batches + sources
``GET  /dataset``                   inspect the active dataset
``POST /batches/{batch_no}/corrections``
                                   replace a batch version and invalidate results
``POST /traces``                   run a trace on a pinned snapshot
``GET  /traces``                    list accepted traces
``GET  /traces/{id}``               re-read a published snapshot result
``POST /quality-checks``            cold-chain temperature evidence for one batch
``GET  /quality-checks``            list recorded checks
``GET  /quality-checks/{id}``       re-read a verdict (409 once invalidated)
``GET  /quality-checks/{id}/invalidations``
                                   correction records for a verdict
``GET  /healthz``                   readiness probe
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import quality, tracing
from .schemas import (
    AffectedSample,
    BatchCorrection,
    BatchOut,
    CorrectionResponse,
    DatasetResponse,
    ErrorBody,
    IntakeRequest,
    IntakeResponse,
    IntervalOut,
    InvalidationInfo,
    PathLeg,
    ProbeQualityResult,
    QualityBatchRef,
    QualityCheckListResponse,
    QualityCheckRequest,
    QualityCheckResponse,
    QualityCheckSummary,
    SnapshotInfo,
    TraceListResponse,
    TraceRequest,
    TraceResponse,
    TraceSummary,
)
from .store import (
    ConflictError,
    InvalidReferenceError,
    NotFoundError,
    Store,
    parse_iso,
)

DEFAULT_DB = os.environ.get("TRACE_DB_PATH", "./data/trace.db")


def create_app(db_path: str | None = None) -> FastAPI:
    path = db_path or os.environ.get("TRACE_DB_PATH", DEFAULT_DB)
    store = Store(path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = store
        store.mark_interrupted()
        yield

    app = FastAPI(
        title="Sample Contamination Trace API",
        version="1.0.0",
        description=(
            "Trace contamination reachability across sample transfer batches.\n\n"
            "* A transfer batch is contaminated when it carries a source sample,"
            " or when it leaves a site within 24 hours of a contaminated batch"
            " arriving there.\n"
            "* A sample is affected when it rides on a contaminated batch. Its"
            " reported path is a full chain from a source-carrying batch,"
            " `first_arrival_at` is the earliest contaminated arrival with the"
            " sample, and `quarantine_until` is that arrival + 24 hours.\n"
            "* Corrections are versioned (`seen_version` optimistic lock),"
            " replace the old record and invalidate snapshots pinned to it.\n"
            "* `POST /traces` pins a snapshot at acceptance time and publishes"
            " results only when complete; failed traces expose no partial rows.\n"
            "* `POST /quality-checks` reviews two probes' readings for one"
            " batch. With ±0.5 °C measurement error and linear variation"
            " between readings it reports lower/upper bounds on the longest"
            " excursion above 8 °C. A release requires complete coverage by"
            " both probes and both upper bounds within 15 minutes; corrections"
            " invalidate stale verdicts."
        ),
        lifespan=lifespan,
        openapi_tags=[
            {"name": "intake", "description": "Register samples and batches."},
            {"name": "corrections", "description": "Versioned batch replacement."},
            {"name": "traces", "description": "Contamination trace snapshots."},
            {
                "name": "quality",
                "description": "Cold-chain temperature evidence per batch.",
            },
        ],
    )
    app.state.store = store

    # ------------------------------------------------------------- error shape

    def error(
        status_code: int, code: str, message: str, details: Any = None
    ) -> JSONResponse:
        body = ErrorBody(code=code, message=message, details=details)
        return JSONResponse(
            status_code=status_code, content=body.model_dump(exclude_none=True)
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return error(
            422,
            "validation_error",
            "request validation failed",
            {"errors": _jsonable_errors(exc.errors())},
        )

    @app.exception_handler(HTTPException)
    async def _http_error_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        if isinstance(exc.detail, dict) and {"code", "message"} <= set(exc.detail):
            return JSONResponse(
                status_code=exc.status_code, content=exc.detail, headers=exc.headers
            )
        return error(exc.status_code, "http_error", str(exc.detail))

    # ---------------------------------------------------------------- helpers

    def _trace_response(bundle: Any) -> TraceResponse:
        trace = bundle["trace"]
        invalidations = bundle["invalidations"]
        if trace["status"] == "complete" and not invalidations:
            affected = [
                AffectedSample(
                    sample_id=row["sample_id"],
                    first_arrival_at=parse_iso(row["first_arrival"]),
                    quarantine_until=parse_iso(row["quarantine_until"]),
                    path=[PathLeg(**leg) for leg in _load_json(row["path_json"])],
                )
                for row in bundle["results"]
            ]
        else:
            # Failed and invalidated snapshots never expose rows.
            affected = []
        return TraceResponse(
            trace_id=int(trace["id"]),
            status=trace["status"],
            has_sources=bool(_load_json(trace["sources_json"])),
            source_sample_ids=_load_json(trace["sources_json"]),
            affected_samples=affected,
            affected_count=len(affected),
            snapshot=SnapshotInfo(
                intake_id=int(trace["intake_id"]),
                accepted_at=parse_iso(trace["accepted_at"]),
                batch_versions={
                    k: int(v) for k, v in bundle["pinned"].items()
                },
                batches=[BatchOut(**row) for row in bundle["batches"]],
            ),
            error=trace["error"],
        )

    def _interval_out(item: dict[str, Any]) -> IntervalOut:
        return IntervalOut(
            start=parse_iso(item["start"]),
            end=parse_iso(item["end"]),
            duration_minutes=float(item["duration_minutes"]),
        )

    def _quality_pending(check_id: int) -> HTTPException:
        """A check in flight has no published verdict yet."""
        return HTTPException(
            status_code=409,
            detail={
                "code": "quality_check_pending",
                "message": "quality check has not been published yet; retry later",
                "details": {"check_id": check_id, "status": "pending"},
            },
        )

    def _quality_response(bundle: dict[str, Any]) -> QualityCheckResponse:
        check = bundle["check"]
        result = bundle["result"]
        batch = bundle["batch"]
        if batch is None:  # pragma: no cover - pinned rows are never deleted
            raise HTTPException(status_code=500, detail="pinned batch missing")
        probes = [
            ProbeQualityResult(
                probe_id=probe["probe_id"],
                reading_count=int(probe["reading_count"]),
                first_reading_at=parse_iso(probe["first_reading_at"]),
                last_reading_at=parse_iso(probe["last_reading_at"]),
                covers_start=bool(probe["covers_start"]),
                covers_end=bool(probe["covers_end"]),
                complete_coverage=bool(probe["complete_coverage"]),
                evaluated_minutes=float(probe["evaluated_minutes"]),
                lower_bound_minutes=float(probe["lower_bound_minutes"]),
                upper_bound_minutes=float(probe["upper_bound_minutes"]),
                guaranteed_excursions=[
                    _interval_out(item)
                    for item in probe["guaranteed_excursions"]
                ],
                possible_excursions=[
                    _interval_out(item) for item in probe["possible_excursions"]
                ],
                evidence_gaps=[
                    _interval_out(item) for item in probe["evidence_gaps"]
                ],
            )
            for probe in result["probes"]
        ]
        return QualityCheckResponse(
            check_id=int(check["id"]),
            status=check["status"],
            releasable=bool(check["releasable"]),
            reason=result["reason"],
            batch=QualityBatchRef(
                batch_no=check["batch_no"],
                version=int(check["batch_version"]),
                departed_at=parse_iso(check["departed_at"]),
                arrived_at=parse_iso(check["arrived_at"]),
                origin=batch["origin"],
                destination=batch["destination"],
            ),
            created_at=parse_iso(check["created_at"]),
            thresholds={
                key: float(value)
                for key, value in result["thresholds"].items()
            },
            anomalous_probe_ids=list(result["anomalous_probe_ids"]),
            incomplete_probe_ids=list(result["incomplete_probe_ids"]),
            uncertain_probe_ids=list(result["uncertain_probe_ids"]),
            probes=probes,
        )

    # ----------------------------------------------------------------- routes

    @app.get("/healthz", tags=["intake"], summary="Readiness probe")
    def healthz() -> dict[str, str]:
        store.dataset()  # touches the database
        return {"status": "ok"}

    @app.post(
        "/intakes",
        response_model=IntakeResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["intake"],
        summary="Register samples and transfer batches",
        responses={
            409: {"model": ErrorBody, "description": "Conflicting batch values"},
            422: {"model": ErrorBody, "description": "Invalid payload"},
        },
    )
    def create_intake(payload: IntakeRequest) -> IntakeResponse:
        samples, batches, explicit_sources = payload.normalised()
        try:
            result = store.ingest(
                samples=samples,
                batches=batches,
                source_sample_ids=explicit_sources,
            )
        except ConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except InvalidReferenceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store.log("intake", {"samples": len(samples), "batches": len(batches)})
        return IntakeResponse(**result)

    @app.get(
        "/dataset",
        response_model=DatasetResponse,
        tags=["intake"],
        summary="Inspect the active dataset",
        responses={404: {"model": ErrorBody, "description": "No dataset yet"}},
    )
    def get_dataset() -> DatasetResponse:
        dataset = store.dataset()
        if dataset is None:
            raise HTTPException(status_code=404, detail="no dataset")
        return DatasetResponse(**dataset)

    @app.post(
        "/batches/{batch_no}/corrections",
        response_model=CorrectionResponse,
        tags=["corrections"],
        summary="Replace a batch version and invalidate dependent results",
        responses={
            404: {"model": ErrorBody, "description": "Unknown batch"},
            409: {"model": ErrorBody, "description": "Stale version or no-op"},
        },
    )
    def correct_batch(
        batch_no: str, payload: BatchCorrection
    ) -> CorrectionResponse:
        try:
            result = store.correct_batch(
                batch_no=batch_no,
                seen_version=payload.seen_version,
                origin=payload.origin,
                destination=payload.destination,
                departed_at=payload.departed_at,
                arrived_at=payload.arrived_at,
            )
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        store.log(
            "correction",
            {
                "batch_no": batch_no,
                "previous_version": result["previous_version"],
                "version": result["version"],
                "invalidated": result["invalidated_trace_ids"],
            },
        )
        return CorrectionResponse(**result)

    @app.post(
        "/traces",
        response_model=TraceResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["traces"],
        summary="Compute contamination reachability on a pinned snapshot",
        responses={
            404: {"model": ErrorBody, "description": "No dataset or unknown source"},
            409: {"model": ErrorBody, "description": "Snapshot became stale"},
            500: {"model": ErrorBody, "description": "Trace failed; nothing published"},
        },
    )
    def run_trace(payload: TraceRequest | None = None) -> TraceResponse:
        request = payload or TraceRequest()
        try:
            trace_id, snapshot, sources = store.accept_trace(
                sources=request.source_sample_ids
            )
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except InvalidReferenceError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        try:
            outcome = tracing.compute(
                samples=snapshot["samples"],
                batches=snapshot["batches"],
                source_sample_ids=sources,
            )
            store.publish_results(
                trace_id, outcome["affected_samples"], pinned=snapshot["pinned"]
            )
        except ConflictError as exc:
            # Snapshot changed under us: refusal, not a partial result.
            store.fail_trace(trace_id, str(exc))
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "snapshot_changed",
                    "message": str(exc),
                    "details": {"trace_id": trace_id, "status": "failed"},
                },
            ) from exc
        except Exception as exc:  # noqa: BLE001 - recorded, never half-published
            store.fail_trace(trace_id, f"{type(exc).__name__}: {exc}")
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "trace_failed",
                    "message": f"{type(exc).__name__}: {exc}",
                    "details": {"trace_id": trace_id, "status": "failed"},
                },
            ) from exc

        stored = store.read_trace(trace_id)
        assert stored is not None
        store.log(
            "trace",
            {"trace_id": trace_id, "affected": len(outcome["affected_samples"])},
        )
        return _trace_response(stored)

    @app.get(
        "/traces",
        response_model=TraceListResponse,
        tags=["traces"],
        summary="List accepted traces",
    )
    def list_traces() -> TraceListResponse:
        rows = store.trace_history()
        return TraceListResponse(
            traces=[
                TraceSummary(
                    trace_id=int(row["id"]),
                    status=row["status"],
                    accepted_at=parse_iso(row["accepted_at"]),
                    completed_at=parse_iso(row["completed_at"])
                    if row["completed_at"]
                    else None,
                    source_sample_ids=_load_json(row["sources_json"]),
                    affected_count=int(row["result_count"]),
                    invalidated=int(row["invalidation_count"]) > 0,
                )
                for row in rows
            ]
        )

    @app.get(
        "/traces/{trace_id}",
        response_model=TraceResponse,
        tags=["traces"],
        summary="Read one pinned snapshot result",
        responses={
            404: {"model": ErrorBody, "description": "Unknown trace"},
            409: {"model": ErrorBody, "description": "Snapshot invalidated"},
        },
    )
    def get_trace(trace_id: int) -> TraceResponse:
        bundle = store.read_trace(trace_id)
        if bundle is None:
            raise HTTPException(status_code=404, detail="trace not found")
        trace = bundle["trace"]
        if bundle["invalidations"]:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "snapshot_invalidated",
                    "message": (
                        "this snapshot was invalidated by a batch correction;"
                        " run a new trace"
                    ),
                    "details": {
                        "trace_id": trace_id,
                        "invalidations": [
                            InvalidationInfo(
                                batch_no=row["batch_no"],
                                version=int(row["version"]),
                                invalidated_at=parse_iso(row["invalidated_at"]),
                                reason=row["reason"],
                            ).model_dump(mode="json")
                            for row in bundle["invalidations"]
                        ],
                    },
                },
            )
        if trace["status"] == "failed":
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "trace_failed",
                    "message": trace["error"] or "trace failed",
                    "details": {"trace_id": trace_id, "status": "failed"},
                },
            )
        if trace["status"] != "complete":
            # A pending trace has no published rows. Returning an empty list
            # with 200 would be indistinguishable from "nothing affected".
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "trace_pending",
                    "message": "trace has not been published yet; retry later",
                    "details": {"trace_id": trace_id, "status": trace["status"]},
                },
            )
        return _trace_response(bundle)

    @app.get(
        "/traces/{trace_id}/invalidations",
        response_model=list[InvalidationInfo],
        tags=["traces"],
        summary="Invalidation records for a trace",
    )
    def get_invalidations(trace_id: int) -> list[InvalidationInfo]:
        if store.trace(trace_id) is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return [
            InvalidationInfo(
                batch_no=row["batch_no"],
                version=int(row["version"]),
                invalidated_at=parse_iso(row["invalidated_at"]),
                reason=row["reason"],
            )
            for row in store.invalidations(trace_id)
        ]

    @app.post(
        "/quality-checks",
        response_model=QualityCheckResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["quality"],
        summary="Review a batch's cold-chain evidence from two probes",
        responses={
            404: {"model": ErrorBody, "description": "No dataset or unknown batch"},
            409: {
                "model": ErrorBody,
                "description": "Batch corrected during verification; nothing recorded",
            },
            500: {
                "model": ErrorBody,
                "description": "Verification failed; nothing recorded",
            },
        },
    )
    def create_quality_check(payload: QualityCheckRequest) -> QualityCheckResponse:
        batch_no, probes = payload.normalised()
        try:
            intake_id, pinned = store.begin_quality_check(batch_no=batch_no)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        try:
            outcome = quality.evaluate(
                batch={
                    "batch_no": pinned["batch_no"],
                    "departed_at": pinned["departed_at"],
                    "arrived_at": pinned["arrived_at"],
                },
                probes=probes,
            )
            check_id = store.publish_quality_check(
                intake_id=intake_id,
                pinned=pinned,
                probes=probes,
                status=outcome["status"],
                releasable=outcome["releasable"],
                result=_quality_result_json(outcome),
            )
        except ConflictError as exc:
            # A correction slipped in between pinning and publication. The
            # verdict would describe superseded values, so nothing is stored.
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "snapshot_changed",
                    "message": str(exc),
                    "details": {
                        "batch_no": pinned["batch_no"],
                        "pinned_version": pinned["version"],
                    },
                },
            ) from exc
        except quality.QualityFailure as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "quality_check_failed",
                    "message": str(exc),
                    "details": {"batch_no": pinned["batch_no"]},
                },
            ) from exc
        except Exception as exc:  # noqa: BLE001 - never record a partial verdict
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "quality_check_failed",
                    "message": f"{type(exc).__name__}: {exc}",
                    "details": {"batch_no": pinned["batch_no"]},
                },
            ) from exc

        bundle = store.read_quality_check(check_id)
        assert bundle is not None
        if bundle["invalidations"]:
            # A correction committed right after publication. Never hand back
            # a release conclusion that is already superseded.
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "snapshot_changed",
                    "message": (
                        "the batch was corrected while the verdict was being"
                        " recorded; the stale verdict was not returned"
                    ),
                    "details": {
                        "check_id": check_id,
                        "batch_no": pinned["batch_no"],
                        "pinned_version": pinned["version"],
                    },
                },
            )
        store.log(
            "quality_check",
            {
                "check_id": check_id,
                "batch_no": pinned["batch_no"],
                "version": pinned["version"],
                "status": outcome["status"],
            },
        )
        return _quality_response(bundle)
    @app.get(
        "/quality-checks",
        response_model=QualityCheckListResponse,
        tags=["quality"],
        summary="List recorded quality checks",
    )
    def list_quality_checks(
        batch_no: str | None = None,
    ) -> QualityCheckListResponse:
        rows = store.quality_check_history(batch_no=batch_no)
        return QualityCheckListResponse(
            checks=[
                QualityCheckSummary(
                    check_id=int(row["id"]),
                    batch_no=row["batch_no"],
                    batch_version=int(row["batch_version"]),
                    status=row["status"],
                    releasable=bool(row["releasable"]),
                    invalidated=store.quality_check_invalidated(int(row["id"])),
                    created_at=parse_iso(row["created_at"]),
                )
                for row in rows
            ]
        )

    @app.get(
        "/quality-checks/{check_id}",
        response_model=QualityCheckResponse,
        tags=["quality"],
        summary="Read one quality-check verdict",
        responses={
            404: {"model": ErrorBody, "description": "Unknown check"},
            409: {
                "model": ErrorBody,
                "description": "Verdict invalidated by a batch correction",
            },
        },
    )
    def get_quality_check(check_id: int) -> QualityCheckResponse:
        bundle = store.read_quality_check(check_id)
        if bundle is None:
            raise HTTPException(status_code=404, detail="quality check not found")
        if bundle["check"]["status"] == "pending":
            raise _quality_pending(check_id)
        if bundle["invalidations"]:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "quality_check_invalidated",
                    "message": (
                        "the batch was corrected after this verdict was"
                        " recorded; run a new quality check"
                    ),
                    "details": {
                        "check_id": check_id,
                        "invalidations": [
                            InvalidationInfo(
                                batch_no=row["batch_no"],
                                version=int(row["version"]),
                                invalidated_at=parse_iso(row["invalidated_at"]),
                                reason=row["reason"],
                            ).model_dump(mode="json")
                            for row in bundle["invalidations"]
                        ],
                    },
                },
            )
        return _quality_response(bundle)

    @app.get(
        "/quality-checks/{check_id}/invalidations",
        response_model=list[InvalidationInfo],
        tags=["quality"],
        summary="Invalidation records for a quality check",
    )
    def get_quality_check_invalidations(check_id: int) -> list[InvalidationInfo]:
        bundle = store.read_quality_check(check_id)
        if bundle is None:
            raise HTTPException(status_code=404, detail="quality check not found")
        if bundle["check"]["status"] == "pending":
            raise _quality_pending(check_id)
        return [
            InvalidationInfo(
                batch_no=row["batch_no"],
                version=int(row["version"]),
                invalidated_at=parse_iso(row["invalidated_at"]),
                reason=row["reason"],
            )
            for row in bundle["invalidations"]
        ]

    return app


def _quality_result_json(outcome: dict[str, Any]) -> dict[str, Any]:
    """Serialise the evaluator's verdict into the stored/response shape."""

    def render(intervals: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                "start": item.start.isoformat(),
                "end": item.end.isoformat(),
                "duration_minutes": item.minutes,
            }
            for item in intervals
        ]

    return {
        "reason": outcome["reason"],
        "thresholds": outcome["thresholds"],
        "anomalous_probe_ids": outcome["anomalous_probe_ids"],
        "incomplete_probe_ids": outcome["incomplete_probe_ids"],
        "uncertain_probe_ids": outcome["uncertain_probe_ids"],
        "probes": [
            {
                "probe_id": probe["probe_id"],
                "reading_count": probe["reading_count"],
                "first_reading_at": probe["first_reading_at"].isoformat(),
                "last_reading_at": probe["last_reading_at"].isoformat(),
                "covers_start": probe["covers_start"],
                "covers_end": probe["covers_end"],
                "complete_coverage": probe["complete_coverage"],
                "evaluated_minutes": probe["evaluated_minutes"],
                "lower_bound_minutes": probe["lower_bound_minutes"],
                "upper_bound_minutes": probe["upper_bound_minutes"],
                "guaranteed_excursions": render(probe["guaranteed_excursions"]),
                "possible_excursions": render(probe["possible_excursions"]),
                "evidence_gaps": render(probe["evidence_gaps"]),
            }
            for probe in outcome["probes"]
        ],
    }


def _load_json(value: str) -> Any:
    return json.loads(value)


def _jsonable_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make pydantic errors JSON-safe (ctx may hold exception objects)."""
    cleaned = []
    for item in errors:
        entry = {k: v for k, v in item.items() if k != "ctx"}
        if "ctx" in item:
            entry["ctx"] = {
                k: str(v) if isinstance(v, BaseException) else v
                for k, v in item["ctx"].items()
            }
        entry["loc"] = list(entry.get("loc", ()))
        cleaned.append(entry)
    return cleaned


app = create_app()
