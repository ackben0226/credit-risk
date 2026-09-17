"""
Ingestion pipeline: raw CSV → interim Parquet.

This module converts the Home Credit raw CSVs into typed Parquet files
under data/interim/. Parquet is faster to read, preserves dtypes, and
gives downstream stages a stable format that doesn't re-parse on every
load.

Design notes
------------
- Validation is a precondition. If validation fails, no Parquet is
  written. The enforcement layer is the gate to ingestion.
- Lineage is preserved via a manifest at artifacts/reports/ingestion_<ts>.json.
  The manifest records:
    - Input CSV SHA-256 (as recorded by the most recent inspection)
    - Output Parquet SHA-256
    - Row count, column count, dtypes
    - Encoding used
    - Duration per file
    - Source inspection report path and timestamp
- Each run overwrites data/interim/<table>.parquet. Historical runs are
  not preserved on disk; they are recoverable from the manifest, which
  points to the input CSV hash.
- The encoding fallback chain matches scripts/inspect_data.py and
  src/credit_risk/data/validate.py. UTF-8 first, then cp1252, then
  latin-1 (which never fails on bytes).
- Ingestion verifies that the raw file SHA-256 matches the inspection
  record. If the raw data changed since inspection, contracts are stale
  and ingestion fails. This is the guard against the stale-artifact
  problem that plagues naive pipelines.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import yaml

from credit_risk.data.validate import (
    JoinContract,
    TableContract,
    ValidationReport,
    Validator,
    load_contracts,
)


logger = logging.getLogger("credit_risk.ingest")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IngestionConfig:
    """Resolved configuration for an ingestion run."""

    project_root: Path
    raw_dir: Path
    interim_dir: Path
    contracts_dir: Path
    reports_dir: Path
    inspection_path: Path
    parquet_compression: str = "snappy"
    parquet_engine: str = "pyarrow"

    def raw_file(self, filename: str) -> Path:
        return self.raw_dir / filename

    def interim_file(self, table_name: str) -> Path:
        return self.interim_dir / f"{table_name}.parquet"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class FileIngestionResult:
    """Result of ingesting one table."""

    table: str
    input_file: str
    output_file: str
    input_sha256: str
    output_sha256: str
    row_count: int
    column_count: int
    encoding_used: str
    dtypes: dict[str, str]
    duration_seconds: float
    parquet_size_bytes: int
    overwrote_existing: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "input_file": self.input_file,
            "output_file": self.output_file,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "encoding_used": self.encoding_used,
            "dtypes": self.dtypes,
            "duration_seconds": round(self.duration_seconds, 3),
            "parquet_size_bytes": self.parquet_size_bytes,
            "overwrote_existing": self.overwrote_existing,
        }


@dataclass
class IngestionReport:
    """Aggregated ingestion output across all tables."""

    generated_at: str
    source_inspection: str
    source_inspection_generated_at: str
    source_contracts_dir: str
    validation_passed: bool
    validation_checks_run: int
    validation_error_count: int
    results: list[FileIngestionResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "source_inspection": self.source_inspection,
            "source_inspection_generated_at": self.source_inspection_generated_at,
            "source_contracts_dir": self.source_contracts_dir,
            "validation": {
                "passed": self.validation_passed,
                "checks_run": self.validation_checks_run,
                "error_count": self.validation_error_count,
            },
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Ingestor
# ---------------------------------------------------------------------------

@dataclass
class Ingestor:
    """
    Converts raw CSVs to interim Parquet with validation as a precondition.

    Fails loudly on validation failure — no Parquet is written. Verifies
    that raw file SHA-256 matches the inspection record before ingesting.
    """

    # Class constant — not a dataclass field. ClassVar tells the dataclass
    # machinery to ignore it when generating __init__.
    ENCODING_CHAIN: ClassVar[tuple[str, ...]] = ("utf-8", "cp1252", "latin-1")

    config: IngestionConfig
    contracts: list[TableContract]
    joins: list[JoinContract]
    inspection: dict[str, Any]
    report: IngestionReport | None = None

    # ---- public API ------------------------------------------------------

    def run(self) -> IngestionReport:
        """Execute ingestion. Returns the aggregated report."""
        self._ensure_directories()

        validation = self._run_validation_precondition()
        if not validation.passed:
            logger.error(
                "Validation failed: %d errors. Ingestion aborted. "
                "No Parquet files written.",
                validation.error_count,
            )
            self.report = self._build_report(validation, results=[])
            self._write_report(self.report)
            return self.report

        self._verify_input_hashes()

        results: list[FileIngestionResult] = []
        for contract in self.contracts:
            logger.info("Ingesting %s", contract.name)
            result = self._ingest_table(contract)
            results.append(result)

        self.report = self._build_report(validation, results=results)
        self._write_report(self.report)
        self._log_summary(self.report)
        return self.report

    # ---- preconditions ---------------------------------------------------

    def _ensure_directories(self) -> None:
        self.config.interim_dir.mkdir(parents=True, exist_ok=True)
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)

    def _run_validation_precondition(self) -> ValidationReport:
        logger.info("Running validation as ingestion precondition")
        validator = Validator(
            raw_dir=self.config.raw_dir,
            tables=self.contracts,
            joins=self.joins,
        )
        return validator.run()

    def _verify_input_hashes(self) -> None:
        """
        Verify that each raw CSV still matches the SHA-256 recorded during
        inspection. If any file has changed, contracts are stale and
        ingestion fails — no silent ingestion of un-inspected data.
        """
        logger.info("Verifying raw file integrity against inspection")
        mismatches: list[str] = []

        for contract in self.contracts:
            path = self.config.raw_file(contract.filename)
            if not path.exists():
                mismatches.append(f"{contract.name}: file missing ({path})")
                continue

            expected = self._inspection_sha(contract.name)
            if expected is None:
                logger.warning(
                    "No SHA-256 recorded for '%s' in inspection — skipping check",
                    contract.name,
                )
                continue

            observed = self._sha256_of_file(path)
            if observed != expected:
                mismatches.append(
                    f"{contract.name}: expected {expected[:12]}…, "
                    f"observed {observed[:12]}…"
                )

        if mismatches:
            details = "\n  ".join(mismatches)
            raise RuntimeError(
                "Raw file(s) differ from inspection record. Contracts are "
                f"stale. Re-run inspection and rebuild contracts.\n  {details}"
            )

        logger.info("All raw files match inspection record")

    def _inspection_sha(self, table: str) -> str | None:
        """Retrieve the input CSV SHA-256 from the inspection report."""
        file_entry = self.inspection.get("files", {}).get(table, {})
        sha = file_entry.get("sha256")
        return sha if isinstance(sha, str) else None

    # ---- per-table ingestion --------------------------------------------

    def _ingest_table(self, contract: TableContract) -> FileIngestionResult:
        start = time.monotonic()

        input_path = self.config.raw_file(contract.filename)
        output_path = self.config.interim_file(contract.name)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Raw file for '{contract.name}' not found: {input_path}"
            )

        overwrote = output_path.exists()

        df, encoding_used = self._read_csv_with_encoding_report(
            input_path, low_memory=False
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(
            output_path,
            engine=self.config.parquet_engine,
            compression=self.config.parquet_compression,
            index=False,
        )

        duration = time.monotonic() - start

        input_sha = self._inspection_sha(contract.name) or ""
        output_sha = self._sha256_of_file(output_path)

        result = FileIngestionResult(
            table=contract.name,
            input_file=contract.filename,
            output_file=str(output_path.relative_to(self.config.project_root)),
            input_sha256=input_sha,
            output_sha256=output_sha,
            row_count=len(df),
            column_count=len(df.columns),
            encoding_used=encoding_used,
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            duration_seconds=duration,
            parquet_size_bytes=output_path.stat().st_size,
            overwrote_existing=overwrote,
        )

        logger.info(
            "  %s: %d rows, %d cols, %s, %.2fs, %.1f MB%s",
            contract.name,
            result.row_count,
            result.column_count,
            encoding_used,
            result.duration_seconds,
            result.parquet_size_bytes / 1e6,
            " (overwrote)" if overwrote else "",
        )
        return result

    # ---- encoding-tolerant CSV reading ----------------------------------

    def _read_csv_with_encoding_report(
        self,
        path: Path,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, str]:
        """
        Read a CSV, trying each encoding in ENCODING_CHAIN.

        Returns the DataFrame and the encoding that succeeded, so the
        caller can record it in the ingestion manifest.
        """
        last_error: Exception | None = None
        for encoding in self.ENCODING_CHAIN:
            try:
                df = pd.read_csv(path, encoding=encoding, **kwargs)
                return df, encoding
            except UnicodeDecodeError as exc:
                last_error = exc
                logger.debug(
                    "Encoding '%s' failed for %s — trying next",
                    encoding, path.name,
                )
                continue
        raise RuntimeError(f"All encodings failed for {path.name}") from last_error

    # ---- SHA-256 helpers -------------------------------------------------

    @staticmethod
    def _sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()

    # ---- report assembly & persistence ----------------------------------

    def _build_report(
        self,
        validation: ValidationReport,
        results: list[FileIngestionResult],
    ) -> IngestionReport:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return IngestionReport(
            generated_at=timestamp,
            source_inspection=str(
                self.config.inspection_path.relative_to(self.config.project_root)
            ),
            source_inspection_generated_at=self.inspection.get("generated_at", ""),
            source_contracts_dir=str(
                self.config.contracts_dir.relative_to(self.config.project_root)
            ),
            validation_passed=validation.passed,
            validation_checks_run=validation.checks_run,
            validation_error_count=validation.error_count,
            results=results,
        )

    def _report_path(self, timestamp: str) -> Path:
        return self.config.reports_dir / f"ingestion_{timestamp}.json"

    def _write_report(self, report: IngestionReport) -> Path:
        out_path = self._report_path(report.generated_at)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info(
            "Ingestion report written: %s",
            out_path.relative_to(self.config.project_root),
        )
        return out_path

    # ---- summary logging -------------------------------------------------

    def _log_summary(self, report: IngestionReport) -> None:
        logger.info("=" * 70)
        logger.info("Ingestion complete")
        logger.info(
            "  Validation: %s (%d checks, %d errors)",
            "PASSED" if report.validation_passed else "FAILED",
            report.validation_checks_run,
            report.validation_error_count,
        )
        logger.info("  Tables ingested: %d", len(report.results))
        total_rows = sum(r.row_count for r in report.results)
        total_bytes = sum(r.parquet_size_bytes for r in report.results)
        total_seconds = sum(r.duration_seconds for r in report.results)
        logger.info("  Total rows: %d", total_rows)
        logger.info("  Total Parquet size: %.1f MB", total_bytes / 1e6)
        logger.info("  Total duration: %.1f s", total_seconds)
        for r in report.results:
            logger.info(
                "    %-24s rows=%12d size=%9.1f MB enc=%s",
                r.table,
                r.row_count,
                r.parquet_size_bytes / 1e6,
                r.encoding_used,
            )
        logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Configuration builder
# ---------------------------------------------------------------------------

def build_config(project_root: Path) -> IngestionConfig:
    """Construct IngestionConfig from configs/data_config.yaml."""
    config_path = project_root / "configs" / "data_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    paths = raw.get("paths", {})
    inspection_block = raw.get("inspection", {})
    inspection_rel = inspection_block.get("report_path")
    if not inspection_rel:
        raise RuntimeError(
            "configs/data_config.yaml missing 'inspection.report_path'. "
            "Run scripts/inspect_data.py first."
        )

    return IngestionConfig(
        project_root=project_root,
        raw_dir=project_root / paths["raw_dir"],
        interim_dir=project_root / paths["interim_dir"],
        contracts_dir=project_root / "configs" / "contracts",
        reports_dir=project_root / paths["reports_dir"],
        inspection_path=project_root / inspection_rel,
    )


def load_inspection(path: Path) -> dict[str, Any]:
    """Load the inspection JSON referenced by the config."""
    if not path.exists():
        raise FileNotFoundError(f"Inspection report not found: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def ingest_all(project_root: Path) -> IngestionReport:
    """
    Top-level convenience function. Reads config, loads contracts and
    inspection, runs the Ingestor, and returns the report.
    """
    config = build_config(project_root)
    contracts, joins = load_contracts(config.contracts_dir)
    inspection = load_inspection(config.inspection_path)

    ingestor = Ingestor(
        config=config,
        contracts=contracts,
        joins=joins,
        inspection=inspection,
    )
    return ingestor.run()