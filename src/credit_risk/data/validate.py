"""
Schema, quality, and relational contracts for the Home Credit raw data.

This module is the *validation* layer of the pipeline:

    Inspection (once)  →  Contracts (this module)  →  Validation (every run)

Contracts are derived from an inspection JSON produced by
`scripts/inspect_data.py` and persisted as YAML under `configs/contracts/`.
They are reviewed once and then treated as the source of truth. If raw
data is refreshed and inspection reveals changes, the contract builder
regenerates the YAML, producing a diff for human review.

Validation runs on every ingestion and on every CI pass. It **asserts**
and **fails loudly**. It does not describe or discover.

Design notes
------------
- Tolerances are explicit. Exact equality on live data is a false-positive
  factory. Every numeric assertion carries a tolerance.
- Failures accumulate. The validator collects all violations across all
  tables and reports them together, rather than raising on the first.
- Contracts are dataclasses. They serialise to and from YAML without
  bespoke parsing code.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


logger = logging.getLogger("credit_risk.validate")


# ---------------------------------------------------------------------------
# Encoding fallback (mirrors scripts/inspect_data.py)
# ---------------------------------------------------------------------------

ENCODING_CHAIN: tuple[str, ...] = ("utf-8", "cp1252", "latin-1")


def read_csv_with_fallback(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV, trying each encoding in ENCODING_CHAIN."""
    last_error: Exception | None = None
    for encoding in ENCODING_CHAIN:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise RuntimeError(f"All encodings failed for {path.name}") from last_error


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnContract:
    """
    Expected properties of a single column.

    All numeric tolerances are relative fractions (0.02 = 2%), applied
    symmetrically around the reference value unless noted otherwise.
    """

    name: str
    dtype: str
    nullable: bool
    null_rate_reference: float | None = None
    null_rate_tolerance: float = 0.02
    is_unique: bool = False
    allowed_values: tuple[Any, ...] | None = None
    min_value: float | None = None
    max_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "nullable": self.nullable,
            "null_rate_reference": self.null_rate_reference,
            "null_rate_tolerance": self.null_rate_tolerance,
            "is_unique": self.is_unique,
            "allowed_values": list(self.allowed_values) if self.allowed_values else None,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColumnContract":
        allowed = data.get("allowed_values")
        return cls(
            name=data["name"],
            dtype=data["dtype"],
            nullable=bool(data["nullable"]),
            null_rate_reference=data.get("null_rate_reference"),
            null_rate_tolerance=float(data.get("null_rate_tolerance", 0.02)),
            is_unique=bool(data.get("is_unique", False)),
            allowed_values=tuple(allowed) if allowed else None,
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
        )


@dataclass(frozen=True)
class TableContract:
    """Expected properties of a single CSV table."""

    name: str
    filename: str
    primary_key: str | None
    row_count_reference: int
    row_count_tolerance: float
    columns: tuple[ColumnContract, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "filename": self.filename,
            "primary_key": self.primary_key,
            "row_count_reference": self.row_count_reference,
            "row_count_tolerance": self.row_count_tolerance,
            "columns": [c.to_dict() for c in self.columns],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TableContract":
        return cls(
            name=data["name"],
            filename=data["filename"],
            primary_key=data.get("primary_key"),
            row_count_reference=int(data["row_count_reference"]),
            row_count_tolerance=float(data["row_count_tolerance"]),
            columns=tuple(ColumnContract.from_dict(c) for c in data["columns"]),
        )


@dataclass(frozen=True)
class JoinContract:
    """Expected properties of a child → parent join."""

    child: str
    child_key: str
    parent: str
    parent_key: str
    orphan_rate_reference: float
    orphan_rate_tolerance: float = 0.02

    def to_dict(self) -> dict[str, Any]:
        return {
            "child": self.child,
            "child_key": self.child_key,
            "parent": self.parent,
            "parent_key": self.parent_key,
            "orphan_rate_reference": self.orphan_rate_reference,
            "orphan_rate_tolerance": self.orphan_rate_tolerance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JoinContract":
        return cls(
            child=data["child"],
            child_key=data["child_key"],
            parent=data["parent"],
            parent_key=data["parent_key"],
            orphan_rate_reference=float(data["orphan_rate_reference"]),
            orphan_rate_tolerance=float(data.get("orphan_rate_tolerance", 0.02)),
        )


# ---------------------------------------------------------------------------
# Violations & reports
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Violation:
    """A single contract failure."""

    table: str
    check: str
    column: str | None
    expected: str
    observed: str
    severity: str = "error"          # "error" | "warning"

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "check": self.check,
            "column": self.column,
            "expected": self.expected,
            "observed": self.observed,
            "severity": self.severity,
        }


@dataclass
class ValidationReport:
    """Aggregated validation output across all tables and joins."""

    generated_at: str
    violations: list[Violation] = field(default_factory=list)
    checks_run: int = 0

    @property
    def passed(self) -> bool:
        return not any(v.severity == "error" for v in self.violations)

    @property
    def error_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "checks_run": self.checks_run,
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "violations": [v.to_dict() for v in self.violations],
        }


# ---------------------------------------------------------------------------
# Contract builder — inspection JSON → contracts
# ---------------------------------------------------------------------------

class ContractBuilder:
    """
    Builds TableContract / ColumnContract / JoinContract objects from an
    inspection JSON produced by scripts/inspect_data.py.

    The output YAML files under configs/contracts/ are reviewed once and
    then treated as the source of truth. Regenerating them after a data
    refresh produces a diff for human review.
    """

    # Default tolerances applied to derived contracts
    DEFAULT_ROW_COUNT_TOLERANCE: float = 0.02
    DEFAULT_NULL_RATE_TOLERANCE: float = 0.02
    DEFAULT_ORPHAN_RATE_TOLERANCE: float = 0.02

    # Tables that carry a primary key. Others have no unique-key contract.
    PRIMARY_KEYS: dict[str, str] = {
        "application_train": "SK_ID_CURR",
        "application_test": "SK_ID_CURR",
        "bureau": "SK_ID_BUREAU",
        "previous_application": "SK_ID_PREV",
    }

    # Columns whose observed cardinality is low enough to lock as an
    # allowed-values set. Above this threshold, we skip enumeration.
    MAX_ENUMERATED_CARDINALITY: int = 20

    # Tables excluded from the modelling contract — reference only.
    NON_MODELLING_TABLES: frozenset[str] = frozenset(
        {"sample_submission", "columns_description"}
    )

    def __init__(self, inspection: dict[str, Any]) -> None:
        self.inspection = inspection

    def build_all(self) -> tuple[list[TableContract], list[JoinContract]]:
        tables = [
            self._build_table(name, rep)
            for name, rep in self.inspection["files"].items()
            if name not in self.NON_MODELLING_TABLES
        ]
        joins = [
            self._build_join(j) for j in self.inspection["relational_integrity"]["joins"]
        ]
        return tables, joins

    def _build_table(self, name: str, report: dict[str, Any]) -> TableContract:
        pk = self.PRIMARY_KEYS.get(name)

        columns = tuple(
            self._build_column(col_name, profile, is_pk=(col_name == pk))
            for col_name, profile in report["column_profiles"].items()
        )

        return TableContract(
            name=name,
            filename=report["file"],
            primary_key=pk,
            row_count_reference=int(report["full_row_count"]),
            row_count_tolerance=self.DEFAULT_ROW_COUNT_TOLERANCE,
            columns=columns,
        )

    def _build_column(
        self,
        name: str,
        profile: dict[str, Any],
        *,
        is_pk: bool,
    ) -> ColumnContract:
        null_rate = profile.get("null_rate")
        dtype = profile["dtype"]
        nullable = bool(null_rate and null_rate > 0)

        # Allowed values: prefer the full unique set emitted by inspection.
        allowed: tuple[Any, ...] | None = None
        n_unique = profile.get("n_unique")
        unique_values = profile.get("unique_values")

        if not is_pk:
            if unique_values is not None:
                allowed = tuple(unique_values)
            elif (
                n_unique is not None
                and 0 < n_unique <= self.MAX_ENUMERATED_CARDINALITY
            ):
                sample = profile.get("sample_values", [])
                if sample and len(sample) == n_unique:
                    allowed = tuple(sample)

        numeric = profile.get("numeric", {}) or {}

        return ColumnContract(
            name=name,
            dtype=dtype,
            nullable=nullable,
            null_rate_reference=null_rate,
            null_rate_tolerance=self.DEFAULT_NULL_RATE_TOLERANCE,
            is_unique=is_pk,
            allowed_values=allowed,
            min_value=numeric.get("min"),
            max_value=numeric.get("max"),
        )
    
    def _build_join(self, join: dict[str, Any]) -> JoinContract:
        return JoinContract(
            child=join["child_table"],
            child_key=join["child_key"],
            parent=join["parent_table"],
            parent_key=join["parent_key"],
            orphan_rate_reference=float(join["orphan_rate"]),
            orphan_rate_tolerance=self.DEFAULT_ORPHAN_RATE_TOLERANCE,
        )


# ---------------------------------------------------------------------------
# Contract persistence
# ---------------------------------------------------------------------------

def save_contracts(
    tables: list[TableContract],
    joins: list[JoinContract],
    out_dir: Path,
) -> None:
    """Persist contracts as YAML files under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for table in tables:
        path = out_dir / f"{table.name}.yaml"
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                table.to_dict(),
                f,
                sort_keys=False,
                default_flow_style=False,
            )

    joins_path = out_dir / "joins.yaml"
    with joins_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"joins": [j.to_dict() for j in joins]},
            f,
            sort_keys=False,
        )

    logger.info("Saved %d table contracts and %d join contracts to %s",
                len(tables), len(joins), out_dir)


def load_contracts(
    contracts_dir: Path,
) -> tuple[list[TableContract], list[JoinContract]]:
    """Load contracts previously persisted by save_contracts()."""
    tables: list[TableContract] = []
    for path in sorted(contracts_dir.glob("*.yaml")):
        if path.name == "joins.yaml":
            continue
        with path.open(encoding="utf-8") as f:
            tables.append(TableContract.from_dict(yaml.safe_load(f)))

    joins_path = contracts_dir / "joins.yaml"
    joins: list[JoinContract] = []
    if joins_path.exists():
        with joins_path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        joins = [JoinContract.from_dict(j) for j in raw.get("joins", [])]

    return tables, joins


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

@dataclass
class Validator:
    """
    Runs contracts against raw data and accumulates violations.

    Fails loudly by design: the caller decides what to do with the report.
    The validator itself never raises on a contract failure — only on
    unrecoverable I/O errors.
    """

    raw_dir: Path
    tables: list[TableContract]
    joins: list[JoinContract]
    report: ValidationReport = field(init=False)

    def __post_init__(self) -> None:
        self.report = ValidationReport(
            generated_at=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        )

    # ---- public API ------------------------------------------------------

    def run(self) -> ValidationReport:
        """Execute all table and join checks. Returns the aggregated report."""
        logger.info("Validating %d tables, %d joins", len(self.tables), len(self.joins))

        for table in self.tables:
            self._validate_table(table)

        for join in self.joins:
            self._validate_join(join)

        self._log_summary()
        return self.report

    # ---- per-table checks -----------------------------------------------

    def _validate_table(self, contract: TableContract) -> None:
        path = self.raw_dir / contract.filename
        if not path.exists():
            self._violate(
                table=contract.name,
                check="file_exists",
                column=None,
                expected=str(path),
                observed="missing",
            )
            return

        logger.info("Validating table %s", contract.name)
        df = read_csv_with_fallback(path, low_memory=False)

        self._check_row_count(contract, len(df))
        self._check_columns_present(contract, df)

        for col in contract.columns:
            if col.name not in df.columns:
                continue  # already recorded as a missing-column violation
            self._validate_column(contract.name, col, df[col.name])

    def _check_row_count(self, contract: TableContract, observed: int) -> None:
        ref = contract.row_count_reference
        tol = contract.row_count_tolerance
        lower = ref * (1 - tol)
        upper = ref * (1 + tol)

        self._record_check()

        if not (lower <= observed <= upper):
            self._violate(
                table=contract.name,
                check="row_count",
                column=None,
                expected=f"in [{int(lower):,}, {int(upper):,}] (ref {ref:,} ±{tol:.1%})",
                observed=f"{observed:,}",
            )

    def _check_columns_present(self, contract: TableContract, df: pd.DataFrame) -> None:
        expected = {c.name for c in contract.columns}
        observed = set(df.columns)

        for missing in sorted(expected - observed):
            self._record_check()
            self._violate(
                table=contract.name,
                check="column_present",
                column=missing,
                expected="present",
                observed="missing",
            )

        for extra in sorted(observed - expected):
            self._record_check()
            self._violate(
                table=contract.name,
                check="column_unexpected",
                column=extra,
                expected="not present",
                observed="present",
                severity="warning",
            )

    def _validate_column(
        self,
        table: str,
        contract: ColumnContract,
        series: pd.Series,
    ) -> None:
        # Nullability
        self._record_check()
        n = len(series)
        null_rate = float(series.isna().sum() / n) if n else 0.0

        if not contract.nullable and null_rate > 0:
            self._violate(
                table=table,
                check="nullability",
                column=contract.name,
                expected="no nulls",
                observed=f"{null_rate:.4%} nulls",
            )
        elif contract.null_rate_reference is not None:
            ref = contract.null_rate_reference
            tol = contract.null_rate_tolerance
            if abs(null_rate - ref) > tol:
                self._violate(
                    table=table,
                    check="null_rate",
                    column=contract.name,
                    expected=f"{ref:.4f} ±{tol:.2%}",
                    observed=f"{null_rate:.4f}",
                )

        # Uniqueness
        if contract.is_unique:
            self._record_check()
            n_unique = int(series.nunique(dropna=False))
            if n_unique != n:
                self._violate(
                    table=table,
                    check="uniqueness",
                    column=contract.name,
                    expected=f"{n:,} unique values",
                    observed=f"{n_unique:,}",
                )

        # Allowed values
        if contract.allowed_values is not None:
            self._record_check()
            observed_values = set(series.dropna().unique().tolist())
            unexpected = observed_values - set(contract.allowed_values)
            if unexpected:
                self._violate(
                    table=table,
                    check="allowed_values",
                    column=contract.name,
                    expected=f"subset of {sorted(contract.allowed_values)}",
                    observed=f"unexpected: {sorted(unexpected)[:5]}",
                )

        # Value ranges (only on numeric columns with observed data)
        if (
            pd.api.types.is_numeric_dtype(series)
            and contract.min_value is not None
            and contract.max_value is not None
            and series.notna().any()
        ):
            self._record_check()
            obs_min = float(series.min())
            obs_max = float(series.max())

            # Small tolerance on numeric range — floats drift
            range_tol = 0.02 * (abs(contract.max_value - contract.min_value) or 1.0)
            if obs_min < contract.min_value - range_tol or obs_max > contract.max_value + range_tol:
                self._violate(
                    table=table,
                    check="value_range",
                    column=contract.name,
                    expected=f"[{contract.min_value}, {contract.max_value}]",
                    observed=f"[{obs_min}, {obs_max}]",
                )

    # ---- per-join checks -------------------------------------------------

    def _validate_join(self, contract: JoinContract) -> None:
        child_path = self._table_filename(contract.child)
        parent_path = self._table_filename(contract.parent)
        if child_path is None or parent_path is None:
            self._violate(
                table=contract.child,
                check="join_tables_resolvable",
                column=None,
                expected=f"{contract.child} → {contract.parent}",
                observed="one or both table contracts missing",
            )
            return

        if not (self.raw_dir / child_path).exists() or not (self.raw_dir / parent_path).exists():
            return  # file-level failure already recorded

        logger.info("Validating join %s.%s → %s.%s",
                    contract.child, contract.child_key,
                    contract.parent, contract.parent_key)

        child_keys = self._load_unique(child_path, contract.child_key)
        parent_keys = self._load_unique(parent_path, contract.parent_key)

        orphans = np.setdiff1d(child_keys, parent_keys, assume_unique=True)
        orphan_rate = len(orphans) / max(len(child_keys), 1)

        self._record_check()
        ref = contract.orphan_rate_reference
        tol = contract.orphan_rate_tolerance
        if abs(orphan_rate - ref) > tol:
            self._violate(
                table=contract.child,
                check="orphan_rate",
                column=contract.child_key,
                expected=f"{ref:.4f} ±{tol:.2%}",
                observed=f"{orphan_rate:.4f}",
            )

    def _load_unique(self, filename: str, key: str) -> np.ndarray:
        path = self.raw_dir / filename
        df = read_csv_with_fallback(path, usecols=[key])
        return np.unique(df[key].to_numpy(dtype=np.int64, copy=False))

    def _table_filename(self, table_name: str) -> str | None:
        for t in self.tables:
            if t.name == table_name:
                return t.filename
        return None

    # ---- bookkeeping -----------------------------------------------------

    def _record_check(self) -> None:
        self.report.checks_run += 1

    def _violate(
        self,
        table: str,
        check: str,
        column: str | None,
        expected: str,
        observed: str,
        severity: str = "error",
    ) -> None:
        self.report.violations.append(
            Violation(
                table=table,
                check=check,
                column=column,
                expected=expected,
                observed=observed,
                severity=severity,
            )
        )

    def _log_summary(self) -> None:
        logger.info("=" * 70)
        logger.info(
            "Validation complete: %d checks, %d errors, %d warnings",
            self.report.checks_run,
            self.report.error_count,
            self.report.warning_count,
        )
        for v in self.report.violations:
            logger.log(
                logging.ERROR if v.severity == "error" else logging.WARNING,
                "  [%s] %s.%s — expected %s, observed %s",
                v.check,
                v.table,
                v.column or "-",
                v.expected,
                v.observed,
            )
        logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Report persistence
# ---------------------------------------------------------------------------

def save_validation_report(report: ValidationReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    logger.info("Validation report written: %s", out_path)


# ---------------------------------------------------------------------------
# Contract builder CLI helper
# ---------------------------------------------------------------------------

def build_contracts_from_inspection(
    inspection_path: Path,
    contracts_dir: Path,
) -> None:
    """Read an inspection JSON and emit reviewed contract YAMLs."""
    with inspection_path.open(encoding="utf-8") as f:
        inspection = json.load(f)

    builder = ContractBuilder(inspection)
    tables, joins = builder.build_all()
    save_contracts(tables, joins, contracts_dir)