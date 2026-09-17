"""
Inspect Home Credit Default Risk CSV files.

Purpose
-------
Ground the project's schema contracts, feature metadata catalogue, and
relational integrity rules in the *actual* data rather than assumptions.

Statistics that feed schema contracts (null counts, min/max, unique
values) are computed over the FULL file, not a sample. Sample-only stats
(mean, std, quantiles) are computed from a bounded head for performance.
This distinction matters: a contract derived from a 200K-row sample of a
27M-row file will fail validation on the full data by construction.

Outputs
-------
- Structured log output (stdout + log file)
- Machine-readable report:
      artifacts/reports/inspection_<timestamp>.json

This script is standalone — it must not import from `src/credit_risk`.
Its output becomes the input to `src/credit_risk/data/validate.py` and
the feature catalogue builder.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("credit_risk.inspect")


def configure_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> None:
    """Configure logging once. Adds a file handler if log_file is provided."""
    if logger.handlers:
        return

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.setLevel(level)
    logger.propagate = False


# ---------------------------------------------------------------------------
# Encoding tolerance
# ---------------------------------------------------------------------------
#
# The Home Credit dataset ships mixed encodings:
#   - The eight modelling CSVs are UTF-8 clean.
#   - HomeCredit_columns_description.csv is Windows-1252 (byte 0x85 appears
#     at position 1283 — the ellipsis character '…' in a description).
#
# Rather than hardcoding one encoding per file, we try a chain. latin-1 is
# the terminal fallback — it maps every byte to a codepoint and never raises.
#
ENCODING_CHAIN: tuple[str, ...] = ("utf-8", "cp1252", "latin-1")


def read_csv_with_fallback(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV, trying each encoding in ENCODING_CHAIN."""
    last_error: Exception | None = None
    for encoding in ENCODING_CHAIN:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
            logger.debug(
                "Encoding '%s' failed for %s — trying next", encoding, path.name
            )
    raise RuntimeError(f"All encodings failed for {path.name}") from last_error


def iter_csv_with_fallback(path: Path, **kwargs: Any) -> Iterator[pd.DataFrame]:
    """
    Chunked CSV read with encoding fallback.

    Yields chunks from the first encoding that succeeds. The decode error
    surfaces on the first chunk request, so we force that request inside
    the try block.
    """
    last_error: Exception | None = None
    for encoding in ENCODING_CHAIN:
        try:
            reader = pd.read_csv(path, encoding=encoding, **kwargs)
            first_chunk = next(iter(reader))
            yield first_chunk
            for chunk in reader:
                yield chunk
            return
        except UnicodeDecodeError as exc:
            last_error = exc
            logger.debug(
                "Encoding '%s' failed for %s — trying next", encoding, path.name
            )
            continue
    raise RuntimeError(f"All encodings failed for {path.name}") from last_error


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InspectionConfig:
    """Resolved configuration for an inspection run."""

    project_root: Path
    raw_dir: Path
    report_dir: Path
    config_path: Path
    files: dict[str, str]
    sample_rows: int = 200_000
    sample_values_per_col: int = 5
    chunk_size: int = 1_000_000
    max_unique_tracking: int = 50

    @property
    def primary_key(self) -> str:
        return "SK_ID_CURR"

    @property
    def target_column(self) -> str:
        return "TARGET"

    def file_path(self, name: str) -> Path:
        return self.raw_dir / self.files[name]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class FileReport:
    """Profile of a single CSV file."""

    name: str
    file: str
    size_bytes: int
    sha256: str
    sampled_rows: int
    full_row_count: int
    n_columns: int
    columns: list[str]
    column_profiles: dict[str, dict[str, Any]]
    key_candidates: list[str] = field(default_factory=list)
    key_uniqueness_full: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "sampled_rows": self.sampled_rows,
            "full_row_count": self.full_row_count,
            "n_columns": self.n_columns,
            "columns": self.columns,
            "column_profiles": self.column_profiles,
            "key_candidates": self.key_candidates,
            "key_uniqueness_full": self.key_uniqueness_full,
        }


@dataclass
class JoinSpec:
    """A single relational integrity check between a child and parent table."""

    child: str
    child_key: str
    parent: str
    parent_key: str


@dataclass
class ColumnFullStats:
    """Contract-relevant statistics computed over the full file."""

    null_count: int
    row_count: int
    min_value: float | None = None
    max_value: float | None = None
    unique_values: tuple[Any, ...] | None = None
    n_unique_exact: int | None = None

    @property
    def null_rate(self) -> float:
        return self.null_count / self.row_count if self.row_count else 0.0


# ---------------------------------------------------------------------------
# Inspector
# ---------------------------------------------------------------------------

@dataclass
class DataInspector:
    """Encapsulates inspection of the Home Credit raw data set."""

    config: InspectionConfig
    file_reports: dict[str, FileReport] = field(default_factory=dict)
    integrity: dict[str, Any] = field(default_factory=dict)
    target: dict[str, Any] = field(default_factory=dict)

    # ---- public API ------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute the full inspection and return the assembled report."""
        self._check_raw_dir()
        self._inspect_all_files()
        self._check_relational_integrity()
        self._check_target_integrity()
        report = self._assemble_report()
        self._write_report(report)
        self._update_config_pointer(report)
        self._log_summary(report)
        return report

    # ---- preconditions ---------------------------------------------------

    def _check_raw_dir(self) -> None:
        if not self.config.raw_dir.exists():
            raise FileNotFoundError(
                f"Raw data directory not found: {self.config.raw_dir}"
            )
        logger.info("Inspecting raw directory: %s", self.config.raw_dir)

    # ---- per-file inspection --------------------------------------------

    def _inspect_all_files(self) -> None:
        for name in self.config.files:
            path = self.config.file_path(name)
            if not path.exists():
                logger.warning("Missing file for '%s': %s", name, path.name)
                continue
            logger.info("Profiling %s (%s)", name, path.name)
            self.file_reports[name] = self._inspect_file(name, path)

    def _inspect_file(self, name: str, path: Path) -> FileReport:
        # Bounded head for sample-only stats
        head = read_csv_with_fallback(
            path, nrows=self.config.sample_rows, low_memory=False
        )
        columns = list(head.columns)

        # Full-data pass
        full_stats = self._compute_full_data_stats(path, columns)
        full_row_count = full_stats["__row_count__"]
        column_stats: dict[str, ColumnFullStats] = full_stats["__columns__"]

        # Merge sample stats with full-data stats
        column_profiles = {
            col: self._profile_column(head[col], column_stats[col])
            for col in columns
        }

        key_candidates = [
            c for c in columns
            if c.upper().startswith("SK_ID")
            or c.upper() in {"SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"}
        ]

        key_uniqueness: dict[str, dict[str, Any]] = {}
        for k in key_candidates:
            st = column_stats.get(k)
            if st is not None and st.n_unique_exact is not None:
                key_uniqueness[k] = {
                    "n_unique": st.n_unique_exact,
                    "n_rows": full_row_count,
                    "is_unique": st.n_unique_exact == full_row_count,
                    "source": "full",
                }
            else:
                key_uniqueness[k] = {
                    "n_unique": int(head[k].nunique()),
                    "n_rows": len(head),
                    "is_unique": bool(head[k].nunique() == len(head)),
                    "source": "sample",
                }

        return FileReport(
            name=name,
            file=path.name,
            size_bytes=path.stat().st_size,
            sha256=self._sha256_of_file(path),
            sampled_rows=len(head),
            full_row_count=full_row_count,
            n_columns=len(columns),
            columns=columns,
            column_profiles=column_profiles,
            key_candidates=key_candidates,
            key_uniqueness_full=key_uniqueness,
        )

    def _count_rows(self, path: Path) -> int:
        count = 0
        for chunk in iter_csv_with_fallback(
            path, usecols=[0], chunksize=self.config.chunk_size, low_memory=False
        ):
            count += len(chunk)
        return count

    @staticmethod
    def _sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()

    # ---- full-data statistics -------------------------------------------

    def _compute_full_data_stats(
        self,
        path: Path,
        columns: list[str],
    ) -> dict[str, Any]:
        """
        Stream the file once, accumulating per-column full-data statistics.

        Returns:
            "__row_count__": int
            "__columns__":   dict[column_name, ColumnFullStats]
        """
        # Determine numeric columns from a tiny probe
        head_probe = read_csv_with_fallback(path, nrows=1000, low_memory=False)
        numeric_cols: set[str] = {
            c for c in columns if pd.api.types.is_numeric_dtype(head_probe[c])
        }

        null_counts: dict[str, int] = {c: 0 for c in columns}
        min_values: dict[str, float] = {c: float("inf") for c in numeric_cols}
        max_values: dict[str, float] = {c: float("-inf") for c in numeric_cols}
        unique_sets: dict[str, set[Any]] = {c: set() for c in columns}
        unique_abandoned: set[str] = set()
        row_count = 0

        cap = self.config.max_unique_tracking

        for chunk in iter_csv_with_fallback(
            path, chunksize=self.config.chunk_size, low_memory=False
        ):
            row_count += len(chunk)
            for col in columns:
                s = chunk[col]

                null_counts[col] += int(s.isna().sum())

                if col in numeric_cols:
                    non_null = s.dropna()
                    if len(non_null):
                        cmin = float(non_null.min())
                        cmax = float(non_null.max())
                        if cmin < min_values[col]:
                            min_values[col] = cmin
                        if cmax > max_values[col]:
                            max_values[col] = cmax

                if col not in unique_abandoned:
                    non_null = s.dropna()
                    unique_sets[col].update(non_null.unique().tolist())
                    if len(unique_sets[col]) > cap:
                        unique_abandoned.add(col)
                        unique_sets[col] = set()

        column_stats: dict[str, ColumnFullStats] = {}
        for col in columns:
            is_numeric = col in numeric_cols
            is_abandoned = col in unique_abandoned

            if is_abandoned:
                unique_values: tuple[Any, ...] | None = None
                n_unique_exact: int | None = None
            else:
                try:
                    ordered = sorted(unique_sets[col])
                except TypeError:
                    ordered = sorted(unique_sets[col], key=repr)
                unique_values = tuple(self._safe_json(v) for v in ordered)
                n_unique_exact = len(ordered)

            column_stats[col] = ColumnFullStats(
                null_count=null_counts[col],
                row_count=row_count,
                min_value=(
                    min_values[col] if is_numeric and min_values[col] != float("inf")
                    else None
                ),
                max_value=(
                    max_values[col] if is_numeric and max_values[col] != float("-inf")
                    else None
                ),
                unique_values=unique_values,
                n_unique_exact=n_unique_exact,
            )

        return {"__row_count__": row_count, "__columns__": column_stats}

    # ---- column profiling -----------------------------------------------

    def _profile_column(
        self,
        sample_series: pd.Series,
        full_stats: ColumnFullStats,
    ) -> dict[str, Any]:
        """
        Merge sample-only stats (mean, std, quantiles) with full-data stats
        (null count, min/max, uniqueness).

        Emits `unique_values` (full set) for low-cardinality columns so
        the contract builder can populate `allowed_values` exactly.
        """
        non_null_sample = sample_series.dropna()

        profile: dict[str, Any] = {
            "dtype": str(sample_series.dtype),
            "null_count": full_stats.null_count,
            "null_rate": round(full_stats.null_rate, 6),
            "n_unique": full_stats.n_unique_exact,
            "n_unique_is_exact": full_stats.n_unique_exact is not None,
        }

        # Emit the full unique set when tracked (bounded by the cap).
        if full_stats.unique_values is not None:
            profile["unique_values"] = list(full_stats.unique_values)

        if pd.api.types.is_numeric_dtype(sample_series) and len(non_null_sample):
            profile["numeric"] = {
                "min": full_stats.min_value,
                "max": full_stats.max_value,
                "mean": self._safe_float(non_null_sample.mean()),
                "std": self._safe_float(non_null_sample.std()),
                "p01": self._safe_float(non_null_sample.quantile(0.01)),
                "p50": self._safe_float(non_null_sample.quantile(0.50)),
                "p99": self._safe_float(non_null_sample.quantile(0.99)),
            }

        # Small preview for human readability
        if full_stats.unique_values is not None:
            preview = list(
                full_stats.unique_values[: self.config.sample_values_per_col]
            )
        else:
            preview = [
                self._safe_json(v)
                for v in non_null_sample.drop_duplicates()
                .head(self.config.sample_values_per_col)
                .tolist()
            ]
        profile["sample_values"] = preview

        return profile

    @staticmethod
    def _safe_float(x: Any) -> float | None:
        if x is None:
            return None
        try:
            f = float(x)
        except (TypeError, ValueError):
            return None
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 6)

    @staticmethod
    def _safe_json(x: Any) -> Any:
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return DataInspector._safe_float(x)
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        try:
            if pd.isna(x):
                return None
        except (TypeError, ValueError):
            pass
        return str(x)

    # ---- key loading & relational integrity -----------------------------

    def _load_unique_keys(self, table: str, key: str) -> np.ndarray:
        path = self.config.file_path(table)
        df = read_csv_with_fallback(path, usecols=[key])
        return np.unique(df[key].to_numpy(dtype=np.int64, copy=False))

    def _duplicate_count(self, table: str, key: str) -> int:
        path = self.config.file_path(table)
        df = read_csv_with_fallback(path, usecols=[key])
        series = df[key]
        return int(len(series) - series.nunique())

    def _check_relational_integrity(self) -> None:
        logger.info("Checking relational integrity")

        app_keys = self._load_unique_keys("application_train", self.config.primary_key)
        bureau_keys = self._load_unique_keys("bureau", "SK_ID_BUREAU")
        prev_keys = self._load_unique_keys("previous_application", "SK_ID_PREV")

        join_specs: list[tuple[JoinSpec, np.ndarray]] = [
            (JoinSpec("bureau", "SK_ID_CURR",
                      "application_train", "SK_ID_CURR"), app_keys),
            (JoinSpec("previous_application", "SK_ID_CURR",
                      "application_train", "SK_ID_CURR"), app_keys),
            (JoinSpec("bureau_balance", "SK_ID_BUREAU",
                      "bureau", "SK_ID_BUREAU"), bureau_keys),
            (JoinSpec("pos_cash_balance", "SK_ID_PREV",
                      "previous_application", "SK_ID_PREV"), prev_keys),
            (JoinSpec("installments_payments", "SK_ID_PREV",
                      "previous_application", "SK_ID_PREV"), prev_keys),
            (JoinSpec("credit_card_balance", "SK_ID_PREV",
                      "previous_application", "SK_ID_PREV"), prev_keys),
        ]

        joins: list[dict[str, Any]] = []
        for spec, parent_keys in join_specs:
            logger.info(
                "  join check: %s.%s → %s.%s",
                spec.child, spec.child_key, spec.parent, spec.parent_key,
            )
            joins.append(self._compute_join_stats(spec, parent_keys))

        self.integrity = {
            "application_train": {
                "n_applicants": int(len(app_keys)),
                "duplicate_sk_id_curr": self._duplicate_count(
                    "application_train", self.config.primary_key
                ),
            },
            "joins": joins,
        }

    def _compute_join_stats(
        self,
        spec: JoinSpec,
        parent_keys: np.ndarray,
    ) -> dict[str, Any]:
        path = self.config.file_path(spec.child)
        df = read_csv_with_fallback(path, usecols=[spec.child_key])
        series = df[spec.child_key]

        child_arr = series.to_numpy(dtype=np.int64, copy=False)
        child_unique = np.unique(child_arr)

        overlap = np.intersect1d(child_unique, parent_keys, assume_unique=True)
        orphans = np.setdiff1d(child_unique, parent_keys, assume_unique=True)

        return {
            "child_table": spec.child,
            "parent_table": spec.parent,
            "child_key": spec.child_key,
            "parent_key": spec.parent_key,
            "child_rows": int(len(child_arr)),
            "child_unique_keys": int(len(child_unique)),
            "orphan_keys": int(len(orphans)),
            "orphan_rate": round(len(orphans) / max(len(child_unique), 1), 6),
            "coverage_of_parent": round(
                len(overlap) / max(len(parent_keys), 1), 6
            ),
        }

    # ---- target integrity -----------------------------------------------

    def _check_target_integrity(self) -> None:
        logger.info("Checking target integrity")
        path = self.config.file_path("application_train")
        df = read_csv_with_fallback(path, usecols=[self.config.target_column])
        series = df[self.config.target_column]

        self.target = {
            "target_column": self.config.target_column,
            "value_counts": {
                str(k): int(v)
                for k, v in series.value_counts(dropna=False).to_dict().items()
            },
            "positive_rate": round(float(series.mean()), 6),
            "null_count": int(series.isna().sum()),
        }

    # ---- report assembly & persistence ----------------------------------

    def _assemble_report(self) -> dict[str, Any]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return {
            "generated_at": timestamp,
            "raw_dir": str(self.config.raw_dir.relative_to(self.config.project_root)),
            "files": {name: rep.to_dict() for name, rep in self.file_reports.items()},
            "relational_integrity": self.integrity,
            "target_integrity": self.target,
        }

    def _report_path(self, timestamp: str) -> Path:
        return self.config.report_dir / f"inspection_{timestamp}.json"

    def _write_report(self, report: dict[str, Any]) -> Path:
        self.config.report_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._report_path(report["generated_at"])
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(
            "Report written: %s",
            out_path.relative_to(self.config.project_root),
        )
        return out_path

    def _update_config_pointer(self, report: dict[str, Any]) -> None:
        with self.config.config_path.open(encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        raw_config["inspection"] = {
            "report_path": str(
                self._report_path(report["generated_at"]).relative_to(
                    self.config.project_root
                )
            ),
            "last_run": report["generated_at"],
        }

        with self.config.config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(raw_config, f, sort_keys=False)

        logger.info(
            "Config pointer updated: %s", self.config.config_path.name
        )

    # ---- summary logging -------------------------------------------------

    def _log_summary(self, report: dict[str, Any]) -> None:
        logger.info("=" * 70)
        logger.info("Files inspected:")
        for name, rep in self.file_reports.items():
            logger.info(
                "  %-24s rows=%12d cols=%d",
                name, rep.full_row_count, rep.n_columns,
            )

        logger.info("Target:")
        logger.info(
            "  positive_rate=%.4f  nulls=%d",
            report["target_integrity"]["positive_rate"],
            report["target_integrity"]["null_count"],
        )

        logger.info("Relational integrity:")
        for join in report["relational_integrity"]["joins"]:
            logger.info(
                "  %-22s -> %-22s orphans=%9d coverage=%.3f",
                join["child_table"],
                join["parent_table"],
                join["orphan_keys"],
                join["coverage_of_parent"],
            )
        logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_config(project_root: Path) -> InspectionConfig:
    config_path = project_root / "configs" / "data_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return InspectionConfig(
        project_root=project_root,
        raw_dir=project_root / raw["paths"]["raw_dir"],
        report_dir=project_root / raw["paths"]["reports_dir"],
        config_path=config_path,
        files=raw["files"],
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    configure_logging(
        log_file=project_root / "artifacts" / "reports" / "inspect.log",
    )

    try:
        config = build_config(project_root)
        DataInspector(config).run()
    except Exception:
        logger.exception("Inspection failed")
        sys.exit(1)


if __name__ == "__main__":
    main()