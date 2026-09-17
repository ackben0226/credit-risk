"""
Feature assembly: join all aggregated tables into modelling matrices.

This module performs the final structural step before transformation:
left-join every per-applicant aggregated table onto the base application
table, producing a single wide matrix with one row per applicant.

    application_train.parquet (base)
            │  LEFT JOIN on SK_ID_CURR
            ├── bureau_aggregated.parquet
            ├── previous_application_aggregated.parquet
            ├── pos_cash_aggregated.parquet
            ├── installments_aggregated.parquet
            └── credit_card_aggregated.parquet
            │
            ▼
    data/processed/application_train_features.parquet

    application_test.parquet (base, no TARGET)
            │  same joins
            ▼
    data/processed/application_test_features.parquet

Design notes
------------
- Left joins preserve every applicant in the base table. Applicants
  absent from an aggregated family receive NaN for that family's
  features.
- Presence indicators (has_*_history) are filled with 0 for applicants
  absent from the family. These are the only columns modified after the
  join.
- All other nulls are preserved. Missingness is informative and is
  handled explicitly in Stage C (feature transformation).
- No imputation, no encoding, no feature selection, no splits. Assembly
  is a structural operation only.
- Column order is deterministic: SK_ID_CURR, TARGET (train only), base
  table columns alphabetically, aggregated features alphabetically. This
  makes the output diffable.

Outputs
-------
- data/processed/application_train_features.parquet
- data/processed/application_test_features.parquet
- artifacts/reports/assembled_feature_catalogue.json
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


logger = logging.getLogger("credit_risk.features.assembly")


# ---------------------------------------------------------------------------
# Known aggregated tables and their presence indicators
# ---------------------------------------------------------------------------
#
# Each tuple is (aggregated_filename, presence_indicator_column).
# The presence indicator is filled with 0 after the left-join for
# applicants absent from the family.
#
AGGREGATED_TABLES: tuple[tuple[str, str], ...] = (
    ("bureau_aggregated.parquet", "has_bureau_history"),
    ("previous_application_aggregated.parquet", "has_previous_application"),
    ("pos_cash_aggregated.parquet", "has_pos_cash_history"),
    ("installments_aggregated.parquet", "has_installment_history"),
    ("credit_card_aggregated.parquet", "has_credit_card_history"),
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AssemblyConfig:
    """Resolved configuration for an assembly run."""

    project_root: Path
    interim_dir: Path
    processed_dir: Path
    reports_dir: Path

    train_file: str = "application_train.parquet"
    test_file: str = "application_test.parquet"
    target_column: str = "TARGET"
    key_column: str = "SK_ID_CURR"

    train_output_file: str = "application_train_features.parquet"
    test_output_file: str = "application_test_features.parquet"
    catalogue_output_file: str = "assembled_feature_catalogue.json"

    # Feature catalogue files emitted by each aggregator
    catalogue_inputs: tuple[str, ...] = (
        "bureau_aggregated_features.json",
        "previous_application_aggregated_features.json",
        "pos_cash_aggregated_features.json",
        "installments_aggregated_features.json",
        "credit_card_aggregated_features.json",
    )

    def train_path(self) -> Path:
        return self.interim_dir / self.train_file

    def test_path(self) -> Path:
        return self.interim_dir / self.test_file

    def aggregated_path(self, filename: str) -> Path:
        return self.interim_dir / filename

    def train_output_path(self) -> Path:
        return self.processed_dir / self.train_output_file

    def test_output_path(self) -> Path:
        return self.processed_dir / self.test_output_file

    def catalogue_output_path(self) -> Path:
        return self.reports_dir / self.catalogue_output_file

    def catalogue_input_path(self, filename: str) -> Path:
        return self.reports_dir / filename


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class FeatureSource:
    """Metadata about one feature's origin."""

    name: str
    source: str
    aggregation: str
    description: str
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "aggregation": self.aggregation,
            "description": self.description,
            "dtype": self.dtype,
        }


@dataclass
class AssemblyReport:
    """Outcome of an assembly run."""

    generated_at: str
    train_input_rows: int
    train_output_rows: int
    train_output_columns: int
    test_input_rows: int
    test_output_rows: int
    test_output_columns: int
    tables_joined: int
    features_from_base: int
    features_from_aggregated: int
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "train": {
                "input_rows": self.train_input_rows,
                "output_rows": self.train_output_rows,
                "output_columns": self.train_output_columns,
            },
            "test": {
                "input_rows": self.test_input_rows,
                "output_rows": self.test_output_rows,
                "output_columns": self.test_output_columns,
            },
            "tables_joined": self.tables_joined,
            "features_from_base": self.features_from_base,
            "features_from_aggregated": self.features_from_aggregated,
            "duration_seconds": round(self.duration_seconds, 3),
        }


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------

@dataclass
class Assembler:
    """
    Joins all aggregated tables onto the base application table.

    Left joins preserve every applicant in the base table. Presence
    indicators are filled with 0 for missing families.
    """

    config: AssemblyConfig
    report: AssemblyReport | None = None

    # ---- public API ------------------------------------------------------

    def run(self) -> AssemblyReport:
        start = time.monotonic()

        self.config.processed_dir.mkdir(parents=True, exist_ok=True)
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)

        # Load base tables
        train = self._load(self.config.train_path(), "application_train")
        test = self._load(self.config.test_path(), "application_test")

        # Load aggregated tables
        aggregated = self._load_all_aggregated()

        # Assemble train
        logger.info("Assembling application_train features")
        train_features = self._assemble_one(train, aggregated, is_train=True)

        # Assemble test
        logger.info("Assembling application_test features")
        test_features = self._assemble_one(test, aggregated, is_train=False)

        # Write outputs
        self._write(train_features, self.config.train_output_path())
        self._write(test_features, self.config.test_output_path())

        # Build and write catalogue
        catalogue = self._build_catalogue(train_features, test_features)
        self._write_catalogue(catalogue)

        duration = time.monotonic() - start
        self.report = self._build_report(
            train_input=train,
            train_output=train_features,
            test_input=test,
            test_output=test_features,
            duration=duration,
        )
        self._log_summary(self.report)
        return self.report

    # ---- input loading ---------------------------------------------------

    def _load(self, path: Path, name: str) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(
                f"Input not found: {path}. Run the earlier pipeline "
                "stages (ingestion, aggregation) first."
            )
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info("  loaded %s: %d rows, %d cols", name, len(df), len(df.columns))
        return df

    def _load_all_aggregated(self) -> dict[str, pd.DataFrame]:
        aggregated: dict[str, pd.DataFrame] = {}
        for filename, _ in AGGREGATED_TABLES:
            path = self.config.aggregated_path(filename)
            name = filename.replace("_aggregated.parquet", "")
            aggregated[filename] = self._load(path, name)
        return aggregated

    # ---- assembly --------------------------------------------------------

    def _assemble_one(
        self,
        base: pd.DataFrame,
        aggregated: dict[str, pd.DataFrame],
        *,
        is_train: bool,
    ) -> pd.DataFrame:
        """
        Left-join all aggregated tables onto the base table.

        Presence indicators are filled with 0 for applicants absent from
        a family. All other nulls are preserved.
        """
        # Validate base has the key
        if self.config.key_column not in base.columns:
            raise KeyError(
                f"Base table missing required key '{self.config.key_column}'"
            )

        result = base.copy()
        base_columns = set(result.columns)
        base_feature_count = len(base_columns)

        # TARGET is expected only in train
        if is_train and self.config.target_column not in result.columns:
            raise KeyError(
                f"Train base table missing target '{self.config.target_column}'"
            )
        if not is_train and self.config.target_column in result.columns:
            logger.warning(
                "Test base table unexpectedly contains '%s'; dropping",
                self.config.target_column,
            )
            result = result.drop(columns=[self.config.target_column])

        # Left-join each aggregated table
        for filename, presence_indicator in AGGREGATED_TABLES:
            agg = aggregated[filename]
            name = filename.replace("_aggregated.parquet", "")

            logger.info("    joining %s", name)

            # Guard against duplicate key column in aggregated table
            if self.config.key_column not in agg.columns:
                raise KeyError(
                    f"Aggregated table '{name}' missing key "
                    f"'{self.config.key_column}'"
                )

            # Compute row count before and after for diagnostics
            n_before = len(result)

            result = result.merge(
                agg,
                on=self.config.key_column,
                how="left",
                validate="one_to_one",
                suffixes=("", f"_{name}"),
            )

            if len(result) != n_before:
                raise RuntimeError(
                    f"Row count changed after joining '{name}': "
                    f"{n_before} → {len(result)}. Duplicate keys in the "
                    "aggregated table."
                )

            # Fill presence indicator with 0 for absent applicants
            if presence_indicator in result.columns:
                result[presence_indicator] = (
                    result[presence_indicator].fillna(0).astype("int8")
                )
            else:
                logger.warning(
                    "    presence indicator '%s' not found after join "
                    "of '%s'", presence_indicator, name
                )

        # Reorder columns deterministically
        result = self._reorder_columns(result, base_columns, is_train=is_train)

        return result

    def _reorder_columns(
        self,
        df: pd.DataFrame,
        base_columns: set[str],
        *,
        is_train: bool,
    ) -> pd.DataFrame:
        """
        Deterministic ordering:
            1. SK_ID_CURR
            2. TARGET (train only)
            3. Base table columns alphabetically
            4. Aggregated feature columns alphabetically
        """
        key = self.config.key_column
        target = self.config.target_column

        base_cols = sorted(c for c in base_columns if c != key and c != target)
        agg_cols = sorted(c for c in df.columns if c not in base_columns)

        ordered = [key]
        if is_train and target in df.columns:
            ordered.append(target)
        ordered.extend(base_cols)
        ordered.extend(agg_cols)

        # Guard: every column must be accounted for exactly once
        if set(ordered) != set(df.columns):
            missing = set(df.columns) - set(ordered)
            extra = set(ordered) - set(df.columns)
            raise RuntimeError(
                f"Column reordering mismatch. Missing: {missing}. "
                f"Extra: {extra}."
            )

        return df[ordered]

    # ---- output writing --------------------------------------------------

    def _write(self, df: pd.DataFrame, path: Path) -> None:
        df.to_parquet(
            path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )
        logger.info(
            "  wrote %s: %d rows, %d cols",
            path.name, len(df), len(df.columns),
        )

    # ---- catalogue -------------------------------------------------------

    def _build_catalogue(
        self,
        train_features: pd.DataFrame,
        test_features: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Consolidate all aggregator catalogues into a master catalogue.

        The base table's columns are catalogued generically (source:
        application_train, aggregation: raw). Aggregated features come
        from the per-aggregator catalogues emitted by each module.
        """
        # Load aggregator catalogues
        aggregated_features: list[FeatureSource] = []
        for catalogue_file in self.config.catalogue_inputs:
            path = self.config.catalogue_input_path(catalogue_file)
            if not path.exists():
                logger.warning("Catalogue not found: %s", path.name)
                continue
            with path.open(encoding="utf-8") as f:
                catalogue = json.load(f)
            for feat in catalogue.get("features", []):
                aggregated_features.append(
                    FeatureSource(
                        name=feat["name"],
                        source=feat.get("source", ""),
                        aggregation=feat.get("aggregation", ""),
                        description=feat.get("description", ""),
                        dtype=feat.get("dtype", ""),
                    )
                )

        # Base columns: catalogue generically
        key = self.config.key_column
        target = self.config.target_column
        base_features: list[FeatureSource] = []
        for col in train_features.columns:
            if col == key or col == target:
                continue
            if col in {f.name for f in aggregated_features}:
                continue
            base_features.append(
                FeatureSource(
                    name=col,
                    source="application_train",
                    aggregation="raw",
                    description="",
                    dtype=str(train_features[col].dtype),
                )
            )

        return {
            "generated_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "train_shape": list(train_features.shape),
            "test_shape": list(test_features.shape),
            "n_base_features": len(base_features),
            "n_aggregated_features": len(aggregated_features),
            "base_features": [f.to_dict() for f in base_features],
            "aggregated_features": [f.to_dict() for f in aggregated_features],
        }

    def _write_catalogue(self, catalogue: dict[str, Any]) -> Path:
        out_path = self.config.catalogue_output_path()
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(catalogue, f, indent=2, default=str)
        logger.info("Consolidated feature catalogue written: %s", out_path.name)
        return out_path

    # ---- reporting -------------------------------------------------------

    def _build_report(
        self,
        train_input: pd.DataFrame,
        train_output: pd.DataFrame,
        test_input: pd.DataFrame,
        test_output: pd.DataFrame,
        duration: float,
    ) -> AssemblyReport:
        base_cols = set(train_input.columns)
        agg_cols = set(train_output.columns) - base_cols

        return AssemblyReport(
            generated_at=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            train_input_rows=len(train_input),
            train_output_rows=len(train_output),
            train_output_columns=len(train_output.columns),
            test_input_rows=len(test_input),
            test_output_rows=len(test_output),
            test_output_columns=len(test_output.columns),
            tables_joined=len(AGGREGATED_TABLES),
            features_from_base=len(base_cols),
            features_from_aggregated=len(agg_cols),
            duration_seconds=duration,
        )

    def _log_summary(self, report: AssemblyReport) -> None:
        logger.info("=" * 70)
        logger.info("Feature assembly complete")
        logger.info("  train input rows:        %d", report.train_input_rows)
        logger.info("  train output rows:       %d", report.train_output_rows)
        logger.info("  train output columns:    %d", report.train_output_columns)
        logger.info("  test input rows:         %d", report.test_input_rows)
        logger.info("  test output rows:        %d", report.test_output_rows)
        logger.info("  test output columns:     %d", report.test_output_columns)
        logger.info("  tables joined:           %d", report.tables_joined)
        logger.info("  base features:           %d", report.features_from_base)
        logger.info("  aggregated features:     %d", report.features_from_aggregated)
        logger.info("  duration:                %.2fs", report.duration_seconds)
        logger.info("=" * 70)