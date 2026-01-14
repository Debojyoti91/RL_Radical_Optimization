from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def motif_enrichment(
    motif_xlsx: str | Path,
    top_csv: str | Path,
    *,
    out_csv: Optional[str | Path] = None,
    name_col: str = "Radical name",
    motif_col: str = "motif_class",
) -> pd.DataFrame:
    """Compute motif-class enrichment between the full dataset and a top subset.

    Parameters
    ----------
    motif_xlsx
        Excel file containing at least [name_col, motif_col] for the full dataset.
    top_csv
        CSV file containing at least [name_col] for the top subset (e.g., top-20).
    out_csv
        If provided, write the enrichment table to this path.
    name_col
        Identifier column for radicals.
    motif_col
        Motif class label column.

    Returns
    -------
    pd.DataFrame with columns:
        motif_class, count_full, pct_full, count_top, pct_top, enrichment
    """
    motif_xlsx = Path(motif_xlsx)
    top_csv = Path(top_csv)

    motif_df = pd.read_excel(motif_xlsx)
    top_df = pd.read_csv(top_csv)

    for df in (motif_df, top_df):
        if name_col not in df.columns:
            raise ValueError(f"Missing required column '{name_col}' in {df.shape} from {motif_xlsx if df is motif_df else top_csv}")
        df[name_col] = df[name_col].astype(str).str.strip()

    if motif_col not in motif_df.columns:
        raise ValueError(f"Missing required column '{motif_col}' in motif table: {motif_xlsx}")

    full_counts = motif_df[motif_col].value_counts(dropna=False)
    top_counts = motif_df[motif_df[name_col].isin(set(top_df[name_col]))][motif_col].value_counts(dropna=False)

    full_pct = full_counts / max(1, int(full_counts.sum()))
    top_pct = top_counts / max(1, int(top_counts.sum()))

    enrichment = (top_pct / full_pct).replace([pd.NA, float("inf"), -float("inf")], 0.0).fillna(0.0)

    out = pd.DataFrame(
        {
            "motif_class": full_counts.index.astype(str),
            "count_full": full_counts.values.astype(int),
            "pct_full": full_pct.values.astype(float),
            "count_top": top_counts.reindex(full_counts.index).fillna(0).astype(int).values,
            "pct_top": top_pct.reindex(full_counts.index).fillna(0).astype(float).values,
            "enrichment": enrichment.reindex(full_counts.index).fillna(0).astype(float).values,
        }
    )

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)

    return out
