#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rlcdft.motifs import motif_enrichment


def parse_args():
    p = argparse.ArgumentParser(description="Compute motif enrichment between full motif table and top subset.")
    p.add_argument("--motif-xlsx", type=str, required=True, help="Excel file with Radical name and motif class columns.")
    p.add_argument("--top-csv", type=str, required=True, help="CSV file with Radical name column for the top subset.")
    p.add_argument("--out-csv", type=str, default="motif_enrichment.csv", help="Output CSV path.")
    p.add_argument("--name-col", type=str, default="Radical name")
    p.add_argument("--motif-col", type=str, default="motif_class")
    return p.parse_args()


def main():
    args = parse_args()
    out = motif_enrichment(
        args.motif_xlsx,
        args.top_csv,
        out_csv=args.out_csv,
        name_col=args.name_col,
        motif_col=args.motif_col,
    )
    print(out.sort_values("enrichment", ascending=False).head(20).to_string(index=False))
    print(f"[OK] Wrote: {Path(args.out_csv).resolve()}")


if __name__ == "__main__":
    main()
