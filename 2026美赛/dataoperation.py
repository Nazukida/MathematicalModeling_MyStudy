"""Utilities for computing Score_soft and D_4 from abilities CSV files.

Usage (CLI):
  python dataoperation.py --path 2026美赛 --pattern "abilities_*.csv" --max-score 100

Functions:
  - compute_score_soft(paths): returns average Importance across given CSV files
  - compute_d4(score_soft, max_score): returns D_4 = score_soft / max_score
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Iterable, List

try:
	import pandas as pd
except Exception:  # pragma: no cover - pandas usually available
	pd = None


def _read_importances_from_csv(path: str, importance_col: str = "Importance") -> List[float]:
	if pd is not None:
		df = pd.read_csv(path)
		if importance_col not in df.columns:
			return []
		series = pd.to_numeric(df[importance_col], errors="coerce").dropna()
		return series.astype(float).tolist()

	# fallback without pandas
	import csv

	vals: List[float] = []
	with open(path, newline='', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		for row in reader:
			v = row.get(importance_col)
			try:
				if v is not None and v != "":
					vals.append(float(v))
			except ValueError:
				continue
	return vals


def compute_score_soft(paths: Iterable[str], importance_col: str = "Importance") -> float:
	"""Compute Score_soft as the mean of the `importance_col` values across CSV files.

	Args:
		paths: iterable of CSV file paths.
		importance_col: name of the numeric column to average (default: 'Importance').

	Returns:
		mean importance (float). If no numeric values found, returns 0.0.
	"""
	values: List[float] = []
	for p in paths:
		values.extend(_read_importances_from_csv(p, importance_col=importance_col))

	if not values:
		return 0.0
	return sum(values) / len(values)


def compute_score_soft_per_file(paths: Iterable[str], importance_col: str = "Importance") -> dict:
	"""Compute Score_soft per file. Returns mapping path -> mean importance."""
	out = {}
	for p in paths:
		vals = _read_importances_from_csv(p, importance_col=importance_col)
		out[p] = (sum(vals) / len(vals)) if vals else 0.0
	return out


def compute_d4(score_soft: float, max_score: float) -> float:
	"""Compute D_4 = Score_soft / Max_Score. If max_score == 0, returns 0.0."""
	try:
		if max_score == 0:
			return 0.0
		return float(score_soft) / float(max_score)
	except Exception:
		return 0.0


def find_csvs_in_dir(directory: str, pattern: str = "abilities_*.csv") -> List[str]:
	p = Path(directory)
	if not p.exists():
		return []
	return [str(x) for x in p.glob(pattern) if x.is_file()]


def main(argv: List[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Compute Score_soft and D_4 from abilities CSV files.")
	parser.add_argument("--path", default='.', help="Directory containing CSV files (default: current dir)")
	parser.add_argument("--pattern", default='abilities_*.csv', help="Glob pattern for CSV files")
	parser.add_argument("--max-score", type=float, default=100.0, help="Maximum score for normalization (default: 100)")
	parser.add_argument("--col", default='Importance', help="CSV column name to average (default: Importance)")
	args = parser.parse_args(argv)

	csvs = find_csvs_in_dir(args.path, args.pattern)
	if not csvs:
		print(f"No CSV files found in {args.path!r} matching pattern {args.pattern!r}")
		return 2

	per_file = compute_score_soft_per_file(csvs, importance_col=args.col)

	print(f"Files: {len(csvs)} found")
	for fp in csvs:
		rel = os.path.relpath(fp)
		score = per_file.get(fp, 0.0)
		d4 = compute_d4(score, args.max_score)
		print(f"  - {rel}")
		print(f"      Score_soft (mean {args.col}): {score:.6f}")
		print(f"      D_4 = Score_soft / Max_Score: {d4:.6f}")

	# also print overall mean across all values for convenience
	overall = compute_score_soft(csvs, importance_col=args.col)
	print(f"Overall Score_soft (all files mean {args.col}): {overall:.6f}")
	print(f"Overall D_4 = {compute_d4(overall, args.max_score):.6f}")
	return 0


if __name__ == '__main__':
	raise SystemExit(main())

