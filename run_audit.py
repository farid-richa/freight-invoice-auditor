"""
Audit carrier invoice lines against contract terms.

    python run_audit.py                         # uses data/raw, writes to output/
    python run_audit.py --data path/to/csvs --out path/to/folder

Outputs:
    output/audit_report.xlsx   summary, carrier claims, exception detail, method
    output/audit_results.csv   one row per invoice line (feeds the Power BI model)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from auditor import load_data, run_audit  # noqa: E402
from auditor.report import write_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=ROOT / "data" / "raw", type=Path)
    ap.add_argument("--out", default=ROOT / "output", type=Path)
    args = ap.parse_args()

    t0 = time.perf_counter()
    results = run_audit(load_data(args.data))
    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / "audit_results.csv", index=False)
    xlsx = write_report(results, args.out / "audit_report.xlsx")
    secs = time.perf_counter() - t0

    ex = results[results.status == "EXCEPTION"]
    print(f"Audited {len(results):,} lines in {secs:.1f}s")
    print(f"Exceptions: {len(ex):,} ({len(ex) / len(results):.1%})  "
          f"Recoverable: {results.recoverable_eur.sum():,.2f} EUR")
    print(f"Report: {xlsx}")


if __name__ == "__main__":
    main()
