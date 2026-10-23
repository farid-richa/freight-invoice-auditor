"""Load and validate the input CSVs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED = {
    "carriers": ["carrier_id", "carrier_name"],
    "rate_card": ["carrier_id", "origin", "destination", "weight_from_kg",
                  "weight_to_kg", "rate_per_kg", "min_charge"],
    "fuel_surcharge": ["carrier_id", "month", "fuel_pct"],
    "accessorial_contract": ["carrier_id", "code", "amount_eur"],
    "shipments": ["shipment_id", "carrier_id", "ship_date", "origin", "destination",
                  "actual_weight_kg", "volume_m3", "accessorials"],
    "invoice_lines": ["line_id", "invoice_id", "carrier_id", "shipment_id", "invoice_date",
                      "billed_weight_kg", "rate_per_kg", "base_charge", "fuel_pct",
                      "fuel_surcharge", "accessorial_codes", "accessorial_charges",
                      "total_amount"],
}


def load_data(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read every input table and fail early if a column is missing."""
    raw_dir = Path(raw_dir)
    data = {}
    for name, cols in REQUIRED.items():
        df = pd.read_csv(raw_dir / f"{name}.csv", keep_default_na=False, na_values=[""])
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"{name}.csv is missing columns: {sorted(missing)}")
        data[name] = df

    for col in ("accessorials",):
        data["shipments"][col] = data["shipments"][col].fillna("")
    data["invoice_lines"]["accessorial_codes"] = data["invoice_lines"]["accessorial_codes"].fillna("")
    data["shipments"]["ship_date"] = pd.to_datetime(data["shipments"]["ship_date"])
    data["invoice_lines"]["invoice_date"] = pd.to_datetime(data["invoice_lines"]["invoice_date"])
    return data
