"""
Core audit logic.

Each invoice line is re-rated from the shipment data and the contract
(rate card, fuel table, agreed accessorials). Any gap between what the
carrier billed and what the contract allows becomes an exception.

Checks, in priority order (a line can fail several; the first one is
reported as the primary exception):

    NO_SHIPMENT           line refers to a shipment we have no record of
    DUPLICATE             shipment already billed on an earlier line
    WEIGHT_OVERBILL       billed weight above chargeable weight
    RATE_ABOVE_CONTRACT   rate per kg above the rate card for that band
    WRONG_FUEL_PCT        fuel % above the carrier's index for the ship month
    UNAGREED_ACCESSORIAL  extra charge not in contract or not on the shipment
    ARITHMETIC            total does not equal the sum of its components
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

VOLUMETRIC_FACTOR = 333      # kg per m3 (road)
WEIGHT_TOLERANCE = 0.01      # 1 % allowed on weight
AMOUNT_TOLERANCE = 0.02      # EUR, absorbs rounding
FUEL_TOLERANCE = 0.0005      # 0.05 percentage points

CHECK_ORDER = [
    "NO_SHIPMENT",
    "DUPLICATE",
    "WEIGHT_OVERBILL",
    "RATE_ABOVE_CONTRACT",
    "WRONG_FUEL_PCT",
    "UNAGREED_ACCESSORIAL",
    "ARITHMETIC",
]


def _rate_lookup(df: pd.DataFrame, rate_card: pd.DataFrame, weight_col: str, prefix: str) -> pd.DataFrame:
    """Attach the contract rate and min charge for the band that weight_col falls in."""
    m = df[["line_id", "carrier_id", "origin", "destination", weight_col]].merge(
        rate_card, on=["carrier_id", "origin", "destination"], how="left"
    )
    m = m[(m[weight_col] >= m.weight_from_kg) & (m[weight_col] < m.weight_to_kg)]
    m = m[["line_id", "rate_per_kg", "min_charge"]].rename(
        columns={"rate_per_kg": f"{prefix}_rate_per_kg", "min_charge": f"{prefix}_min_charge"}
    )
    return df.merge(m, on="line_id", how="left")


def _split(codes: str) -> list[str]:
    return [c for c in str(codes).split("|") if c]


def run_audit(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    inv = data["invoice_lines"].copy()
    shp = data["shipments"]
    rate_card = data["rate_card"]
    fuel = data["fuel_surcharge"]
    acc = data["accessorial_contract"]

    # --- join each line to its shipment -------------------------------------------
    df = inv.merge(
        shp[["shipment_id", "ship_date", "origin", "destination",
             "actual_weight_kg", "volume_m3", "accessorials"]],
        on="shipment_id", how="left",
    )
    df["chk_NO_SHIPMENT"] = df.ship_date.isna()

    # --- duplicates: same shipment billed more than once (keep the earliest) -------
    df = df.sort_values(["invoice_date", "line_id"])
    df["chk_DUPLICATE"] = df.duplicated("shipment_id", keep="first") & ~df.chk_NO_SHIPMENT
    df = df.sort_index()

    # --- weight ----------------------------------------------------------------------
    df["chargeable_weight_kg"] = pd.concat(
        [df.actual_weight_kg, (df.volume_m3 * VOLUMETRIC_FACTOR).round(1)], axis=1
    ).max(axis=1)
    df["chk_WEIGHT_OVERBILL"] = df.billed_weight_kg > df.chargeable_weight_kg * (1 + WEIGHT_TOLERANCE)

    # --- rate: contract rate for the band of the *billed* weight ----------------------
    df = _rate_lookup(df, rate_card, "billed_weight_kg", "billed_band")
    df["chk_RATE_ABOVE_CONTRACT"] = df.rate_per_kg > df.billed_band_rate_per_kg + 1e-4

    # --- fuel -------------------------------------------------------------------------
    df["ship_month"] = df.ship_date.dt.strftime("%Y-%m")
    df = df.merge(
        fuel.rename(columns={"month": "ship_month", "fuel_pct": "contract_fuel_pct"}),
        on=["carrier_id", "ship_month"], how="left",
    )
    df["chk_WRONG_FUEL_PCT"] = df.fuel_pct > df.contract_fuel_pct + FUEL_TOLERANCE

    # --- accessorials -----------------------------------------------------------------
    contract = acc.groupby("carrier_id").apply(
        lambda g: dict(zip(g.code, g.amount_eur)), include_groups=False
    ).to_dict()

    def accessorial_check(row) -> tuple[bool, float, str]:
        if pd.isna(row.ship_date):
            return False, 0.0, ""
        allowed = contract.get(row.carrier_id, {})
        # Count units, not just codes: v1 compared sets and missed an
        # accessorial billed twice when the shipment only had it once.
        on_shipment = Counter(_split(row.accessorials))
        billed = Counter(_split(row.accessorial_codes))
        problems = [
            code for code, n in billed.items()
            if code not in allowed or n > on_shipment.get(code, 0)
        ]
        expected_amt = sum(allowed[c] * n for c, n in on_shipment.items() if c in allowed)
        # amount check catches a correct code billed at a higher price
        over_amount = row.accessorial_charges > expected_amt + AMOUNT_TOLERANCE
        return bool(problems) or over_amount, round(expected_amt, 2), "|".join(problems)

    res = df.apply(accessorial_check, axis=1, result_type="expand")
    df["chk_UNAGREED_ACCESSORIAL"] = res[0].astype(bool)
    df["expected_accessorials"] = res[1].astype(float)
    df["unagreed_codes"] = res[2]

    # --- arithmetic -------------------------------------------------------------------
    components = df.base_charge + df.fuel_surcharge + df.accessorial_charges
    df["chk_ARITHMETIC"] = (df.total_amount - components).abs() > AMOUNT_TOLERANCE

    # --- what the line *should* have cost --------------------------------------------
    df = _rate_lookup(df, rate_card, "chargeable_weight_kg", "contract")
    df["expected_base"] = pd.concat(
        [df.contract_min_charge, df.chargeable_weight_kg * df.contract_rate_per_kg], axis=1
    ).max(axis=1).round(2)
    df["expected_fuel"] = (df.expected_base * df.contract_fuel_pct).round(2)
    df["expected_total"] = (df.expected_base + df.expected_fuel + df.expected_accessorials).round(2)

    # lines that should not exist at all are fully recoverable
    no_pay = df.chk_NO_SHIPMENT | df.chk_DUPLICATE
    df.loc[no_pay, "expected_total"] = 0.0
    df["variance_eur"] = (df.total_amount - df.expected_total).round(2)

    # --- classify ---------------------------------------------------------------------
    chk_cols = [f"chk_{c}" for c in CHECK_ORDER]
    df[chk_cols] = df[chk_cols].fillna(False).astype(bool)
    df["exception_count"] = df[chk_cols].sum(axis=1)
    df["exceptions"] = df[chk_cols].apply(
        lambda r: "|".join(c[4:] for c, v in r.items() if v), axis=1
    )
    df["primary_exception"] = df.exceptions.str.split("|").str[0].replace("", "CLEAN")
    df["status"] = df.exception_count.map(lambda n: "EXCEPTION" if n else "CLEAN")
    # money is only claimed back where the line is an exception and overbilled
    df["recoverable_eur"] = df.variance_eur.where(
        (df.status == "EXCEPTION") & (df.variance_eur > AMOUNT_TOLERANCE), 0.0
    ).round(2)

    df = df.merge(data["carriers"], on="carrier_id", how="left")
    out_cols = [
        "line_id", "invoice_id", "carrier_id", "carrier_name", "shipment_id",
        "invoice_date", "ship_date", "origin", "destination",
        "actual_weight_kg", "volume_m3", "chargeable_weight_kg", "billed_weight_kg",
        "rate_per_kg", "contract_rate_per_kg", "fuel_pct", "contract_fuel_pct",
        "accessorial_codes", "unagreed_codes",
        "base_charge", "fuel_surcharge", "accessorial_charges", "total_amount",
        "expected_base", "expected_fuel", "expected_accessorials", "expected_total",
        "variance_eur", "recoverable_eur",
        "status", "primary_exception", "exceptions", "exception_count",
    ] + chk_cols
    return df[out_cols].sort_values("line_id").reset_index(drop=True)
