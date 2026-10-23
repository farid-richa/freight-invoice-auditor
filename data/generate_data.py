"""
Generate a synthetic freight-billing dataset for the auditor.

Everything here is fictional: carrier names, lanes, rates and shipments are
random. The goal is a realistic *shape* of data (contracted rate card, fuel
surcharge table, agreed accessorials, shipments and carrier invoices) with
known billing errors planted on purpose, so the auditor can be measured
against a ground truth.

Usage:
    python data/generate_data.py            # writes CSVs to data/raw/
    python data/generate_data.py --seed 7   # different but reproducible run
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent / "raw"

ZONES = ["VLC", "MAD", "BCN", "SEV", "BIO", "ZAZ"]
WEIGHT_BANDS = [(0, 100), (100, 500), (500, 1000), (1000, 5000), (5000, 999999)]  # last band open-ended
BAND_DISCOUNT = [1.00, 0.70, 0.52, 0.38, 0.27]  # price per kg falls with weight
VOLUMETRIC_FACTOR = 333  # kg per m3, common for road freight

CARRIERS = [
    ("C01", "Levante Express (fictional)"),
    ("C02", "Meseta Haulage (fictional)"),
    ("C03", "Norte Cargo (fictional)"),
    ("C04", "Costa Freight (fictional)"),
    ("C05", "Rio Transportes (fictional)"),
    ("C06", "Sierra Logistics (fictional)"),
]

ACCESSORIALS = {  # code: (description, base amount EUR)
    "TAIL_LIFT": ("Tail-lift delivery", 35.0),
    "WAITING_HR": ("Waiting time per hour", 45.0),
    "ADR": ("Dangerous goods handling", 60.0),
    "RES_DELIV": ("Residential delivery", 25.0),
}

ERROR_TYPES = [
    "WEIGHT_OVERBILL",
    "RATE_ABOVE_CONTRACT",
    "WRONG_FUEL_PCT",
    "UNAGREED_ACCESSORIAL",
    "ARITHMETIC",
    "DUPLICATE",
    "NO_SHIPMENT",
]


def build_rate_card(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for carrier_id, _ in CARRIERS:
        carrier_factor = rng.uniform(0.9, 1.15)
        for o in ZONES:
            for d in ZONES:
                if o == d:
                    continue
                base_per_kg = rng.uniform(0.22, 0.38) * carrier_factor
                min_charge = round(rng.uniform(38, 65) * carrier_factor, 2)
                for (lo, hi), disc in zip(WEIGHT_BANDS, BAND_DISCOUNT):
                    rows.append(
                        dict(
                            carrier_id=carrier_id,
                            origin=o,
                            destination=d,
                            weight_from_kg=lo,
                            weight_to_kg=hi,
                            rate_per_kg=round(base_per_kg * disc, 4),
                            min_charge=min_charge,
                        )
                    )
    return pd.DataFrame(rows)


def build_fuel_table(rng: np.random.Generator, months: list[str]) -> pd.DataFrame:
    rows = []
    for carrier_id, _ in CARRIERS:
        level = rng.uniform(0.11, 0.16)
        for m in months:
            level = float(np.clip(level + rng.normal(0, 0.006), 0.08, 0.20))
            rows.append(dict(carrier_id=carrier_id, month=m, fuel_pct=round(level, 4)))
    return pd.DataFrame(rows)


def build_accessorial_contract(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for carrier_id, _ in CARRIERS:
        # each carrier has 3 of the 4 accessorials in its contract
        codes = rng.choice(list(ACCESSORIALS), size=3, replace=False)
        for code in codes:
            desc, amt = ACCESSORIALS[code]
            rows.append(
                dict(
                    carrier_id=carrier_id,
                    code=code,
                    description=desc,
                    amount_eur=round(amt * rng.uniform(0.9, 1.15), 2),
                )
            )
    return pd.DataFrame(rows)


def build_shipments(rng, n, acc_contract) -> pd.DataFrame:
    carrier_ids = [c for c, _ in CARRIERS]
    dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.integers(0, 181, n), unit="D")
    origins = rng.choice(ZONES, n)
    dests = np.array([rng.choice([z for z in ZONES if z != o]) for o in origins])
    # weight: mix of parcels-ish pallets and full loads, log-normal
    weight = np.round(np.clip(rng.lognormal(mean=5.6, sigma=1.3, size=n), 5, 23500), 1)
    # density varies: some light/bulky shipments go volumetric
    density = rng.choice([450, 300, 180], size=n, p=[0.65, 0.25, 0.10])
    volume = np.round(weight / density, 3)

    carriers = rng.choice(carrier_ids, n, p=[0.24, 0.20, 0.18, 0.15, 0.13, 0.10])
    allowed = acc_contract.groupby("carrier_id")["code"].apply(list).to_dict()
    acc = []
    for c in carriers:
        used = [code for code in allowed[c] if rng.random() < 0.08]
        acc.append("|".join(used))

    return pd.DataFrame(
        dict(
            shipment_id=[f"SH{100000 + i}" for i in range(n)],
            carrier_id=carriers,
            ship_date=dates.date,
            origin=origins,
            destination=dests,
            actual_weight_kg=weight,
            volume_m3=volume,
            accessorials=acc,
        )
    )


def correct_charges(ship: pd.Series, rate_card, fuel, acc_contract) -> dict:
    """What the carrier *should* bill for one shipment, per contract."""
    chargeable = max(ship.actual_weight_kg, round(ship.volume_m3 * VOLUMETRIC_FACTOR, 1))
    lane = rate_card[
        (rate_card.carrier_id == ship.carrier_id)
        & (rate_card.origin == ship.origin)
        & (rate_card.destination == ship.destination)
        & (rate_card.weight_from_kg <= chargeable)
        & (rate_card.weight_to_kg > chargeable)
    ].iloc[0]
    base = round(max(lane.min_charge, chargeable * lane.rate_per_kg), 2)
    month = str(ship.ship_date)[:7]
    pct = fuel[(fuel.carrier_id == ship.carrier_id) & (fuel.month == month)].fuel_pct.iloc[0]
    fuel_amt = round(base * pct, 2)
    codes = [c for c in ship.accessorials.split("|") if c]
    acc_amt = 0.0
    for code in codes:
        acc_amt += acc_contract[
            (acc_contract.carrier_id == ship.carrier_id) & (acc_contract.code == code)
        ].amount_eur.iloc[0]
    return dict(
        billed_weight_kg=chargeable,
        rate_per_kg=lane.rate_per_kg,
        base_charge=base,
        fuel_pct=pct,
        fuel_surcharge=fuel_amt,
        accessorial_codes="|".join(codes),
        accessorial_charges=round(acc_amt, 2),
    )


def build_invoices(rng, shipments, rate_card, fuel, acc_contract, error_rate):
    lines, truth = [], []
    months = sorted(fuel.month.unique())
    contract_codes = acc_contract.groupby("carrier_id")["code"].apply(set).to_dict()

    for _, s in shipments.iterrows():
        c = correct_charges(s, rate_card, fuel, acc_contract)
        error = None
        if rng.random() < error_rate:
            error = rng.choice(
                ["WEIGHT_OVERBILL", "RATE_ABOVE_CONTRACT", "WRONG_FUEL_PCT",
                 "UNAGREED_ACCESSORIAL", "ARITHMETIC"],
                p=[0.28, 0.22, 0.20, 0.18, 0.12],
            )

        if error == "WEIGHT_OVERBILL":
            c["billed_weight_kg"] = round(c["billed_weight_kg"] * rng.uniform(1.08, 1.35), 1)
            lane = rate_card[
                (rate_card.carrier_id == s.carrier_id)
                & (rate_card.origin == s.origin)
                & (rate_card.destination == s.destination)
                & (rate_card.weight_from_kg <= c["billed_weight_kg"])
                & (rate_card.weight_to_kg > c["billed_weight_kg"])
            ].iloc[0]
            c["rate_per_kg"] = lane.rate_per_kg
            c["base_charge"] = round(max(lane.min_charge, c["billed_weight_kg"] * lane.rate_per_kg), 2)
            c["fuel_surcharge"] = round(c["base_charge"] * c["fuel_pct"], 2)
        elif error == "RATE_ABOVE_CONTRACT":
            c["rate_per_kg"] = round(c["rate_per_kg"] * rng.uniform(1.06, 1.25), 4)
            c["base_charge"] = round(max(c["base_charge"] * 1.06,
                                         c["billed_weight_kg"] * c["rate_per_kg"]), 2)
            c["fuel_surcharge"] = round(c["base_charge"] * c["fuel_pct"], 2)
        elif error == "WRONG_FUEL_PCT":
            m = str(s.ship_date)[:7]
            # carrier applies a higher % (e.g. a later month's index or a flat rate)
            c["fuel_pct"] = round(c["fuel_pct"] + rng.uniform(0.02, 0.06), 4)
            c["fuel_surcharge"] = round(c["base_charge"] * c["fuel_pct"], 2)
        elif error == "UNAGREED_ACCESSORIAL":
            not_in_contract = set(ACCESSORIALS) - contract_codes[s.carrier_id]
            extra = rng.choice(sorted(not_in_contract)) if rng.random() < 0.5 else "WAITING_HR"
            codes = [x for x in c["accessorial_codes"].split("|") if x]
            if extra in codes:  # already legitimately used, bill an extra unit instead
                extra = "WAITING_HR"
            codes.append(extra)
            c["accessorial_codes"] = "|".join(codes)
            c["accessorial_charges"] = round(c["accessorial_charges"] + ACCESSORIALS[extra][1], 2)

        total = round(c["base_charge"] + c["fuel_surcharge"] + c["accessorial_charges"], 2)
        if error == "ARITHMETIC":
            total = round(total + rng.uniform(15, 120), 2)

        line = dict(
            shipment_id=s.shipment_id,
            carrier_id=s.carrier_id,
            invoice_date=(pd.Timestamp(s.ship_date) + pd.Timedelta(days=int(rng.integers(5, 25)))).date(),
            billed_weight_kg=c["billed_weight_kg"],
            rate_per_kg=c["rate_per_kg"],
            base_charge=c["base_charge"],
            fuel_pct=c["fuel_pct"],
            fuel_surcharge=c["fuel_surcharge"],
            accessorial_codes=c["accessorial_codes"],
            accessorial_charges=c["accessorial_charges"],
            total_amount=total,
        )
        lines.append(line)
        truth.append(error or "CLEAN")

    inv = pd.DataFrame(lines)
    inv["truth"] = truth

    # duplicates: re-bill some clean lines on a later invoice
    clean_idx = inv.index[inv.truth == "CLEAN"]
    dup = inv.loc[rng.choice(clean_idx, size=int(len(inv) * 0.012), replace=False)].copy()
    dup["invoice_date"] = [d + pd.Timedelta(days=int(rng.integers(20, 45))) for d in pd.to_datetime(dup.invoice_date)]
    dup["invoice_date"] = pd.to_datetime(dup["invoice_date"]).dt.date
    dup["truth"] = "DUPLICATE"

    # phantom lines: shipment ids that do not exist
    ph = inv.sample(n=int(len(inv) * 0.005), random_state=int(rng.integers(1e9))).copy()
    ph["shipment_id"] = [f"SH{900000 + i}" for i in range(len(ph))]
    ph["truth"] = "NO_SHIPMENT"

    inv = pd.concat([inv, dup, ph], ignore_index=True)
    inv = inv.sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)

    # group lines into monthly invoices per carrier
    inv["invoice_month"] = pd.to_datetime(inv.invoice_date).dt.strftime("%Y%m")
    inv["invoice_id"] = "INV-" + inv.carrier_id + "-" + inv.invoice_month
    inv["line_id"] = [f"L{1000000 + i}" for i in range(len(inv))]
    inv = inv.drop(columns="invoice_month")

    truth_df = inv[["line_id", "truth"]].copy()
    inv = inv.drop(columns="truth")
    cols = ["line_id", "invoice_id", "carrier_id", "shipment_id", "invoice_date",
            "billed_weight_kg", "rate_per_kg", "base_charge", "fuel_pct", "fuel_surcharge",
            "accessorial_codes", "accessorial_charges", "total_amount"]
    return inv[cols], truth_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shipments", type=int, default=5000)
    ap.add_argument("--error-rate", type=float, default=0.12)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    RAW.mkdir(parents=True, exist_ok=True)

    months = [f"2026-{m:02d}" for m in range(1, 7)]
    carriers = pd.DataFrame(CARRIERS, columns=["carrier_id", "carrier_name"])
    rate_card = build_rate_card(rng)
    fuel = build_fuel_table(rng, months)
    acc = build_accessorial_contract(rng)
    shipments = build_shipments(rng, args.shipments, acc)
    invoices, truth = build_invoices(rng, shipments, rate_card, fuel, acc, args.error_rate)

    carriers.to_csv(RAW / "carriers.csv", index=False)
    rate_card.to_csv(RAW / "rate_card.csv", index=False)
    fuel.to_csv(RAW / "fuel_surcharge.csv", index=False)
    acc.to_csv(RAW / "accessorial_contract.csv", index=False)
    shipments.to_csv(RAW / "shipments.csv", index=False)
    invoices.to_csv(RAW / "invoice_lines.csv", index=False)
    truth.to_csv(RAW / "ground_truth.csv", index=False)

    print(f"shipments: {len(shipments):,}  invoice lines: {len(invoices):,}")
    print(truth.truth.value_counts().to_string())


if __name__ == "__main__":
    main()
