"""
Unit tests: each rule on a tiny hand-built dataset where the right answer is obvious.

    pytest -q
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auditor import run_audit  # noqa: E402


@pytest.fixture
def base():
    """One carrier, one lane, one shipment of 200 kg billed exactly per contract."""
    carriers = pd.DataFrame([{"carrier_id": "C01", "carrier_name": "Test Carrier"}])
    rate_card = pd.DataFrame([
        dict(carrier_id="C01", origin="VLC", destination="MAD", weight_from_kg=0,
             weight_to_kg=100, rate_per_kg=0.50, min_charge=40.0),
        dict(carrier_id="C01", origin="VLC", destination="MAD", weight_from_kg=100,
             weight_to_kg=999999, rate_per_kg=0.40, min_charge=40.0),
    ])
    fuel = pd.DataFrame([dict(carrier_id="C01", month="2026-03", fuel_pct=0.10)])
    acc = pd.DataFrame([dict(carrier_id="C01", code="TAIL_LIFT", description="", amount_eur=30.0)])
    shipments = pd.DataFrame([dict(
        shipment_id="SH1", carrier_id="C01", ship_date=pd.Timestamp("2026-03-10"),
        origin="VLC", destination="MAD", actual_weight_kg=200.0, volume_m3=0.4,
        accessorials="TAIL_LIFT",
    )])
    # correct bill: 200 kg * 0.40 = 80.00 base, 8.00 fuel, 30.00 tail lift = 118.00
    line = dict(
        line_id="L1", invoice_id="INV1", carrier_id="C01", shipment_id="SH1",
        invoice_date=pd.Timestamp("2026-03-20"), billed_weight_kg=200.0, rate_per_kg=0.40,
        base_charge=80.0, fuel_pct=0.10, fuel_surcharge=8.0,
        accessorial_codes="TAIL_LIFT", accessorial_charges=30.0, total_amount=118.0,
    )
    return dict(carriers=carriers, rate_card=rate_card, fuel_surcharge=fuel,
                accessorial_contract=acc, shipments=shipments, line=line)


def audit(base, **changes):
    line = {**base["line"], **changes}
    data = {k: v for k, v in base.items() if k != "line"}
    data["invoice_lines"] = pd.DataFrame([line])
    return run_audit(data).iloc[0]


def test_clean_line_passes(base):
    r = audit(base)
    assert r.status == "CLEAN"
    assert r.expected_total == pytest.approx(118.0)
    assert r.recoverable_eur == 0


def test_weight_overbill(base):
    r = audit(base, billed_weight_kg=250.0, base_charge=100.0, fuel_surcharge=10.0, total_amount=140.0)
    assert r.primary_exception == "WEIGHT_OVERBILL"
    assert r.recoverable_eur == pytest.approx(22.0)


def test_volumetric_weight_is_not_an_overbill(base):
    # 1.0 m3 * 333 = 333 kg chargeable, so billing 333 kg is correct
    base["shipments"].loc[0, "volume_m3"] = 1.0
    r = audit(base, billed_weight_kg=333.0, base_charge=133.2, fuel_surcharge=13.32, total_amount=176.52)
    assert r.status == "CLEAN"


def test_rate_above_contract(base):
    r = audit(base, rate_per_kg=0.45, base_charge=90.0, fuel_surcharge=9.0, total_amount=129.0)
    assert r.primary_exception == "RATE_ABOVE_CONTRACT"
    assert r.recoverable_eur == pytest.approx(11.0)


def test_wrong_fuel_pct(base):
    r = audit(base, fuel_pct=0.14, fuel_surcharge=11.2, total_amount=121.2)
    assert r.primary_exception == "WRONG_FUEL_PCT"
    assert r.recoverable_eur == pytest.approx(3.2)


def test_accessorial_not_in_contract(base):
    r = audit(base, accessorial_codes="TAIL_LIFT|ADR", accessorial_charges=90.0, total_amount=178.0)
    assert r.primary_exception == "UNAGREED_ACCESSORIAL"
    assert r.unagreed_codes == "ADR"


def test_accessorial_billed_twice(base):
    # the case v1 missed: right code, billed two units for one on the shipment
    r = audit(base, accessorial_codes="TAIL_LIFT|TAIL_LIFT", accessorial_charges=60.0, total_amount=148.0)
    assert r.primary_exception == "UNAGREED_ACCESSORIAL"
    assert r.recoverable_eur == pytest.approx(30.0)


def test_arithmetic(base):
    r = audit(base, total_amount=150.0)
    assert r.primary_exception == "ARITHMETIC"
    assert r.recoverable_eur == pytest.approx(32.0)


def test_no_shipment_is_fully_recoverable(base):
    r = audit(base, shipment_id="SH999")
    assert r.primary_exception == "NO_SHIPMENT"
    assert r.recoverable_eur == pytest.approx(118.0)


def test_duplicate_keeps_first_and_flags_later(base):
    data = {k: v for k, v in base.items() if k != "line"}
    first = base["line"]
    second = {**first, "line_id": "L2", "invoice_id": "INV2", "invoice_date": pd.Timestamp("2026-04-15")}
    data["invoice_lines"] = pd.DataFrame([second, first])  # order in file should not matter
    res = run_audit(data).set_index("line_id")
    assert res.loc["L1", "status"] == "CLEAN"
    assert res.loc["L2", "primary_exception"] == "DUPLICATE"
    assert res.loc["L2", "recoverable_eur"] == pytest.approx(118.0)


def test_min_charge_overbill_has_no_money_impact(base):
    # 30 kg shipment: contract = max(40 min, 30*0.5=15) = 40. Billing 60 kg -> 30 still < 40.
    base["shipments"].loc[0, ["actual_weight_kg", "volume_m3", "accessorials"]] = [30.0, 0.05, ""]
    r = audit(base, billed_weight_kg=60.0, rate_per_kg=0.50, base_charge=40.0, fuel_surcharge=4.0,
              accessorial_codes="", accessorial_charges=0.0, total_amount=44.0)
    assert r.primary_exception == "WEIGHT_OVERBILL"
    assert r.recoverable_eur == 0
