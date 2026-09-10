#!/usr/bin/env python3
"""
Techno-Economic Feasibility Assessment for Battery Energy Storage Systems (BESS).

Evaluates C&I BESS applications across Kenya, Uganda, Tanzania, Rwanda, and DRC
using parametric algebraic models—no external 8760-hour time-series required.

Assumptions:
- Kenya WRMA/levies rolled into 60% effective VAT markup on base energy.
- Uganda declining block: 5% marginal discount above 100,000 kWh/month proxy.
- Benchmark TOU shift: 800 kWh/day = 200 kW × 4 h peak window.
- Unified BESS CAPEX sized on max(kWh) across modules; benefits summed without
  double-counting energy capacity.
- All scorecard/NPV values reported in USD.
- Power factor default 0.90 for peak-shaving energy sizing.

Dependencies: numpy, pandas, matplotlib, tabulate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Global techno-economic benchmarks
# ---------------------------------------------------------------------------

DISCOUNT_RATE = 0.08
BESS_LIFESPAN_YEARS = 10
SOLAR_LIFESPAN_YEARS = 25

BESS_CAPEX_USD_PER_KWH = 300.0
BESS_OM_USD_PER_KWH_YR = 5.0
BESS_ROUND_TRIP_EFF = 0.88
BESS_MAX_DOD = 0.80
BESS_ANNUAL_FADE = 0.015
BESS_ANNUAL_CYCLES = 350

SOLAR_CAPEX_USD_PER_KWP = 800.0
SOLAR_OM_USD_PER_KWP_YR = 15.0
SOLAR_DEGRADATION_YR = 0.005
SOLAR_YIELD_KWH_PER_KWP = 1500.0

DIESEL_CAPEX_USD_PER_KW = 250.0
DIESEL_FUEL_USD_PER_L = 1.40
DIESEL_SFC_L_PER_KWH = 0.28
DIESEL_OM_USD_PER_KWH = 0.02
DIESEL_BACKUP_HRS_YR = 500.0

# Hardcoded levelized costs (USD/kWh) per specification
LCOS_BESS = 0.1881
LCOE_SOLAR = 0.0625
LCOE_SOLAR_BESS = 0.2590
LCOE_DIESEL = 0.4865

UGANDA_DECLINING_BLOCK_THRESHOLD_KWH_MO = 100_000
UGANDA_DECLINING_BLOCK_DISCOUNT = 0.05
TANZANIA_SPP_FOSSIL_CAP = 0.25
KENYA_VAT_MARKUP = 1.60


@dataclass(frozen=True)
class GlobalBenchmarks:
    """Frozen container for global techno-economic constants."""

    discount_rate: float = DISCOUNT_RATE
    bess_lifespan_years: int = BESS_LIFESPAN_YEARS
    solar_lifespan_years: int = SOLAR_LIFESPAN_YEARS
    bess_capex_usd_per_kwh: float = BESS_CAPEX_USD_PER_KWH
    bess_om_usd_per_kwh_yr: float = BESS_OM_USD_PER_KWH_YR
    bess_round_trip_eff: float = BESS_ROUND_TRIP_EFF
    bess_max_dod: float = BESS_MAX_DOD
    bess_annual_fade: float = BESS_ANNUAL_FADE
    bess_annual_cycles: int = BESS_ANNUAL_CYCLES
    solar_capex_usd_per_kwp: float = SOLAR_CAPEX_USD_PER_KWP
    solar_om_usd_per_kwp_yr: float = SOLAR_OM_USD_PER_KWP_YR
    solar_degradation_yr: float = SOLAR_DEGRADATION_YR
    solar_yield_kwh_per_kwp: float = SOLAR_YIELD_KWH_PER_KWP
    diesel_capex_usd_per_kw: float = DIESEL_CAPEX_USD_PER_KW
    diesel_fuel_usd_per_l: float = DIESEL_FUEL_USD_PER_L
    diesel_sfc_l_per_kwh: float = DIESEL_SFC_L_PER_KWH
    diesel_om_usd_per_kwh: float = DIESEL_OM_USD_PER_KWH
    lcos_bess: float = LCOS_BESS
    lcoe_solar: float = LCOE_SOLAR
    lcoe_solar_bess: float = LCOE_SOLAR_BESS
    lcoe_diesel: float = LCOE_DIESEL


BENCHMARKS = GlobalBenchmarks()


@dataclass
class ConsumerProfile:
    """Unified consumer input for all feasibility modules."""

    country: Optional[str]
    tier: str
    delta_kva: float = 500.0
    spike_duration_min: float = 30.0
    power_factor: float = 0.90
    daily_energy_shift_kwh: float = 800.0
    kenya_itou_baseline_kwh: float = 400.0
    itou_compliance_factor: float = 1.0
    backup_kw: float = 500.0
    outage_hours: Optional[float] = None
    avg_outage_duration_h: float = 4.0
    event_cost_usd: float = 25_000.0
    n_outages: Optional[float] = None
    transformer_kva: float = 750.0
    use_solar_bess: bool = True
    monthly_energy_kwh: Optional[float] = None


@dataclass
class ModuleResult:
    """Result container for a single analysis module."""

    module_name: str
    annual_benefit_usd: float
    capex_usd: float
    bess_kwh: float
    simple_payback_yr: float
    metrics: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


@dataclass
class FeasibilityResult:
    """Aggregated feasibility output for one country/profile."""

    country: str
    peak_shaving: ModuleResult
    tou_arbitrage: ModuleResult
    diesel_displacement: ModuleResult
    industrial_ups: ModuleResult
    unified_bess_kwh: float
    total_capex_usd: float
    total_npv_usd: float
    blended_payback_yr: float
    annual_benefit_usd: float
    flags: list[str] = field(default_factory=list)


@dataclass
class CountryConfig:
    """Per-country tariff, reliability, and regulatory configuration."""

    name: str
    fx_local_per_usd: float
    currency_code: str
    demand_charges_local: dict[str, float]
    energy_tariffs_local: dict[str, float]
    tou_windows: dict[str, tuple[int, int]]
    reliability: dict[str, float]
    regulatory: dict[str, Any]
    tier_map: dict[str, str]
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Country registry
# ---------------------------------------------------------------------------

COUNTRIES: dict[str, CountryConfig] = {
    "Kenya": CountryConfig(
        name="Kenya",
        fx_local_per_usd=130.0,
        currency_code="KES",
        demand_charges_local={
            "CI1": 1100.0,
            "CI2": 700.0,
            "CI3": 370.0,
            "CI4": 300.0,
            "CI5": 300.0,
            "CI6": 200.0,
        },
        energy_tariffs_local={
            "base_kwh": 13.74,
            "fuel_surcharge_kwh": 3.14,
            "forex_kwh": 0.715,
            "peak_effective": (13.74 + 3.14 + 0.715) * KENYA_VAT_MARKUP,
            "offpeak_itou_base": 13.74 * 0.5 * KENYA_VAT_MARKUP,
        },
        tou_windows={
            "offpeak": (22, 6),
            "peak": (6, 22),
        },
        reliability={"saidi_hrs": 18.0, "saifi": 12.0},
        regulatory={"itou_incremental_only": True},
        tier_map={
            "medium_industrial": "CI2",
            "CI1": "CI1",
            "CI2": "CI2",
            "CI3": "CI3",
        },
        notes=[
            "Frequent transmission cascade trips (Suswa node bottlenecks).",
            "ITOU: 50% off-peak discount only on incremental consumption above baseline.",
        ],
    ),
    "Uganda": CountryConfig(
        name="Uganda",
        fx_local_per_usd=3700.0,
        currency_code="UGX",
        demand_charges_local={
            "20": 0.0,
            "30.1": 0.0,
            "40": 0.0,
        },
        energy_tariffs_local={
            "peak_kwh": 378.2,
            "shoulder_kwh": 308.1,
            "offpeak_kwh": 239.3,
        },
        tou_windows={
            "peak": (18, 0),
            "shoulder": (6, 18),
            "offpeak": (0, 6),
        },
        reliability={"saidi_hrs": 15.0, "saifi": 10.0},
        regulatory={"declining_block_threshold_kwh": UGANDA_DECLINING_BLOCK_THRESHOLD_KWH_MO},
        tier_map={
            "medium_industrial": "30.1",
            "20": "20",
            "30.1": "30.1",
            "40": "40",
        },
        notes=["Declining block discount for consumption > 100,000 kWh/month."],
    ),
    "Tanzania": CountryConfig(
        name="Tanzania",
        fx_local_per_usd=2600.0,
        currency_code="TZS",
        demand_charges_local={
            "T1": 0.0,
            "T2": 15004.0,
            "T3-MV": 13200.0,
        },
        energy_tariffs_local={
            "T2_kwh": 195.0,
            "T3_kwh": 195.0,
        },
        tou_windows={
            "peak": (18, 22),
            "offpeak": (22, 6),
        },
        reliability={"saidi_hrs": 28.2, "saifi": 17.69},
        regulatory={"spp_fossil_cap": TANZANIA_SPP_FOSSIL_CAP},
        tier_map={
            "medium_industrial": "T3-MV",
            "T1": "T1",
            "T2": "T2",
            "T3-MV": "T3-MV",
        },
        notes=[
            "SPP 2025: hybrid fossil backup capped at 25% of installed capacity.",
            "SAIDI = 28.2 hrs/yr; SAIFI = 17.69 interruptions/yr.",
        ],
    ),
    "Rwanda": CountryConfig(
        name="Rwanda",
        fx_local_per_usd=1350.0,
        currency_code="Frw",
        demand_charges_local={
            "small_industrial_peak": 11017.0,
            "small_industrial_shoulder": 4008.0,
            "small_industrial_offpeak": 0.0,
            "medium_industrial_peak": 10514.0,
            "medium_industrial_shoulder": 3588.0,
            "medium_industrial_offpeak": 0.0,
            "large_industrial_peak": 7184.0,
            "large_industrial_shoulder": 2004.0,
            "large_industrial_offpeak": 0.0,
        },
        energy_tariffs_local={
            "peak_kwh": 127.0,
            "shoulder_kwh": 113.0,
            "offpeak_kwh": 89.0,
        },
        tou_windows={
            "peak": (18, 23),
            "shoulder": (8, 18),
            "offpeak": (23, 8),
        },
        reliability={"saidi_hrs": 1.48, "saifi": 2.5},
        regulatory={},
        tier_map={
            "medium_industrial": "medium_industrial",
            "small_industrial": "small_industrial",
            "large_industrial": "large_industrial",
        },
        notes=[
            "Time-differentiated demand charges: peak/shoulder/off-peak by volume tier.",
            "Highly stable grid (SAIDI = 1.48 hrs/yr).",
        ],
    ),
    "DRC": CountryConfig(
        name="DRC",
        fx_local_per_usd=1.0,
        currency_code="USD",
        demand_charges_local={
            "industrial": 0.0,
            "industrial_mining": 0.0,
        },
        energy_tariffs_local={
            "base_kwh": 0.085,
            "accessible_kwh": 0.085,
        },
        tou_windows={
            "peak": (8, 22),
            "offpeak": (22, 8),
        },
        reliability={"saidi_hrs": 3500.0, "saifi": 120.0},
        regulatory={"grid_deficit_pct": 0.50, "structural_deficit": True},
        tier_map={
            "medium_industrial": "industrial",
            "industrial": "industrial",
            "industrial_mining": "industrial_mining",
        },
        notes=[
            ">50% structural grid deficit (2,100 MW vs 4,500 MW peak).",
            "Heavy reliance on bilateral USD contracts or diesel/HFO backup.",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def to_usd(local_amount: float, country: str) -> float:
    """Convert local currency amount to USD using embedded FX rates."""
    cfg = COUNTRIES[country]
    return local_amount / cfg.fx_local_per_usd


def resolve_tier(country: str, tier: str) -> str:
    """Map generic tier label to country-specific tariff code."""
    cfg = COUNTRIES[country]
    return cfg.tier_map.get(tier, tier)


def get_demand_charge_usd_per_kva_month(country: str, tier: str) -> float:
    """Return monthly demand charge in USD/kVA for country/tier."""
    cfg = COUNTRIES[country]
    resolved = resolve_tier(country, tier)

    if country == "Rwanda":
        key = f"{resolved}_peak"
        local = cfg.demand_charges_local.get(key, cfg.demand_charges_local["medium_industrial_peak"])
        return to_usd(local, country)

    if country == "Kenya":
        local = cfg.demand_charges_local.get(resolved, cfg.demand_charges_local["CI2"])
        return to_usd(local, country)

    if country == "Tanzania":
        local = cfg.demand_charges_local.get(resolved, cfg.demand_charges_local["T3-MV"])
        return to_usd(local, country)

    return 0.0


def get_tou_tariffs_usd(country: str, tier: str) -> tuple[float, float, float]:
    """
    Return (peak, shoulder, offpeak) energy tariffs in USD/kWh.

    Shoulder equals peak where not applicable.
    """
    cfg = COUNTRIES[country]
    t = cfg.energy_tariffs_local

    if country == "Kenya":
        peak = to_usd(t["peak_effective"], country)
        offpeak = to_usd(t["offpeak_itou_base"], country)
        return peak, peak, offpeak

    if country == "Uganda":
        peak = to_usd(t["peak_kwh"], country)
        shoulder = to_usd(t["shoulder_kwh"], country)
        offpeak = to_usd(t["offpeak_kwh"], country)
        return peak, shoulder, offpeak

    if country == "Tanzania":
        resolved = resolve_tier(country, tier)
        key = "T3_kwh" if "T3" in resolved else "T2_kwh"
        rate = to_usd(t.get(key, t["T2_kwh"]), country)
        return rate * 1.15, rate, rate * 0.85

    if country == "Rwanda":
        peak = to_usd(t["peak_kwh"], country)
        shoulder = to_usd(t["shoulder_kwh"], country)
        offpeak = to_usd(t["offpeak_kwh"], country)
        return peak, shoulder, offpeak

    # DRC
    base = t.get("base_kwh", 0.085)
    return base, base, base * 0.90


def npv(cashflows: list[float], rate: float = DISCOUNT_RATE) -> float:
    """Compute net present value of cashflow series (year 0 = index 0)."""
    return float(sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows)))


def verify_lcos_table() -> pd.DataFrame:
    """Sanity-check embedded LCOS/LCOE values and return summary table."""
    diesel_fuel = DIESEL_FUEL_USD_PER_L * DIESEL_SFC_L_PER_KWH
    diesel_lcoe_calc = diesel_fuel + DIESEL_OM_USD_PER_KWH

    rows = [
        ("BESS LCOS", LCOS_BESS, LCOS_BESS, "Hardcoded benchmark"),
        ("Solar PV LCOE", LCOE_SOLAR, LCOE_SOLAR, "Hardcoded benchmark"),
        ("Solar+BESS LCOE", LCOE_SOLAR_BESS, LCOE_SOLAR_BESS, "Hardcoded benchmark"),
        ("Diesel LCOE", LCOE_DIESEL, diesel_lcoe_calc, "Fuel + O&M check"),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Specified ($/kWh)", "Computed ($/kWh)", "Note"])
    return df


# ---------------------------------------------------------------------------
# Module 1: Peak Shaving
# ---------------------------------------------------------------------------


def analyze_peak_shaving(profile: ConsumerProfile, country: str) -> ModuleResult:
    """
    Peak shaving / demand charge management analysis.

    Annual savings = ΔkVA × C_demand × 12
    E_BESS = (ΔkVA × PF × duration/60) / (DoD × η)
    """
    b = BENCHMARKS
    c_demand = get_demand_charge_usd_per_kva_month(country, profile.tier)
    annual_savings = profile.delta_kva * c_demand * 12.0

    e_bess = (
        profile.delta_kva
        * profile.power_factor
        * (profile.spike_duration_min / 60.0)
        / (b.bess_max_dod * b.bess_round_trip_eff)
    )
    capex = e_bess * b.bess_capex_usd_per_kwh
    payback = capex / annual_savings if annual_savings > 0 else float("inf")

    transformer = profile.transformer_kva or profile.delta_kva * 1.25
    thermal_relief = profile.delta_kva / transformer

    flags: list[str] = []
    if country == "Rwanda":
        flags.append("Rwanda: High Peak TOU Demand Penalty (peak kVA charge applies)")
    if country == "Kenya" and resolve_tier(country, profile.tier) == "CI1":
        flags.append("Kenya CI1: Highest LV demand charge tier")

    return ModuleResult(
        module_name="Peak Shaving",
        annual_benefit_usd=annual_savings,
        capex_usd=capex,
        bess_kwh=e_bess,
        simple_payback_yr=payback,
        metrics={
            "demand_charge_usd_per_kva_month": c_demand,
            "annual_demand_savings_usd": annual_savings,
            "bess_energy_kwh": e_bess,
            "transformer_thermal_relief_score": thermal_relief,
        },
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Module 2: TOU Arbitrage
# ---------------------------------------------------------------------------


def analyze_tou_arbitrage(profile: ConsumerProfile, country: str) -> ModuleResult:
    """
    Time-of-use arbitrage / load shifting analysis.

    Margin = tariff_peak - (tariff_offpeak / η_ACAC)
    Annual profit = E_shift_effective × (margin - LCOS) × 350 days
    """
    b = BENCHMARKS
    peak_usd, shoulder_usd, offpeak_usd = get_tou_tariffs_usd(country, profile.tier)
    tariff_peak = peak_usd
    tariff_offpeak = offpeak_usd

    e_shift = profile.daily_energy_shift_kwh
    e_shift_effective = e_shift
    flags: list[str] = []

    if country == "Kenya":
        eligible = max(0.0, e_shift - profile.kenya_itou_baseline_kwh)
        e_shift_effective = eligible * profile.itou_compliance_factor
        flags.append("Kenya: ITOU Baseline Hurdle (discount on incremental off-peak only)")
        if e_shift_effective < e_shift * 0.5:
            flags.append(f"Kenya ITOU: only {e_shift_effective:.0f}/{e_shift:.0f} kWh/day eligible")

    if country == "Uganda":
        monthly_proxy = e_shift * 30.0
        if monthly_proxy > UGANDA_DECLINING_BLOCK_THRESHOLD_KWH_MO:
            tariff_peak *= 1.0 - UGANDA_DECLINING_BLOCK_DISCOUNT
            flags.append("Uganda: Declining block discount applied to peak rate")

    margin = tariff_peak - (tariff_offpeak / b.bess_round_trip_eff)
    unit_profit = margin - b.lcos_bess
    annual_profit = e_shift_effective * unit_profit * b.bess_annual_cycles

    if margin < b.lcos_bess:
        flags.append(f"Negative arbitrage: margin ${margin:.4f}/kWh < LCOS ${b.lcos_bess:.4f}/kWh")

    e_bess = e_shift_effective / b.bess_max_dod if e_shift_effective > 0 else 0.0
    capex = e_bess * b.bess_capex_usd_per_kwh
    payback = capex / annual_profit if annual_profit > 0 else float("inf")

    return ModuleResult(
        module_name="TOU Arbitrage",
        annual_benefit_usd=max(0.0, annual_profit),
        capex_usd=capex,
        bess_kwh=e_bess,
        simple_payback_yr=payback,
        metrics={
            "tariff_peak_usd_kwh": tariff_peak,
            "tariff_offpeak_usd_kwh": tariff_offpeak,
            "net_margin_usd_kwh": margin,
            "unit_profit_usd_kwh": unit_profit,
            "daily_shift_effective_kwh": e_shift_effective,
            "annual_arbitrage_profit_usd": annual_profit,
        },
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Module 3: Diesel Displacement
# ---------------------------------------------------------------------------


def analyze_diesel_displacement(profile: ConsumerProfile, country: str) -> ModuleResult:
    """
    Diesel displacement and LCOE comparison.

    Savings = H_outage × P_backup × (LCOE_diesel - LCOE_alt)
    """
    b = BENCHMARKS
    cfg = COUNTRIES[country]

    h_outage = profile.outage_hours
    if h_outage is None:
        h_outage = cfg.reliability["saidi_hrs"]

    annual_energy = h_outage * profile.backup_kw
    lcoe_alt = b.lcoe_solar_bess if profile.use_solar_bess else b.lcos_bess
    savings_per_kwh = b.lcoe_diesel - lcoe_alt
    annual_savings = annual_energy * savings_per_kwh

    e_storage = profile.backup_kw * profile.avg_outage_duration_h
    if profile.use_solar_bess:
        solar_capex = profile.backup_kw * b.solar_capex_usd_per_kwp
        bess_capex = e_storage * b.bess_capex_usd_per_kwh
        capex = solar_capex + bess_capex
    else:
        capex = e_storage * b.bess_capex_usd_per_kwh

    payback = capex / annual_savings if annual_savings > 0 else float("inf")
    flags: list[str] = []

    if country == "Tanzania":
        diesel_utilization = h_outage / 8760.0
        if diesel_utilization > TANZANIA_SPP_FOSSIL_CAP:
            haircut = TANZANIA_SPP_FOSSIL_CAP / diesel_utilization
            annual_savings *= haircut
            flags.append(
                f"Tanzania SPP 2025: fossil cap haircut ({haircut:.0%} of savings retained)"
            )

    if country == "DRC":
        flags.append("DRC: Highest Diesel Displacement Feasibility (structural grid deficit)")

    if country in ("DRC", "Tanzania") and annual_savings > 100_000:
        flags.append(f"{country}: High diesel displacement savings potential")

    return ModuleResult(
        module_name="Diesel Displacement",
        annual_benefit_usd=annual_savings,
        capex_usd=capex,
        bess_kwh=e_storage,
        simple_payback_yr=payback,
        metrics={
            "outage_hours_yr": h_outage,
            "annual_backup_energy_kwh": annual_energy,
            "lcoe_diesel_usd_kwh": b.lcoe_diesel,
            "lcoe_alternative_usd_kwh": lcoe_alt,
            "savings_per_kwh_usd": savings_per_kwh,
            "annual_diesel_savings_usd": annual_savings,
        },
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Module 4: Industrial UPS
# ---------------------------------------------------------------------------


def analyze_industrial_ups(profile: ConsumerProfile, country: str) -> ModuleResult:
    """
    Industrial UPS / downtime loss prevention.

    Value = N_outages × C_event (BESS 0 ms transfer vs diesel 30-60 s latency).
    """
    b = BENCHMARKS
    cfg = COUNTRIES[country]

    n_outages = profile.n_outages
    if n_outages is None:
        n_outages = cfg.reliability["saifi"]

    annual_value = n_outages * profile.event_cost_usd

    ride_through_min = 5.0
    e_ups = (
        profile.backup_kw
        * (ride_through_min / 60.0)
        / (b.bess_max_dod * b.bess_round_trip_eff)
    )
    capex = e_ups * b.bess_capex_usd_per_kwh
    payback = capex / annual_value if annual_value > 0 else float("inf")

    flags = [
        "BESS inverter: ~0 ms transfer vs diesel genset 30-60 s start latency",
    ]
    if country == "Kenya":
        flags.append("Kenya: Suswa cascade trips elevate process-interruption risk")

    return ModuleResult(
        module_name="Industrial UPS",
        annual_benefit_usd=annual_value,
        capex_usd=capex,
        bess_kwh=e_ups,
        simple_payback_yr=payback,
        metrics={
            "n_outages_yr": n_outages,
            "event_cost_usd": profile.event_cost_usd,
            "annual_avoided_loss_usd": annual_value,
            "ups_bess_kwh_informational": e_ups,
        },
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Consolidated feasibility summary
# ---------------------------------------------------------------------------


def compute_feasibility_summary(profile: ConsumerProfile, country: str) -> FeasibilityResult:
    """
    Aggregate all modules into unified BESS sizing and 10-year NPV cashflow model.

    Unified BESS sized on max(kWh) across modules; benefits summed without
    double-counting CAPEX.
    """
    b = BENCHMARKS

    ps = analyze_peak_shaving(profile, country)
    tou = analyze_tou_arbitrage(profile, country)
    diesel = analyze_diesel_displacement(profile, country)
    ups = analyze_industrial_ups(profile, country)

    unified_bess_kwh = max(ps.bess_kwh, tou.bess_kwh, diesel.bess_kwh, ups.bess_kwh)
    bess_capex = unified_bess_kwh * b.bess_capex_usd_per_kwh

    solar_capex = 0.0
    if profile.use_solar_bess and diesel.annual_benefit_usd > 0:
        solar_capex = profile.backup_kw * b.solar_capex_usd_per_kwp

    total_capex = bess_capex + solar_capex

    annual_benefit = (
        ps.annual_benefit_usd
        + tou.annual_benefit_usd
        + diesel.annual_benefit_usd
        + ups.annual_benefit_usd
    )

    cashflows = [-total_capex]
    for year in range(1, b.bess_lifespan_years + 1):
        fade_factor = (1.0 - b.bess_annual_fade) ** (year - 1)
        gross = annual_benefit * fade_factor
        om = b.bess_om_usd_per_kwh_yr * unified_bess_kwh * fade_factor
        if profile.use_solar_bess:
            om += b.solar_om_usd_per_kwp_yr * profile.backup_kw * fade_factor
        cashflows.append(gross - om)

    total_npv = npv(cashflows, b.discount_rate)
    blended_payback = total_capex / annual_benefit if annual_benefit > 0 else float("inf")

    all_flags = ps.flags + tou.flags + diesel.flags + ups.flags
    all_flags.extend(COUNTRIES[country].notes[:1])

    return FeasibilityResult(
        country=country,
        peak_shaving=ps,
        tou_arbitrage=tou,
        diesel_displacement=diesel,
        industrial_ups=ups,
        unified_bess_kwh=unified_bess_kwh,
        total_capex_usd=total_capex,
        total_npv_usd=total_npv,
        blended_payback_yr=blended_payback,
        annual_benefit_usd=annual_benefit,
        flags=list(dict.fromkeys(all_flags)),
    )


def format_module_table(result: FeasibilityResult) -> str:
    """Format module-level breakdown as a tabulate string."""
    rows = [
        [
            result.peak_shaving.module_name,
            f"${result.peak_shaving.annual_benefit_usd:,.0f}",
            f"${result.peak_shaving.capex_usd:,.0f}",
            f"{result.peak_shaving.bess_kwh:.1f}",
            f"{result.peak_shaving.simple_payback_yr:.2f}",
        ],
        [
            result.tou_arbitrage.module_name,
            f"${result.tou_arbitrage.annual_benefit_usd:,.0f}",
            f"${result.tou_arbitrage.capex_usd:,.0f}",
            f"{result.tou_arbitrage.bess_kwh:.1f}",
            f"{result.tou_arbitrage.simple_payback_yr:.2f}",
        ],
        [
            result.diesel_displacement.module_name,
            f"${result.diesel_displacement.annual_benefit_usd:,.0f}",
            f"${result.diesel_displacement.capex_usd:,.0f}",
            f"{result.diesel_displacement.bess_kwh:.1f}",
            f"{result.diesel_displacement.simple_payback_yr:.2f}",
        ],
        [
            result.industrial_ups.module_name,
            f"${result.industrial_ups.annual_benefit_usd:,.0f}",
            f"${result.industrial_ups.capex_usd:,.0f}",
            f"{result.industrial_ups.bess_kwh:.1f}",
            f"{result.industrial_ups.simple_payback_yr:.2f}",
        ],
    ]
    headers = ["Module", "Annual Benefit (USD)", "Module CAPEX (USD)", "BESS kWh", "Payback (yr)"]
    return tabulate(rows, headers=headers, tablefmt="grid")


def print_feasibility_report(result: FeasibilityResult, title: str) -> None:
    """Print a full feasibility report for one profile/country."""
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")
    print(format_module_table(result))
    print(f"\nUnified BESS sizing: {result.unified_bess_kwh:.1f} kWh")
    print(f"Total CAPEX (unified): ${result.total_capex_usd:,.0f}")
    print(f"Total annual benefit:  ${result.annual_benefit_usd:,.0f}")
    print(f"Blended simple payback: {result.blended_payback_yr:.2f} years")
    print(f"10-year NPV (@ {DISCOUNT_RATE:.0%}): ${result.total_npv_usd:,.0f}")

    tou_margin = result.tou_arbitrage.metrics.get("net_margin_usd_kwh", 0.0)
    print(f"TOU net margin: ${tou_margin:.4f}/kWh")

    if result.flags:
        print("\nQualitative flags:")
        for flag in result.flags:
            print(f"  - {flag}")


# ---------------------------------------------------------------------------
# Cross-country comparison
# ---------------------------------------------------------------------------

BENCHMARK_PROFILE = ConsumerProfile(
    country=None,
    tier="medium_industrial",
    delta_kva=500,
    spike_duration_min=30,
    daily_energy_shift_kwh=800,
    backup_kw=500,
    outage_hours=None,
    event_cost_usd=25_000,
    transformer_kva=750,
)

COUNTRY_TIER_DEFAULTS = {
    "Kenya": "CI2",
    "Uganda": "30.1",
    "Tanzania": "T3-MV",
    "Rwanda": "medium_industrial",
    "DRC": "industrial",
}


def run_cross_country_comparison(profile: ConsumerProfile) -> pd.DataFrame:
    """
    Evaluate a standardized industrial profile across all five countries.

    Returns a comparative feasibility scorecard DataFrame (USD reporting).
    """
    scorecard_rows: list[dict[str, Any]] = []

    for country in COUNTRIES:
        p = ConsumerProfile(
            country=country,
            tier=COUNTRY_TIER_DEFAULTS[country],
            delta_kva=profile.delta_kva,
            spike_duration_min=profile.spike_duration_min,
            power_factor=profile.power_factor,
            daily_energy_shift_kwh=profile.daily_energy_shift_kwh,
            kenya_itou_baseline_kwh=profile.kenya_itou_baseline_kwh,
            itou_compliance_factor=profile.itou_compliance_factor,
            backup_kw=profile.backup_kw,
            outage_hours=profile.outage_hours,
            avg_outage_duration_h=profile.avg_outage_duration_h,
            event_cost_usd=profile.event_cost_usd,
            n_outages=profile.n_outages,
            transformer_kva=profile.transformer_kva,
            use_solar_bess=profile.use_solar_bess,
        )
        result = compute_feasibility_summary(p, country)

        tou_margin = result.tou_arbitrage.metrics.get("net_margin_usd_kwh", 0.0)
        flags_str = "; ".join(result.flags[:3])

        scorecard_rows.append(
            {
                "Country": country,
                "Peak Payback (yr)": round(result.peak_shaving.simple_payback_yr, 2),
                "TOU Net Margin ($/kWh)": round(tou_margin, 4),
                "Diesel Savings ($/yr)": round(result.diesel_displacement.annual_benefit_usd, 0),
                "Total NPV ($)": round(result.total_npv_usd, 0),
                "Flags": flags_str,
            }
        )

    df = pd.DataFrame(scorecard_rows)

    print(f"\n{'=' * 72}")
    print("CROSS-COUNTRY COMPARATIVE FEASIBILITY SCORECARD (USD)")
    print(f"{'=' * 72}")
    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))

    print("\nQualitative benchmark summary:")
    print("  - DRC: Highest Diesel Displacement Feasibility (structural grid deficit)")
    print("  - Rwanda: High Peak TOU Demand Penalty (time-differentiated kVA charges)")
    print("  - Kenya: ITOU Baseline Hurdle limits off-peak arbitrage eligibility")
    print("  - Tanzania: Elevated SAIDI drives backup displacement value")
    print("  - Uganda: Strong TOU spread for Large Industrial Code 30.1")

    _plot_cross_country_chart(df)
    return df


def _plot_cross_country_chart(df: pd.DataFrame) -> None:
    """Generate grouped bar chart comparing NPV and peak payback across countries."""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    x = np.arange(len(df))
    width = 0.35

    npv_vals = df["Total NPV ($)"].values / 1e6
    bars1 = ax1.bar(x - width / 2, npv_vals, width, label="NPV (M USD)", color="#2ecc71")
    ax1.set_ylabel("NPV (Million USD)")
    ax1.set_xlabel("Country")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Country"].values)

    ax2 = ax1.twinx()
    payback = df["Peak Payback (yr)"].values
    bars2 = ax2.bar(x + width / 2, payback, width, label="Peak Payback (yr)", color="#3498db")
    ax2.set_ylabel("Peak Shaving Payback (years)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("BESS Feasibility: Cross-Country Comparison")
    fig.tight_layout()

    chart_path = "bess_cross_country_comparison.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"\nChart saved: {chart_path}")


# ---------------------------------------------------------------------------
# Demo test cases
# ---------------------------------------------------------------------------

CASE_A_KENYA = ConsumerProfile(
    country="Kenya",
    tier="CI1",
    delta_kva=350,
    spike_duration_min=45,
    daily_energy_shift_kwh=600,
    kenya_itou_baseline_kwh=400,
    itou_compliance_factor=0.85,
    backup_kw=250,
    outage_hours=12,
    event_cost_usd=85_000,
    n_outages=8,
    transformer_kva=500,
)

CASE_B_DRC = ConsumerProfile(
    country="DRC",
    tier="industrial_mining",
    delta_kva=800,
    spike_duration_min=60,
    daily_energy_shift_kwh=2000,
    backup_kw=2000,
    outage_hours=3500,
    event_cost_usd=150_000,
    n_outages=120,
    transformer_kva=2500,
)


def main() -> None:
    """Run LCOS verification, demo cases, and cross-country benchmark."""
    print("BESS Techno-Economic Feasibility Assessment")
    print("East Africa: Kenya | Uganda | Tanzania | Rwanda | DRC")
    print(f"Discount rate: {DISCOUNT_RATE:.0%} | BESS lifespan: {BESS_LIFESPAN_YEARS} yr")

    lcos_df = verify_lcos_table()
    print("\nEmbedded LCOS/LCOE Verification:")
    print(tabulate(lcos_df, headers="keys", tablefmt="grid", showindex=False))

    result_a = compute_feasibility_summary(CASE_A_KENYA, "Kenya")
    print_feasibility_report(
        result_a,
        "CASE A: Plastics Extrusion Factory — Nairobi, Kenya (CI1)",
    )

    result_b = compute_feasibility_summary(CASE_B_DRC, "DRC")
    print_feasibility_report(
        result_b,
        "CASE B: Mining Dewatering & Processing — Katanga, DRC",
    )

    run_cross_country_comparison(BENCHMARK_PROFILE)


if __name__ == "__main__":
    main()
